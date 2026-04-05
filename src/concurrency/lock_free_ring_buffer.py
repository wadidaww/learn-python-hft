"""
src/concurrency/lock_free_ring_buffer.py
==========================================
Single-Producer Single-Consumer (SPSC) ring buffer using shared memory.

HFT relevance
-------------
A ring buffer (circular buffer) is the fundamental data structure for
zero-copy, lock-free IPC between a producer and a consumer.  It is used
everywhere in HFT:

  - LMAX Disruptor (Java) — the canonical ring buffer for trading systems
  - Linux kernel `io_uring` — ring buffer for async I/O
  - DPDK rte_ring — lock-free multi-producer ring buffer
  - Aeron messaging — persistent ring buffer over shared memory

For SPSC (single producer, single consumer), no locks are needed:
  - Producer reads `tail`, writes to `buffer[tail % size]`, increments `tail`
  - Consumer reads `head`, reads `buffer[head % size]`, increments `head`
  - The invariant: `head <= tail` and `tail - head <= size`

This implementation uses `multiprocessing.shared_memory` (Python 3.8+)
for cross-process communication without serialisation overhead.

Limitations of this Python implementation:
  - `ctypes` atomics are NOT memory-barrier safe on all architectures.
    For true lock-free behaviour, you need C-level atomic stores/loads.
  - This is pedagogical: in production, use a C extension or C++ with
    `std::atomic<uint64_t>` and `memory_order_release/acquire`.

TODO: Implement a true atomic version using cffi and __atomic_store_n.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import struct
import time
from multiprocessing.shared_memory import SharedMemory


# ---------------------------------------------------------------------------
# Ring buffer layout in shared memory
# ---------------------------------------------------------------------------
# Memory layout (bytes):
#   [0:8]    head index (uint64, written by consumer)
#   [8:16]   tail index (uint64, written by producer)
#   [16:...]  circular data buffer (SLOT_SIZE × CAPACITY bytes)
#
# Using fixed-size slots avoids variable-length framing overhead.

SLOT_SIZE = 64       # bytes per message slot (cache-line aligned)
CAPACITY = 4096      # number of slots (power of 2 for fast modulo via mask)
_MASK = CAPACITY - 1
_HEADER_SIZE = 16    # bytes for head + tail indices


class SPSCRingBuffer:
    """Single-Producer Single-Consumer ring buffer over shared memory.

    Designed for one writer process/thread and one reader process/thread.
    Uses `multiprocessing.shared_memory` for zero-copy cross-process IPC.

    Args:
        name:    Shared memory block name.  Pass None to create; pass the
                 name returned by `shm_name` to attach.
        create:  True to allocate a new block, False to attach to existing.

    Example (producer side)::

        rb = SPSCRingBuffer(name=None, create=True)
        rb.try_write(b"hello world")
        # pass rb.shm_name to consumer process

    Example (consumer side)::

        rb = SPSCRingBuffer(name="psm_abc123", create=False)
        data = rb.try_read()
    """

    def __init__(self, name: str | None, create: bool = True) -> None:
        total_size = _HEADER_SIZE + SLOT_SIZE * CAPACITY
        if create:
            self._shm = SharedMemory(name=name, create=True, size=total_size)
            # Initialise indices to zero
            struct.pack_into("QQ", self._shm.buf, 0, 0, 0)
        else:
            self._shm = SharedMemory(name=name, create=False)

        # Map head and tail as ctypes values pointing into shared memory
        # NOTE: Not atomically safe from Python — see module docstring.
        self._buf = self._shm.buf

    @property
    def shm_name(self) -> str:
        """The shared memory block name to pass to the consumer.

        Returns:
            Shared memory identifier string.
        """
        return self._shm.name

    def _read_head(self) -> int:
        (val,) = struct.unpack_from("Q", self._buf, 0)
        return val

    def _read_tail(self) -> int:
        (val,) = struct.unpack_from("Q", self._buf, 8)
        return val

    def _write_head(self, val: int) -> None:
        struct.pack_into("Q", self._buf, 0, val)

    def _write_tail(self, val: int) -> None:
        struct.pack_into("Q", self._buf, 8, val)

    def try_write(self, data: bytes) -> bool:
        """Try to write a message to the ring buffer.

        If the buffer is full, returns False without blocking.

        Args:
            data: Bytes to write. Must be <= SLOT_SIZE bytes.

        Returns:
            True if written, False if buffer was full (back-pressure).

        Raises:
            ValueError: If data exceeds SLOT_SIZE bytes.
        """
        if len(data) > SLOT_SIZE:
            raise ValueError(f"Message size {len(data)} exceeds SLOT_SIZE {SLOT_SIZE}")

        tail = self._read_tail()
        head = self._read_head()
        if tail - head >= CAPACITY:
            return False  # full

        slot_offset = _HEADER_SIZE + (tail & _MASK) * SLOT_SIZE
        # Write length prefix (2 bytes) + data
        msg_len = len(data)
        self._buf[slot_offset : slot_offset + 2] = struct.pack("H", msg_len)
        self._buf[slot_offset + 2 : slot_offset + 2 + msg_len] = data

        # Increment tail AFTER writing (release semantics in C; best-effort here)
        self._write_tail(tail + 1)
        return True

    def try_read(self) -> bytes | None:
        """Try to read the next message from the ring buffer.

        If the buffer is empty, returns None without blocking.

        Returns:
            Next message bytes, or None if empty.
        """
        head = self._read_head()
        tail = self._read_tail()
        if head == tail:
            return None  # empty

        slot_offset = _HEADER_SIZE + (head & _MASK) * SLOT_SIZE
        (msg_len,) = struct.unpack_from("H", self._buf, slot_offset)
        data = bytes(self._buf[slot_offset + 2 : slot_offset + 2 + msg_len])

        # Increment head AFTER reading (acquire semantics in C)
        self._write_head(head + 1)
        return data

    def close(self, unlink: bool = False) -> None:
        """Release the shared memory mapping.

        Args:
            unlink: If True, also delete the shared memory block (call
                    only from the creator, after all consumers have closed).
        """
        self._shm.close()
        if unlink:
            self._shm.unlink()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _producer_proc(shm_name: str, n: int) -> None:
    """Producer process: writes n messages to the ring buffer."""
    rb = SPSCRingBuffer(name=shm_name, create=False)
    payload = b"A" * 32
    written = 0
    while written < n:
        if rb.try_write(payload):
            written += 1
        # else: spin — in production, add a brief sleep or yield
    rb.close()


def _consumer_proc(shm_name: str, n: int, result_queue: "multiprocessing.Queue[float]") -> None:
    """Consumer process: reads n messages and reports throughput."""
    rb = SPSCRingBuffer(name=shm_name, create=False)
    read = 0
    t0 = time.perf_counter_ns()
    while read < n:
        msg = rb.try_read()
        if msg is not None:
            read += 1
    elapsed_ns = time.perf_counter_ns() - t0
    rb.close()
    result_queue.put(elapsed_ns)


def bench_ring_buffer(n: int = 100_000) -> None:
    """Benchmark SPSC ring buffer throughput between two processes.

    Args:
        n: Number of 32-byte messages to transfer.
    """
    rb = SPSCRingBuffer(name=None, create=True)
    shm_name = rb.shm_name

    result_q: multiprocessing.Queue[float] = multiprocessing.Queue()

    prod = multiprocessing.Process(target=_producer_proc, args=(shm_name, n))
    cons = multiprocessing.Process(target=_consumer_proc, args=(shm_name, n, result_q))

    cons.start()
    prod.start()
    prod.join()
    cons.join()
    rb.close(unlink=True)

    elapsed_ns = result_q.get()
    rate = n / (elapsed_ns / 1e9)
    ns_per_msg = elapsed_ns / n
    print(f"  Ring buffer: {n:,} msgs  →  {rate:>12,.0f} msg/s  ({ns_per_msg:.1f} ns/msg)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SPSC Ring Buffer Benchmark ===\n")
    bench_ring_buffer(100_000)
    print(
        "\nNote: Python struct-based atomics are not truly lock-free."
        "\nFor production, use a C extension with std::atomic<uint64_t>."
    )

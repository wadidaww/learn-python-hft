"""
src/concurrency/multiproc_queue.py
=====================================
IPC comparison: multiprocessing.Queue vs multiprocessing.Pipe.

HFT relevance
-------------
Inter-process communication is the latency bottleneck in any
multiprocessing architecture.  Choosing the right IPC mechanism is
critical for a Python-based HFT system:

  | Mechanism           | Latency  | Throughput | Notes                    |
  |---------------------|----------|------------|--------------------------|
  | multiprocessing.Queue | 50–200 µs | ~50k/s  | Thread-safe, buffered    |
  | multiprocessing.Pipe  | 20–100 µs | ~100k/s | Duplex, lower overhead   |
  | shared_memory + flag  | 1–5 µs  | ~1M/s    | Requires explicit sync   |
  | mmap + atomic        | < 1 µs  | ~10M/s   | Kernel-bypass equivalent |

For the hottest paths (e.g., strategy → OMS), use shared memory with
a ring buffer (see lock_free_ring_buffer.py).

Queues are suitable for:
  - Configuration / control messages (low frequency)
  - Error reporting and logging
  - Research pipelines where latency is secondary to convenience
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import os
import time
from multiprocessing import Pipe, Process, Queue
from typing import Any


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _queue_producer(q: "Queue[Any]", n: int) -> None:
    """Send n messages through a Queue.

    Args:
        q: Shared multiprocessing Queue.
        n: Number of messages to send.
    """
    msg = {"type": "tick", "price": 99.5, "qty": 100}
    for i in range(n):
        q.put(msg)
    q.put(None)  # sentinel


def _queue_consumer(q: "Queue[Any]") -> None:
    """Receive messages from a Queue until sentinel is received.

    Args:
        q: Shared multiprocessing Queue.
    """
    while True:
        msg = q.get()
        if msg is None:
            break


def _pipe_producer(conn: multiprocessing.connection.Connection, n: int) -> None:
    """Send n messages through a Pipe.

    Args:
        conn: The send end of a Pipe.
        n:    Number of messages to send.
    """
    msg = {"type": "tick", "price": 99.5, "qty": 100}
    for i in range(n):
        conn.send(msg)
    conn.send(None)  # sentinel
    conn.close()


def _pipe_consumer(conn: multiprocessing.connection.Connection) -> None:
    """Receive messages from a Pipe until sentinel.

    Args:
        conn: The receive end of a Pipe.
    """
    while True:
        msg = conn.recv()
        if msg is None:
            break
    conn.close()


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_queue(n: int = 10_000) -> float:
    """Measure end-to-end throughput of multiprocessing.Queue.

    Args:
        n: Number of messages to transfer.

    Returns:
        Messages per second.
    """
    q: Queue[Any] = Queue(maxsize=0)
    prod = Process(target=_queue_producer, args=(q, n))
    cons = Process(target=_queue_consumer, args=(q,))

    t0 = time.perf_counter()
    cons.start()
    prod.start()
    prod.join()
    cons.join()
    elapsed = time.perf_counter() - t0

    rate = n / elapsed
    print(f"  Queue   ({n:>7,} msgs): {elapsed*1000:.1f} ms  →  {rate:>10,.0f} msg/s")
    return rate


def bench_pipe(n: int = 10_000) -> float:
    """Measure end-to-end throughput of multiprocessing.Pipe.

    A Pipe uses an OS pipe (anonymous pipe on Unix) under the hood.
    It is faster than Queue because Queue wraps a Pipe with additional
    locking and serialisation overhead.

    Args:
        n: Number of messages to transfer.

    Returns:
        Messages per second.
    """
    recv_conn, send_conn = Pipe(duplex=False)
    prod = Process(target=_pipe_producer, args=(send_conn, n))
    cons = Process(target=_pipe_consumer, args=(recv_conn,))

    t0 = time.perf_counter()
    cons.start()
    prod.start()
    prod.join()
    cons.join()
    elapsed = time.perf_counter() - t0

    rate = n / elapsed
    print(f"  Pipe    ({n:>7,} msgs): {elapsed*1000:.1f} ms  →  {rate:>10,.0f} msg/s")
    return rate


def show_pipe_latency(n_samples: int = 1_000) -> None:
    """Measure per-message round-trip latency through a duplex Pipe.

    Uses a ping-pong pattern: main process sends, child echoes back,
    main measures RTT.

    Args:
        n_samples: Number of round-trips to measure.

    TODO: Compare with shared_memory + busy-poll to see the latency floor.
    """
    parent_conn, child_conn = Pipe(duplex=True)

    def echo(conn: multiprocessing.connection.Connection) -> None:
        for _ in range(n_samples):
            msg = conn.recv()
            conn.send(msg)
        conn.close()

    child = Process(target=echo, args=(child_conn,))
    child.start()

    rtts: list[int] = []
    payload = b"x" * 64
    for _ in range(n_samples):
        t0 = time.perf_counter_ns()
        parent_conn.send(payload)
        parent_conn.recv()
        rtts.append(time.perf_counter_ns() - t0)

    child.join()
    parent_conn.close()

    rtts.sort()
    p50 = rtts[len(rtts) // 2]
    p99 = rtts[int(len(rtts) * 0.99)]
    print(
        f"\n  Pipe RTT (n={n_samples}):"
        f"  min={rtts[0]//1000} µs"
        f"  p50={p50//1000} µs"
        f"  p99={p99//1000} µs"
        f"  max={rtts[-1]//1000} µs"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== IPC Throughput Comparison ===\n")
    print(f"  PID: {os.getpid()}")
    print()
    n = 20_000
    bench_queue(n)
    bench_pipe(n)
    show_pipe_latency(500)
    print(
        "\nConclusion:"
        "\n  Pipe > Queue for throughput."
        "\n  For lowest latency, use shared_memory + ring buffer (lock_free_ring_buffer.py)."
    )

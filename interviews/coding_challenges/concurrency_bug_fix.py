"""
interviews/coding_challenges/concurrency_bug_fix.py
=====================================================
Interview Problem: Identify and fix a race condition.

Problem Statement (as asked at Citadel / Jane Street)
------------------------------------------------------
The following code implements a shared order counter used by multiple
threads to assign unique order IDs.  It contains a classic race condition.

Task:
  1. Identify the race condition.
  2. Fix it using the minimal change.
  3. Explain why the original code is broken on a multicore machine.
  4. Discuss lock-free alternatives.

Interview follow-ups:
  1. What is the cost of a lock acquisition in Python?  (~100–500 ns)
  2. Can you implement a lock-free counter?  (Use ctypes atomic, or
     multiprocessing.Value with lock=False + explicit memory barriers)
  3. What happens if you use `threading.local()` instead?
  4. How does the GIL affect this?  (CPython's GIL means a += 1 on an
     int IS atomic at the bytecode level — but don't rely on this!
     The GIL can be released between the LOAD and STORE bytecodes on
     non-integer types, and this assumption breaks with Jython/PyPy.)
"""

from __future__ import annotations

import threading
import time
from typing import Final


# ---------------------------------------------------------------------------
# BROKEN CODE — Race condition
# ---------------------------------------------------------------------------
# Uncomment to observe the bug:
#
# class BrokenOrderCounter:
#     """BROKEN: Not thread-safe.
#
#     The operation `self.count += 1` compiles to:
#       LOAD_ATTR   self.count
#       LOAD_CONST  1
#       BINARY_ADD
#       STORE_ATTR  self.count
#
#     Between LOAD_ATTR and STORE_ATTR, another thread can also read
#     the same value and increment it.  Result: two threads get the same
#     order ID — catastrophic in a trading system.
#
#     Note: In CPython with the GIL, simple integer += IS effectively
#     atomic at the Python level (GIL released on bytecode boundary, not
#     within a bytecode).  But this is an implementation detail, NOT a
#     guarantee.  Code for correctness, not for CPython internals.
#     """
#
#     def __init__(self) -> None:
#         self.count = 0
#
#     def next_id(self) -> int:
#         """THIS IS BROKEN — do not use."""
#         self.count += 1       # Non-atomic read-modify-write
#         return self.count


# ---------------------------------------------------------------------------
# FIXED CODE — Solution 1: threading.Lock
# ---------------------------------------------------------------------------

class OrderCounter:
    """Thread-safe order ID counter using a mutex.

    The mutex ensures that only one thread can execute the
    read-modify-write sequence at a time.

    Cost: ~100–500 ns per acquisition (uncontended).
    Recommendation: Use for low-frequency counters (order entry, not
    per-tick processing).
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._lock = threading.Lock()

    def next_id(self) -> int:
        """Return the next unique order ID (thread-safe).

        Returns:
            Monotonically increasing integer ID.
        """
        with self._lock:
            self._count += 1
            return self._count

    @property
    def current(self) -> int:
        """Current counter value (approximate — may change immediately).

        Returns:
            Current count at time of read.
        """
        with self._lock:
            return self._count


# ---------------------------------------------------------------------------
# FIXED CODE — Solution 2: Per-thread counters (lock-free)
# ---------------------------------------------------------------------------

_THREAD_LOCAL = threading.local()


def get_thread_order_id() -> int:
    """Return a per-thread order ID without any locking.

    Uses thread-local storage: each thread has its own counter.
    IDs are not globally unique — combine with thread ID for uniqueness:
        order_id = (thread_id << 32) | local_counter

    This is the lock-free alternative: O(1) and zero contention.

    Returns:
        Per-thread sequential integer.
    """
    if not hasattr(_THREAD_LOCAL, "counter"):
        _THREAD_LOCAL.counter = 0
    _THREAD_LOCAL.counter += 1
    return _THREAD_LOCAL.counter


# ---------------------------------------------------------------------------
# Verification: prove the fixed counter produces no duplicates
# ---------------------------------------------------------------------------

def verify_thread_safety(
    n_threads: int = 10,
    n_ids_per_thread: int = 10_000,
) -> None:
    """Verify that OrderCounter produces no duplicate IDs under concurrency.

    Args:
        n_threads:         Number of concurrent threads.
        n_ids_per_thread:  IDs each thread will request.
    """
    counter = OrderCounter()
    all_ids: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        local_ids = [counter.next_id() for _ in range(n_ids_per_thread)]
        with lock:
            all_ids.extend(local_ids)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.perf_counter_ns()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed_ns = time.perf_counter_ns() - t0

    total = n_threads * n_ids_per_thread
    unique = len(set(all_ids))
    ns_per_id = elapsed_ns / total

    print(f"OrderCounter verification ({n_threads} threads × {n_ids_per_thread:,} IDs):")
    print(f"  Total IDs generated: {total:,}")
    print(f"  Unique IDs:          {unique:,}")
    print(f"  Duplicates:          {total - unique}")
    print(f"  Throughput:          {total / (elapsed_ns/1e9):,.0f} IDs/s")
    print(f"  Latency per ID:      {ns_per_id:.1f} ns")

    assert unique == total, f"BUG: {total - unique} duplicate order IDs detected!"
    print("  ✓ No duplicates — thread-safe verified")


def verify_thread_local_ids(n_threads: int = 4, n_per_thread: int = 1000) -> None:
    """Verify per-thread counter correctness.

    Args:
        n_threads:    Number of threads.
        n_per_thread: IDs each thread generates.
    """
    results: dict[int, list[int]] = {}
    lock = threading.Lock()

    def worker() -> None:
        tid = threading.get_ident()
        ids = [get_thread_order_id() for _ in range(n_per_thread)]
        with lock:
            results[tid] = ids

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nPer-thread counter ({n_threads} threads × {n_per_thread} IDs):")
    for tid, ids in results.items():
        print(f"  Thread {tid & 0xFFFF}: IDs {ids[0]}..{ids[-1]} ({len(ids)} total)")
    print("  Note: IDs are per-thread sequential, not globally unique")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Interview: Race Condition Fix ===\n")
    verify_thread_safety()
    verify_thread_local_ids()

    print(
        "\nKey interview points:"
        "\n  1. Always use a lock for shared mutable state."
        "\n  2. Per-thread counters eliminate contention entirely."
        "\n  3. CPython GIL makes simple int += 'accidentally' atomic;"
        "\n     this is an implementation detail — never rely on it."
        "\n  4. Lock acquisition cost: ~100-500 ns uncontended."
        "\n     At 1M orders/s, this is the bottleneck → use per-thread IDs."
    )

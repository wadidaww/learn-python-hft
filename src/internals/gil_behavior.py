"""
src/internals/gil_behavior.py
================================
Demonstrates the GIL (Global Interpreter Lock) and its impact on
CPU-bound vs I/O-bound concurrency.

HFT relevance
-------------
The GIL is CPython's mechanism to protect the interpreter's internal
state.  Only one thread can execute Python bytecode at a time.

  - **CPU-bound threads**: No parallelism.  4 threads on a 4-core machine
    will not be faster than 1 thread — the GIL serialises them.
  - **I/O-bound threads**: The GIL is released during blocking I/O syscalls
    (`recv`, `send`, `read`, `write`).  Threads provide real concurrency.
  - **multiprocessing**: Each process has its own GIL.  True CPU parallelism
    at the cost of IPC overhead (~10–100 µs per message).

In an HFT context:
  - Use `asyncio` for many concurrent I/O streams (market data feeds).
  - Use `multiprocessing` for CPU-intensive calculations (risk, pricing).
  - Use Cython `nogil` or C extensions for hot-path numerics.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from typing import Callable


# ---------------------------------------------------------------------------
# CPU-bound workload
# ---------------------------------------------------------------------------

def cpu_bound_task(n: int = 10_000_000) -> int:
    """Pure Python CPU-bound work: counting down from n.

    This keeps the GIL held the entire time (no I/O, no C extension
    call that releases the GIL).

    Args:
        n: Number of iterations.

    Returns:
        Final counter value (always 0).
    """
    total = 0
    for i in range(n):
        total += i
    return total


def io_bound_task(sleep_s: float = 0.1) -> None:
    """Simulate I/O-bound work via sleep.

    `time.sleep` releases the GIL, allowing other threads to run.
    This is representative of waiting on a network recv or disk read.

    Args:
        sleep_s: Seconds to sleep (simulating I/O wait).
    """
    time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def run_with_threads(
    fn: Callable[[], None],
    n_threads: int,
    label: str,
) -> float:
    """Run *fn* concurrently in *n_threads* OS threads.

    Args:
        fn:        Zero-argument callable.
        n_threads: Number of concurrent threads.
        label:     Human-readable label for output.

    Returns:
        Wall-clock elapsed time in seconds.
    """
    threads = [threading.Thread(target=fn) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    print(f"  [threads ×{n_threads}] {label}: {elapsed:.3f}s")
    return elapsed


def run_with_processes(
    fn: Callable[[], None],
    n_procs: int,
    label: str,
) -> float:
    """Run *fn* concurrently in *n_procs* separate processes.

    Each process has its own GIL → true CPU parallelism.
    Overhead: process creation ~10 ms, IPC not needed here.

    Args:
        fn:      Zero-argument callable (must be picklable).
        n_procs: Number of parallel processes.
        label:   Human-readable label for output.

    Returns:
        Wall-clock elapsed time in seconds.
    """
    procs = [multiprocessing.Process(target=fn) for _ in range(n_procs)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    print(f"  [procs   ×{n_procs}] {label}: {elapsed:.3f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demo_cpu_bound(n_workers: int = 4) -> None:
    """Show that CPU-bound threads do NOT scale due to the GIL.

    Expected result:
      - Single thread:  T seconds
      - N threads:      ~T seconds (GIL serialises, no speedup)
      - N processes:    ~T/N seconds (true parallelism)

    TODO: Run this on a machine with N physical cores and observe that
          threads give *no* speedup while processes give ~N× speedup.
    """
    work_n = 5_000_000
    fn = lambda: cpu_bound_task(work_n)  # noqa: E731

    print(f"\n=== CPU-bound benchmark (n={work_n:,}, workers={n_workers}) ===")
    print("  Baseline (1 thread):")
    run_with_threads(fn, 1, "cpu_bound baseline")
    print(f"  {n_workers} threads (GIL prevents parallelism):")
    run_with_threads(fn, n_workers, "cpu_bound threads")
    print(f"  {n_workers} processes (true parallelism):")
    run_with_processes(fn, n_workers, "cpu_bound processes")
    print("  → Threads ≈ baseline; processes ≈ baseline/N")


def demo_io_bound(n_workers: int = 4, sleep_s: float = 0.2) -> None:
    """Show that I/O-bound threads DO scale — GIL released during sleep.

    Expected result:
      - Single thread:  sleep_s × n_workers seconds
      - N threads:      ~sleep_s seconds (all sleep concurrently)
      - N processes:    ~sleep_s seconds (also concurrent, higher overhead)

    TODO: Replace sleep with actual socket recv to simulate market data.
    """
    fn = lambda: io_bound_task(sleep_s)  # noqa: E731

    print(f"\n=== I/O-bound benchmark (sleep={sleep_s}s, workers={n_workers}) ===")
    print("  Baseline (1 thread):")
    run_with_threads(fn, 1, "io_bound baseline")
    print(f"  {n_workers} threads (GIL released during sleep):")
    run_with_threads(fn, n_workers, "io_bound threads")
    print(f"  {n_workers} processes:")
    run_with_processes(fn, n_workers, "io_bound processes")
    print("  → Both threads and processes take ~sleep_s (not sleep_s × N)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GIL Behavior Demonstration")
    print("=" * 50)
    demo_cpu_bound(n_workers=4)
    demo_io_bound(n_workers=4)
    print(
        "\nConclusion:"
        "\n  CPU-bound → use multiprocessing or Cython nogil sections."
        "\n  I/O-bound → threads or asyncio both work; asyncio has lower overhead."
    )

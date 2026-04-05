"""
src/optimization/gc_tuning.py
================================
Demonstrate GC tuning to reduce latency spikes in a trading process.

HFT relevance
-------------
CPython uses reference counting as the primary memory management strategy.
Most objects are freed immediately when their reference count drops to zero
(no GC involvement).

However, CPython also has a **cycle collector** (`gc` module) that handles
reference cycles (e.g., A → B → A).  This cycle collector runs periodically
and can cause latency spikes of 1–50 ms in production.

Default GC thresholds: 700 / 10 / 10
  - Generation 0 (young): collect after 700 allocations
  - Generation 1 (middle): collect after 10 gen-0 collections
  - Generation 2 (old):    collect after 10 gen-1 collections

In a trading process:
  - During market hours (hot path): `gc.disable()` or raise thresholds.
  - During off-hours (EOD reconciliation): `gc.collect()` explicitly.
  - Avoid creating cyclic structures in hot paths entirely.

Best practices:
  1. Use `__slots__` to avoid __dict__ (reduces allocation pressure).
  2. Avoid cyclic references in hot-path objects.
  3. Call `gc.collect()` only during quiet periods.
  4. Use `gc.freeze()` (Python 3.7+) to move long-lived objects to
     generation 2 and prevent them from being re-scanned.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Demonstrate GC pause
# ---------------------------------------------------------------------------

def measure_gc_pause(n_objects: int = 100_000) -> None:
    """Create cyclic garbage and measure collection pause.

    A cyclic structure (e.g. doubly-linked list) cannot be freed by
    reference counting alone.  The cycle collector must run.

    Args:
        n_objects: Number of cyclic objects to create.
    """
    # Enable GC with default thresholds
    gc.enable()
    gc.collect()  # Start clean

    # Create cyclic references (doubly-linked list)
    head = None
    prev = None
    nodes = []
    for i in range(n_objects):
        node: dict[str, Any] = {"value": i, "prev": prev, "next": None}
        if prev:
            prev["next"] = node
        if head is None:
            head = node
        prev = node
        nodes.append(node)

    # Measure a collection
    gc_stats_before = gc.get_count()
    t0 = time.perf_counter_ns()
    collected = gc.collect()
    gc_ns = time.perf_counter_ns() - t0

    print(f"  GC collect({n_objects:,} cyclic objects): {gc_ns/1e6:.2f} ms  (collected={collected})")
    del nodes, head, prev


def measure_gc_disabled(n_objects: int = 100_000) -> None:
    """Measure allocation time with GC disabled.

    With `gc.disable()`, the cycle collector never runs.
    Objects are freed by reference counting when the loop completes.
    No collection pauses during the hot path.

    Args:
        n_objects: Number of objects to allocate.
    """
    gc.disable()

    t0 = time.perf_counter_ns()
    for i in range(n_objects):
        _ = {"value": i, "nested": {"a": 1}}
    alloc_ns = time.perf_counter_ns() - t0

    gc.enable()
    print(f"  Alloc {n_objects:,} dicts (GC off): {alloc_ns/1e6:.2f} ms")


# ---------------------------------------------------------------------------
# GC tuning strategies
# ---------------------------------------------------------------------------

def tune_gc_for_trading() -> None:
    """Apply recommended GC settings for a trading process.

    Strategy:
      1. Raise generation 0 threshold to reduce collection frequency.
      2. Call `gc.freeze()` to exclude long-lived objects from collection.
      3. Disable GC during market hours; re-enable at EOD.

    Call this at process startup, after all long-lived objects are created
    (order book, strategy state, connection pool, etc.).
    """
    # Raise gen-0 threshold: collect less often during bursts
    # Default: (700, 10, 10) → New: (50_000, 10, 10)
    gc.set_threshold(50_000, 10, 10)

    # Freeze current objects: they will never be GC-scanned again
    # This is safe for objects that will live for the process lifetime.
    gc.freeze()

    # Collect any remaining cyclic garbage before the hot path
    gc.collect()

    print("  GC tuned: threshold=50000/10/10, long-lived objects frozen")
    print(f"  Frozen objects: {gc.get_freeze_count()}")
    print(f"  Current thresholds: {gc.get_threshold()}")


def restore_gc_defaults() -> None:
    """Restore GC to default settings (call at EOD / maintenance window).

    Performs a full collection to clean up accumulated cycles.
    """
    gc.unfreeze()
    gc.set_threshold(700, 10, 10)
    collected = gc.collect()
    print(f"  GC restored to defaults. Collected {collected} objects.")


# ---------------------------------------------------------------------------
# Measure allocation profile
# ---------------------------------------------------------------------------

def measure_allocation_pressure(n: int = 100_000) -> None:
    """Show the difference in GC pressure between slotted and dict-based classes.

    Args:
        n: Number of objects to create.
    """
    class PlainOrder:
        def __init__(self, oid: int, price: float) -> None:
            self.oid = oid
            self.price = price

    class SlottedOrder:
        __slots__ = ("oid", "price")

        def __init__(self, oid: int, price: float) -> None:
            self.oid = oid
            self.price = price

    gc.collect()
    before = gc.get_count()

    t0 = time.perf_counter_ns()
    plain_orders = [PlainOrder(i, float(i)) for i in range(n)]
    plain_ns = time.perf_counter_ns() - t0
    plain_count = gc.get_count()
    del plain_orders

    gc.collect()
    t0 = time.perf_counter_ns()
    slotted_orders = [SlottedOrder(i, float(i)) for i in range(n)]
    slotted_ns = time.perf_counter_ns() - t0
    del slotted_orders

    print(f"\n--- Allocation pressure (n={n:,}) ---")
    print(f"  Plain class:   {plain_ns/1e6:.1f} ms  (GC count after: {plain_count})")
    print(f"  Slotted class: {slotted_ns/1e6:.1f} ms")
    print(f"  Speedup: {plain_ns/slotted_ns:.2f}×")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== GC Tuning Demo ===\n")

    print("1. Default GC settings:")
    print(f"   Thresholds: {gc.get_threshold()}")
    print(f"   GC enabled: {gc.isenabled()}")
    print()

    print("2. Cyclic GC pause measurement:")
    measure_gc_pause(50_000)
    print()

    print("3. Allocation with GC disabled:")
    measure_gc_disabled(50_000)
    print()

    print("4. Tuning GC for trading:")
    tune_gc_for_trading()
    print()

    print("5. Allocation pressure comparison:")
    measure_allocation_pressure(50_000)
    print()

    print("6. Restore GC defaults (EOD):")
    restore_gc_defaults()
    print()

    print(
        "Summary:"
        "\n  - Use gc.disable() during ultra-low-latency critical sections."
        "\n  - Use gc.freeze() to exclude long-lived objects from scan."
        "\n  - Use __slots__ to reduce object allocation overhead."
        "\n  - Avoid cyclic references in hot-path data structures."
    )

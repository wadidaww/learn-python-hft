"""
src/internals/memory_layout.py
================================
Demonstrates Python's memory model and its implications for cache
performance in a low-latency trading system.

HFT relevance
-------------
In C++, a `std::vector<double>` stores doubles *contiguously* in memory.
A CPU prefetcher can stride over them efficiently: iterating 1M doubles
costs ~1 ms.

In Python, a `list[float]` stores *pointers* to heap-allocated `float`
objects.  Each float is a 24-byte PyObject (reference count + type ptr
+ value), allocated anywhere in the heap.  Iterating 1M floats triggers
1M pointer dereferences → cache misses → 10–100× slower than C.

Solutions in order of preference:
  1. `numpy.ndarray` — contiguous C array, SIMD-friendly
  2. `array.array`   — contiguous but no SIMD support
  3. `__slots__`     — reduces per-object overhead, but still heap-allocated
  4. `dataclass(frozen=True, slots=True)` — same as __slots__
"""

from __future__ import annotations

import array
import sys
import time
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# 1. Object size comparison
# ---------------------------------------------------------------------------

def show_object_sizes() -> None:
    """Print the in-memory size of common Python objects.

    Note: `sys.getsizeof` returns the *shallow* size — it does not
    recurse into referenced objects.  A list's getsizeof is the size
    of the list shell + pointer array; it does NOT include the objects.
    """
    print("=== Object sizes (sys.getsizeof) ===")
    objects: list[tuple[str, Any]] = [
        ("int(0)", 0),
        ("int(2**30)", 2**30),
        ("float(1.0)", 1.0),
        ("bool", True),
        ("empty list", []),
        ("list[10 ints]", list(range(10))),
        ("empty dict", {}),
        ("empty tuple", ()),
        ("empty str", ""),
        ("str('hello')", "hello"),
        ("bytes(10)", bytes(10)),
        ("array.array('d', 10)", array.array("d", [0.0] * 10)),
    ]
    for label, obj in objects:
        print(f"  {label:<30} {sys.getsizeof(obj):>8} bytes")


# ---------------------------------------------------------------------------
# 2. __slots__ vs plain class
# ---------------------------------------------------------------------------

class Order:
    """Standard Python class — each instance has a __dict__ (56+ bytes).

    The __dict__ itself is a hash map, adding ~200 bytes of overhead per
    instance.  For 1M live orders, this wastes ~200 MB vs a slotted class.
    """

    def __init__(self, order_id: int, price: float, qty: int) -> None:
        self.order_id = order_id
        self.price = price
        self.qty = qty


class SlottedOrder:
    """Slotted class — eliminates __dict__, reduces per-instance overhead.

    HFT relevance: order books can hold millions of live orders.  Using
    __slots__ can reduce memory by 30–60% and improve cache hit rates
    when iterating over order collections.

    TODO: Measure the actual size difference on your machine.
    """

    __slots__ = ("order_id", "price", "qty")

    def __init__(self, order_id: int, price: float, qty: int) -> None:
        self.order_id = order_id
        self.price = price
        self.qty = qty


@dataclass(frozen=True, slots=True)
class FrozenOrder:
    """Frozen slotted dataclass — immutable, hashable, cache-friendly.

    `frozen=True` makes the object hashable (can be used in sets/dicts).
    `slots=True` (Python 3.10+) generates __slots__ automatically.
    """

    order_id: int
    price: float
    qty: int


def show_slot_sizes() -> None:
    """Compare memory footprint of Order vs SlottedOrder."""
    print("\n=== __slots__ memory comparison ===")
    o1 = Order(1, 99.5, 100)
    o2 = SlottedOrder(1, 99.5, 100)
    o3 = FrozenOrder(1, 99.5, 100)
    print(f"  Order (no slots):      {sys.getsizeof(o1):>6} bytes  (+ __dict__: {sys.getsizeof(o1.__dict__)} bytes)")
    print(f"  SlottedOrder:          {sys.getsizeof(o2):>6} bytes  (no __dict__)")
    print(f"  FrozenOrder (slots):   {sys.getsizeof(o3):>6} bytes  (no __dict__)")


# ---------------------------------------------------------------------------
# 3. list[float] vs array.array('d') — cache locality benchmark
# ---------------------------------------------------------------------------

def bench_list_vs_array(n: int = 1_000_000) -> None:
    """Compare iteration speed of list[float] vs array.array.

    A `list` holds PyObject* pointers → pointer-chasing on iteration.
    An `array.array('d')` stores raw doubles contiguously → cache-friendly.

    Expected result: array.array is 3–5× faster to sum over.

    TODO: Add a numpy.ndarray comparison — it should be 10–50× faster
          than list due to SIMD vectorisation.
    """
    data_list: list[float] = [float(i) for i in range(n)]
    data_array: array.array = array.array("d", range(n))

    # Warm up
    _ = sum(data_list[:1000])
    _ = sum(data_array[:1000])

    t0 = time.perf_counter_ns()
    total_list = sum(data_list)
    list_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    total_array = sum(data_array)
    array_ns = time.perf_counter_ns() - t0

    print(f"\n=== list vs array.array (n={n:,}) ===")
    print(f"  list[float]  sum: {list_ns / 1e6:.2f} ms  (result={total_list:.0f})")
    print(f"  array.array  sum: {array_ns / 1e6:.2f} ms  (result={total_array:.0f})")
    print(f"  Speedup: {list_ns / array_ns:.1f}×")


# ---------------------------------------------------------------------------
# 4. Memory fragmentation illustration
# ---------------------------------------------------------------------------

def show_fragmentation() -> None:
    """Illustrate Python heap fragmentation.

    CPython uses a custom allocator (pymalloc) for objects ≤512 bytes.
    Objects are allocated from *arenas* (256 KB each) → *pools* (4 KB)
    → *blocks* (fixed-size buckets).

    Fragmentation occurs when short-lived objects leave holes in pools,
    preventing the OS from reclaiming memory even after `gc.collect()`.

    HFT implication: in a long-running trading process, allocating and
    freeing millions of order objects per day leads to fragmented arenas.
    Use object pools (see object_pool.py) to keep memory stable.

    TODO: Use `tracemalloc` to trace peak memory during an order burst.
    """
    import tracemalloc

    tracemalloc.start()

    # Simulate burst allocation: 100k orders created and discarded
    orders = [Order(i, float(i), i % 100) for i in range(100_000)]
    del orders  # all freed, but pymalloc arenas may not return to OS

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    print("\n=== Top 3 memory allocations (tracemalloc) ===")
    for stat in top_stats[:3]:
        print(f"  {stat}")

    tracemalloc.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    show_object_sizes()
    show_slot_sizes()
    bench_list_vs_array()
    show_fragmentation()

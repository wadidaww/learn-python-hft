"""
src/internals/object_pool.py
==============================
Object pool pattern to reduce GC pressure during high-frequency events.

HFT relevance
-------------
In a Python-based order management system, thousands of Order objects
may be created and destroyed per second.  CPython's reference-counting
GC handles most of this, but cyclic garbage collection pauses (triggered
by `gc.collect()`) can cause latency spikes of 1–100 ms.

An object pool pre-allocates objects and reuses them, avoiding:
  1. `malloc` overhead for new PyObjects.
  2. Dealloc overhead when objects are freed.
  3. GC cycle detection for pooled objects (since they're always referenced).

This pattern is standard in Java (e.g. Chronicle Map, Disruptor) and
is equally valuable in Python for tick-by-tick processing.

Trade-offs:
  - Increased memory usage (pool holds objects even when "idle").
  - Objects must be explicitly returned to the pool (discipline required).
  - Not thread-safe by default — use a lock or per-thread pools.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic Object Pool
# ---------------------------------------------------------------------------

class ObjectPool(Generic[T]):
    """A thread-safe object pool that reuses pre-allocated instances.

    Args:
        factory:  Callable that creates a new instance when the pool is empty.
        max_size: Maximum number of idle objects to retain.

    Example::

        pool = ObjectPool(factory=Order, max_size=10_000)
        order = pool.acquire()
        order.reset(order_id=1, price=99.5, qty=100)
        # ... use order ...
        pool.release(order)
    """

    def __init__(
        self,
        factory: type[T],
        max_size: int = 10_000,
    ) -> None:
        self._factory = factory
        self._max_size = max_size
        self._pool: deque[T] = deque()
        self._lock = threading.Lock()
        self._created = 0
        self._hits = 0
        self._misses = 0

    def acquire(self) -> T:
        """Acquire an object from the pool, or create a new one.

        Returns:
            A recycled or newly created instance.

        Note:
            The returned object may contain stale state from its previous
            use.  Always call a ``reset()`` method before using it.
        """
        with self._lock:
            if self._pool:
                self._hits += 1
                return self._pool.popleft()
            self._misses += 1
            self._created += 1
            return self._factory()  # type: ignore[call-arg]

    def release(self, obj: T) -> None:
        """Return an object to the pool for reuse.

        Args:
            obj: The object to recycle. Must not be used after this call.
        """
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)
            # else: discard — pool is full, let GC handle it

    def pre_allocate(self, count: int) -> None:
        """Pre-fill the pool with *count* instances.

        Call this at startup to avoid allocation latency during market hours.

        Args:
            count: Number of objects to pre-create.
        """
        with self._lock:
            needed = min(count, self._max_size - len(self._pool))
            for _ in range(needed):
                self._pool.append(self._factory())  # type: ignore[call-arg]
                self._created += 1

    @property
    def stats(self) -> dict[str, int]:
        """Return pool statistics.

        Returns:
            Dict with keys: size, created, hits, misses.
        """
        with self._lock:
            return {
                "size": len(self._pool),
                "created": self._created,
                "hits": self._hits,
                "misses": self._misses,
            }


# ---------------------------------------------------------------------------
# Pooled Order object
# ---------------------------------------------------------------------------

class PooledOrder:
    """A reusable order object designed for pool usage.

    Uses __slots__ to minimise per-instance memory and avoid __dict__
    allocation (which would add a ~200-byte overhead per object).

    TODO: Add a `valid` flag to catch use-after-release bugs in testing.
    """

    __slots__ = ("order_id", "price", "qty", "side", "symbol")

    def __init__(self) -> None:
        # Defaults — will be overwritten by reset()
        self.order_id: int = 0
        self.price: float = 0.0
        self.qty: int = 0
        self.side: str = ""
        self.symbol: str = ""

    def reset(
        self,
        order_id: int,
        price: float,
        qty: int,
        side: str,
        symbol: str,
    ) -> "PooledOrder":
        """Reset all fields and return self for chaining.

        Args:
            order_id: Unique order identifier.
            price:    Limit price.
            qty:      Order quantity.
            side:     'B' (buy) or 'S' (sell).
            symbol:   Instrument ticker.

        Returns:
            self (for method chaining).
        """
        self.order_id = order_id
        self.price = price
        self.qty = qty
        self.side = side
        self.symbol = symbol
        return self

    def __repr__(self) -> str:
        return (
            f"PooledOrder(id={self.order_id}, {self.side} "
            f"{self.qty}@{self.price} {self.symbol})"
        )


# ---------------------------------------------------------------------------
# Benchmark: pool vs raw allocation
# ---------------------------------------------------------------------------

def bench_pool_vs_alloc(n: int = 500_000) -> None:
    """Compare pool-based reuse vs fresh allocation for order objects.

    Expected result: pool is ~2–5× faster due to avoiding malloc/GC.

    TODO: Run under `gc.disable()` and observe the difference narrows
          (raw alloc becomes faster without GC overhead).
    """
    pool: ObjectPool[PooledOrder] = ObjectPool(PooledOrder, max_size=n)
    pool.pre_allocate(n)

    # --- Pooled path ---
    t0 = time.perf_counter_ns()
    for i in range(n):
        obj = pool.acquire()
        obj.reset(i, float(i), i % 1000, "B", "AAPL")
        pool.release(obj)
    pool_ns = time.perf_counter_ns() - t0

    # --- Raw allocation path ---
    t0 = time.perf_counter_ns()
    for i in range(n):
        obj2 = PooledOrder()
        obj2.reset(i, float(i), i % 1000, "B", "AAPL")
        # obj2 is discarded → reference count drops to 0 → dealloc
    alloc_ns = time.perf_counter_ns() - t0

    print(f"Pool reuse:    {pool_ns / n:.1f} ns/op  ({pool_ns / 1e6:.0f} ms total)")
    print(f"Raw alloc:     {alloc_ns / n:.1f} ns/op  ({alloc_ns / 1e6:.0f} ms total)")
    print(f"Speedup:       {alloc_ns / pool_ns:.2f}×")
    print(f"Pool stats:    {pool.stats}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Object Pool Benchmark ===\n")
    bench_pool_vs_alloc(500_000)

"""
interviews/coding_challenges/latency_sensitive_algo.py
========================================================
Interview Problem: Calculate VWAP from a stream with minimal memory footprint.

Problem Statement (as asked at Optiver / Citadel / HRT)
---------------------------------------------------------
You are receiving a continuous stream of trades: (price, volume) tuples.
Implement a data structure that:
  1. Computes the current VWAP (Volume-Weighted Average Price) in O(1).
  2. Supports a rolling window of N seconds (VWAP over the last N seconds).
  3. Minimises memory allocations per tick (critical for low latency).
  4. Is thread-safe for a single-writer, single-reader architecture.

Constraints:
  - 500k+ ticks/second in peak conditions.
  - Memory must be bounded (no unbounded list growth).
  - Use Python 3.10+.

Follow-up questions (typical interview):
  1. How would you make this lock-free?
  2. How would you handle clock drift / out-of-order ticks?
  3. What changes if you need per-symbol VWAP simultaneously?
  4. How would you persist this for crash recovery?
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Solution 1: Simple (unbounded) VWAP accumulator
# ---------------------------------------------------------------------------

class VWAPAccumulator:
    """Streaming VWAP with O(1) update and O(1) query.

    Maintains running sum of price*volume and total volume.
    No per-tick allocation.

    Limitation: Does not support a rolling time window.
    Use for session-level VWAP (e.g., full day).
    """

    __slots__ = ("_pv_sum", "_v_sum")

    def __init__(self) -> None:
        self._pv_sum: float = 0.0
        self._v_sum: float = 0.0

    def update(self, price: float, volume: float) -> None:
        """Record a new trade.

        Args:
            price:  Trade price.
            volume: Trade volume.
        """
        self._pv_sum += price * volume
        self._v_sum += volume

    @property
    def vwap(self) -> float:
        """Current VWAP.

        Returns:
            VWAP value, or 0.0 if no trades recorded.
        """
        return self._pv_sum / self._v_sum if self._v_sum > 0 else 0.0

    def reset(self) -> None:
        """Reset state (e.g., start of new trading session)."""
        self._pv_sum = 0.0
        self._v_sum = 0.0


# ---------------------------------------------------------------------------
# Solution 2: Rolling window VWAP (bounded memory)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _TradeBucket:
    """Aggregated trades within a time bucket.

    Using fixed-size time buckets instead of per-tick entries bounds
    memory at O(window_s / bucket_s) regardless of tick rate.

    Args:
        ts_ns:   Bucket start timestamp in nanoseconds.
        pv_sum:  Sum of price * volume in this bucket.
        v_sum:   Total volume in this bucket.
    """

    ts_ns: int
    pv_sum: float
    v_sum: float


class RollingVWAP:
    """Rolling N-second VWAP with bounded memory.

    Uses time bucketing: ticks within the same bucket_ms interval are
    aggregated.  Memory = O(window_s * 1000 / bucket_ms).

    For 60-second window with 100ms buckets: 600 buckets max.
    For 500k ticks/second, each bucket holds ~50k ticks → O(1) per tick.

    Args:
        window_s:  Rolling window size in seconds.
        bucket_ms: Bucket granularity in milliseconds.

    Interview note: This is a classic sliding window problem.  The key
    insight is that aggregating into buckets converts O(N ticks) memory
    to O(N buckets) memory.  Discuss trade-offs of bucket size.
    """

    def __init__(self, window_s: float = 60.0, bucket_ms: int = 100) -> None:
        self._window_ns: int = int(window_s * 1e9)
        self._bucket_ns: int = int(bucket_ms * 1e6)
        self._buckets: deque[_TradeBucket] = deque()
        self._total_pv: float = 0.0
        self._total_v: float = 0.0

    def update(self, price: float, volume: float, ts_ns: int | None = None) -> None:
        """Record a new trade.

        Args:
            price:  Trade price.
            volume: Trade volume.
            ts_ns:  Timestamp in nanoseconds (default: perf_counter_ns).
        """
        if ts_ns is None:
            ts_ns = time.perf_counter_ns()

        # Expire old buckets outside the window
        cutoff = ts_ns - self._window_ns
        while self._buckets and self._buckets[0].ts_ns < cutoff:
            old = self._buckets.popleft()
            self._total_pv -= old.pv_sum
            self._total_v -= old.v_sum

        # Find or create the current bucket
        bucket_start = (ts_ns // self._bucket_ns) * self._bucket_ns
        if not self._buckets or self._buckets[-1].ts_ns != bucket_start:
            self._buckets.append(_TradeBucket(bucket_start, 0.0, 0.0))

        # Accumulate into current bucket
        pv = price * volume
        self._buckets[-1].pv_sum += pv
        self._buckets[-1].v_sum += volume
        self._total_pv += pv
        self._total_v += volume

    @property
    def vwap(self) -> float:
        """Current rolling VWAP.

        Returns:
            VWAP over the last N seconds, or 0.0 if no data.
        """
        return self._total_pv / self._total_v if self._total_v > 0 else 0.0

    @property
    def bucket_count(self) -> int:
        """Number of active time buckets (diagnostic).

        Returns:
            Current number of buckets in memory.
        """
        return len(self._buckets)


# ---------------------------------------------------------------------------
# Benchmark and validation
# ---------------------------------------------------------------------------

def bench_vwap(n_ticks: int = 1_000_000) -> None:
    """Benchmark streaming VWAP update throughput.

    Args:
        n_ticks: Number of simulated ticks.
    """
    import random

    accumulator = VWAPAccumulator()
    rolling = RollingVWAP(window_s=5.0, bucket_ms=50)

    prices = [99.0 + random.gauss(0, 0.5) for _ in range(n_ticks)]
    volumes = [float(random.randint(100, 10_000)) for _ in range(n_ticks)]

    print(f"VWAP benchmark (n={n_ticks:,})")

    t0 = time.perf_counter_ns()
    for p, v in zip(prices, volumes):
        accumulator.update(p, v)
    acc_ns = time.perf_counter_ns() - t0
    print(f"  VWAPAccumulator: {acc_ns/n_ticks:.1f} ns/tick  VWAP={accumulator.vwap:.4f}")

    # Fixed timestamp to avoid perf_counter overhead in benchmark
    base_ts = time.perf_counter_ns()
    t0 = time.perf_counter_ns()
    for i, (p, v) in enumerate(zip(prices, volumes)):
        rolling.update(p, v, ts_ns=base_ts + i * 1_000)  # 1 µs per tick
    roll_ns = time.perf_counter_ns() - t0
    print(f"  RollingVWAP:     {roll_ns/n_ticks:.1f} ns/tick  VWAP={rolling.vwap:.4f}  buckets={rolling.bucket_count}")


if __name__ == "__main__":
    print("=== Interview Problem: Streaming VWAP ===\n")
    bench_vwap(500_000)

    print("\nKey insights:")
    print("  1. VWAPAccumulator: O(1) time, O(1) space — for session VWAP")
    print("  2. RollingVWAP: O(1) amortised time, O(window/bucket) space")
    print("  3. Bucket aggregation is the key to bounding memory")
    print("  4. Avoid per-tick allocation: __slots__ + deque reuse")

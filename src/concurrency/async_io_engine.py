"""
src/concurrency/async_io_engine.py
====================================
Minimal async event loop for multiplexed market-data streams using asyncio.

HFT relevance
-------------
A single-threaded asyncio event loop can handle thousands of concurrent
TCP/UDP connections with ~100 ns per coroutine switch (no OS thread
context switch, no kernel involvement).

Comparison with threads:
  - OS thread switch:  ~1–10 µs (kernel scheduler, register save/restore)
  - asyncio coroutine: ~100–300 ns (Python frame switch, cooperative)

asyncio is ideal for:
  - Managing many simultaneous market-data subscriptions
  - Feeding normalized data to downstream consumers
  - Implementing a mock exchange for testing

asyncio is NOT ideal for:
  - CPU-bound computation (GIL still applies, no parallelism)
  - Latency-critical order submission (single event loop = single-threaded)

The `await` latency: each `await` point is where the event loop *can*
switch to another coroutine.  The actual switch cost is ~100–500 ns;
the worst-case delay depends on how long a coroutine runs without yielding.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import AsyncGenerator


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MarketDataTick:
    """Immutable market data tick.

    Args:
        symbol:    Instrument ticker symbol.
        bid:       Best bid price.
        ask:       Best ask price.
        timestamp: Nanosecond timestamp from perf_counter_ns.
    """

    symbol: str
    bid: float
    ask: float
    timestamp: int


# ---------------------------------------------------------------------------
# Simulated market data generator
# ---------------------------------------------------------------------------

async def simulate_feed(
    symbol: str,
    tick_interval_s: float = 0.001,
    n_ticks: int = 50,
) -> AsyncGenerator[MarketDataTick, None]:
    """Async generator simulating a market-data feed for one instrument.

    In production, this would be a socket reader wrapping a UDP multicast
    or TCP unicast feed.  Here we simulate ticks with random walk prices.

    Args:
        symbol:         Instrument ticker.
        tick_interval_s: Simulated inter-tick interval in seconds.
        n_ticks:        Number of ticks to generate before stopping.

    Yields:
        MarketDataTick instances.

    Note:
        `await asyncio.sleep(0)` is a zero-cost yield that allows other
        coroutines to run — it does NOT sleep for any real time.
        Use a small positive value to simulate real tick rates.
    """
    mid = 100.0 + random.uniform(-5, 5)
    spread = 0.02
    for _ in range(n_ticks):
        mid += random.gauss(0, 0.01)
        tick = MarketDataTick(
            symbol=symbol,
            bid=round(mid - spread / 2, 4),
            ask=round(mid + spread / 2, 4),
            timestamp=time.perf_counter_ns(),
        )
        yield tick
        await asyncio.sleep(tick_interval_s)


# ---------------------------------------------------------------------------
# Fan-out consumer
# ---------------------------------------------------------------------------

class MarketDataEngine:
    """Async engine that subscribes to multiple feeds and dispatches ticks.

    Architecture:
        Feed coroutines → asyncio.Queue → consumer coroutines

    The queue decouples producers (feeds) from consumers (strategy,
    risk engine).  In production, replace the queue with a ring buffer
    (see lock_free_ring_buffer.py) for lower latency.

    TODO: Add backpressure: if the queue is full, measure how many ticks
          are dropped (critical for measuring feed handler capacity).
    """

    def __init__(self, queue_size: int = 10_000) -> None:
        self._queue: asyncio.Queue[MarketDataTick] = asyncio.Queue(
            maxsize=queue_size
        )
        self._tick_count = 0
        self._start_ns: int = 0

    async def ingest_feed(self, symbol: str, n_ticks: int = 50) -> None:
        """Consume one feed and push ticks to the central queue.

        Args:
            symbol:  Instrument ticker for this feed.
            n_ticks: Number of ticks to ingest.
        """
        async for tick in simulate_feed(symbol, tick_interval_s=0, n_ticks=n_ticks):
            await self._queue.put(tick)

    async def process_ticks(self, expected: int) -> None:
        """Drain the queue and process each tick.

        In production, this would normalize the tick, update internal
        state, and forward to strategy callbacks.

        Args:
            expected: Total number of ticks to process before stopping.
        """
        self._start_ns = time.perf_counter_ns()
        while self._tick_count < expected:
            tick = await self._queue.get()
            self._on_tick(tick)
            self._queue.task_done()

    def _on_tick(self, tick: MarketDataTick) -> None:
        """Process a single tick.  This is the hot path.

        Keep this method lean: no Python-level allocations, no logging.
        In production, use a pre-allocated ring buffer and write directly
        to shared memory for the strategy process.

        Args:
            tick: Incoming market data tick.
        """
        self._tick_count += 1

    async def run(self, symbols: list[str], ticks_per_symbol: int = 1000) -> None:
        """Start all feeds and the consumer concurrently.

        Args:
            symbols:          List of instrument tickers to subscribe.
            ticks_per_symbol: Ticks to generate per symbol.
        """
        total = len(symbols) * ticks_per_symbol
        feed_tasks = [
            asyncio.create_task(self.ingest_feed(s, ticks_per_symbol))
            for s in symbols
        ]
        consumer_task = asyncio.create_task(self.process_ticks(total))
        await asyncio.gather(*feed_tasks)
        await consumer_task

    def report(self, symbols: list[str], ticks_per_symbol: int) -> None:
        """Print throughput statistics.

        Args:
            symbols:          Symbols that were processed.
            ticks_per_symbol: Ticks generated per symbol.
        """
        elapsed_s = (time.perf_counter_ns() - self._start_ns) / 1e9
        total = len(symbols) * ticks_per_symbol
        print(f"Processed {total:,} ticks in {elapsed_s:.3f}s")
        print(f"Throughput: {total / elapsed_s:,.0f} ticks/s")
        print(f"Latency per tick: {elapsed_s * 1e9 / total:.1f} ns")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
               "META", "NVDA", "JPM", "GS", "MS"]
    ticks_per_symbol = 500

    engine = MarketDataEngine(queue_size=50_000)
    await engine.run(symbols, ticks_per_symbol)
    engine.report(symbols, ticks_per_symbol)


if __name__ == "__main__":
    print("=== Async Market Data Engine ===\n")
    asyncio.run(_main())

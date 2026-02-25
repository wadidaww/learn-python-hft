"""
benchmarks/throughput_test.py
==============================
Throughput micro-benchmarks measuring messages per second for:
  - Order book insert (Python dict-based)
  - Binary struct pack/unpack
  - JSON encode/decode vs struct
  - asyncio task dispatch rate

HFT relevance
-------------
Throughput (messages/second) and latency are orthogonal metrics.
A system with 1 µs latency and a 1M msg/s throughput ceiling can still
drop messages under a burst.  HFT systems design for *both*: low tail
latency AND sufficient throughput to absorb market-data bursts.

Typical exchange feed rates: 50k–500k msg/s (ITCH), up to 10M msg/s
(aggregated multi-venue).  Python can handle ~500k–2M simple ops/s;
for higher rates, Cython or C extensions are required.
"""

from __future__ import annotations

import json
import struct
import time
from typing import Any


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _measure_throughput(label: str, fn: Any, n: int = 1_000_000) -> None:
    """Run *fn* n times and print messages-per-second.

    Args:
        label: Human-readable benchmark name.
        fn:    Zero-argument callable (the operation to benchmark).
        n:     Number of iterations.
    """
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    elapsed_ns = time.perf_counter_ns() - t0
    elapsed_s = elapsed_ns / 1e9
    rate = n / elapsed_s
    ns_per_op = elapsed_ns / n
    print(f"[{label}]  {rate:>12,.0f} ops/s  ({ns_per_op:.1f} ns/op)")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_dict_insert(n: int = 1_000_000) -> None:
    """Benchmark raw dict insert — the inner loop of a hash-map order book.

    A Python dict lookup / insert is O(1) amortized and typically costs
    ~50–200 ns.  For an order book with millions of price levels this is
    acceptable, but `sortedcontainers.SortedDict` or a C-level `bintrees`
    implementation is preferred for iteration over a price range.

    TODO: Compare `dict` vs `sortedcontainers.SortedDict` for range queries.
    """
    d: dict[int, int] = {}
    counter = 0

    def insert() -> None:
        nonlocal counter
        d[counter] = counter
        counter += 1

    _measure_throughput("dict_insert", insert, n)


def bench_struct_pack(n: int = 2_000_000) -> None:
    """Benchmark binary struct packing — simulating order message serialisation.

    `struct.pack` is implemented in C and avoids Python object allocation
    in the hot path.  It is the fastest Python-level serialisation mechanism,
    typically 10–50× faster than `json.dumps`.

    Format: !HId  → big-endian unsigned short (2B), int (4B), double (8B)
    Total size: 14 bytes — comparable to a minimal binary order entry packet.
    """
    fmt = struct.Struct("!HId")

    def pack() -> None:
        fmt.pack(1, 100, 99.5)

    _measure_throughput("struct_pack", pack, n)


def bench_json_encode(n: int = 500_000) -> None:
    """Benchmark JSON encoding — a common but slow serialisation path.

    JSON is human-readable but allocates heavily and involves string
    operations.  It is suitable for configuration and logging, but
    never for a hot market-data path.

    TODO: Replace `json` with `ujson` (C extension) and compare.
    """
    msg: dict[str, Any] = {
        "type": "NEW_ORDER",
        "order_id": 12345678,
        "price": 99.5,
        "qty": 100,
        "side": "B",
    }

    def encode() -> None:
        json.dumps(msg)

    _measure_throughput("json_encode", encode, n)


def bench_json_decode(n: int = 500_000) -> None:
    """Benchmark JSON decoding — the inbound parse path for REST/WebSocket APIs.

    TODO: Compare `json.loads` vs `ujson.loads` vs `orjson.loads`.
    """
    raw = b'{"type":"NEW_ORDER","order_id":12345678,"price":99.5,"qty":100,"side":"B"}'

    def decode() -> None:
        json.loads(raw)

    _measure_throughput("json_decode", decode, n)


def bench_struct_unpack(n: int = 2_000_000) -> None:
    """Benchmark binary struct unpacking — simulating inbound feed parsing.

    This is ~10-20× faster than JSON decoding for equivalent data.
    """
    fmt = struct.Struct("!HId")
    raw = fmt.pack(1, 100, 99.5)

    def unpack() -> None:
        fmt.unpack(raw)

    _measure_throughput("struct_unpack", unpack, n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Python HFT Throughput Benchmarks")
    print("=" * 60)
    bench_dict_insert()
    bench_struct_pack()
    bench_struct_unpack()
    bench_json_encode()
    bench_json_decode()
    print()
    print("Key takeaway: struct is 10-20x faster than JSON for binary data.")
    print("Use binary protocols on all hot paths; JSON only for config/logs.")

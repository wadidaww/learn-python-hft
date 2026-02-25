"""
src/market_data/fix_parser.py
================================
FIX protocol parser: naive string-split vs optimised byte-level parser.

HFT relevance
-------------
FIX (Financial Information eXchange) is the industry standard protocol
for order entry and execution reports.  A FIX message looks like:

  8=FIX.4.2\x019=65\x0135=D\x0149=SENDER\x0156=TARGET\x0111=ORDER1\x0155=AAPL\x0154=1\x0138=100\x0144=99.5\x0110=123\x01

Where:
  - '\x01' (SOH, ASCII 1) is the field delimiter
  - Each field is tag=value
  - Tag 8 = BeginString, 35 = MsgType, 55 = Symbol, etc.

Parser performance comparison:
  - Naive (str.split + dict): ~3–10 µs per message
  - Byte-level (memoryview + manual scan): ~0.5–2 µs per message
  - C extension (e.g. quickfix, simplefix): ~0.1–0.5 µs per message

Key insight: `bytes.split()` allocates a new list and new bytes objects
for each call.  A byte-level scanner using `memoryview` avoids most
allocations, keeping GC pressure low.

TODO:
  - Implement a zero-copy parser using a pre-allocated dict and
    memoryview slicing (avoids all string allocation).
  - Benchmark with `cProfile` to identify where allocations occur.
"""

from __future__ import annotations

import re
import time
from typing import Iterator

# FIX field delimiter (SOH = Start of Heading, ASCII 1)
SOH = b"\x01"
SOH_INT = 1  # ord('\x01')

# Common FIX tags (subset)
TAG_BEGIN_STRING = 8
TAG_BODY_LENGTH = 9
TAG_MSG_TYPE = 35
TAG_SENDER = 49
TAG_TARGET = 56
TAG_ORDER_ID = 11
TAG_SYMBOL = 55
TAG_SIDE = 54  # 1=Buy, 2=Sell
TAG_QTY = 38
TAG_PRICE = 44
TAG_CHECKSUM = 10


# ---------------------------------------------------------------------------
# Sample FIX message factory
# ---------------------------------------------------------------------------

def make_fix_order(
    order_id: str,
    symbol: str,
    side: int,
    qty: int,
    price: float,
) -> bytes:
    """Build a minimal FIX 4.2 New Order Single (D) message.

    Args:
        order_id: Unique client order ID.
        symbol:   Instrument symbol.
        side:     1=Buy, 2=Sell.
        qty:      Order quantity.
        price:    Limit price.

    Returns:
        Raw FIX message as bytes.
    """
    body = (
        f"35=D\x01"
        f"49=CLIENT\x01"
        f"56=EXCHANGE\x01"
        f"11={order_id}\x01"
        f"55={symbol}\x01"
        f"54={side}\x01"
        f"38={qty}\x01"
        f"44={price}\x01"
    ).encode()

    header = f"8=FIX.4.2\x019={len(body)}\x01".encode()
    raw = header + body
    # Checksum: sum of all byte values mod 256, formatted as 3 digits
    checksum = sum(raw) % 256
    raw += f"10={checksum:03d}\x01".encode()
    return raw


# ---------------------------------------------------------------------------
# Parser 1: Naive (split-based)
# ---------------------------------------------------------------------------

def parse_fix_naive(raw: bytes) -> dict[int, str]:
    """Parse a FIX message using bytes.split — simple but allocates heavily.

    Each `split` creates a new list and new bytes objects.  For 100k
    messages/second, this generates significant GC pressure.

    Args:
        raw: Raw FIX message bytes.

    Returns:
        Dict mapping tag (int) to value (str).
    """
    result: dict[int, str] = {}
    for field in raw.split(SOH):
        if not field or b"=" not in field:
            continue
        tag_bytes, _, value_bytes = field.partition(b"=")
        try:
            result[int(tag_bytes)] = value_bytes.decode()
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# Parser 2: Optimised (memoryview / byte scan)
# ---------------------------------------------------------------------------

def parse_fix_optimised(raw: bytes) -> dict[int, bytes]:
    """Parse a FIX message using manual byte scanning with memoryview.

    Uses a memoryview to avoid copying data when slicing.  Tags and values
    are returned as `bytes` views (not decoded strings) to avoid UTF-8
    decode overhead in the hot path.

    In a production zero-copy parser, you would keep a pre-allocated
    result dict and clear it between messages to avoid dict allocation.

    Args:
        raw: Raw FIX message bytes.

    Returns:
        Dict mapping tag (int) to value (bytes view).
    """
    result: dict[int, bytes] = {}
    mv = memoryview(raw)
    length = len(mv)
    pos = 0

    while pos < length:
        # Find '='
        eq_pos = pos
        while eq_pos < length and mv[eq_pos] != ord("="):
            eq_pos += 1
        if eq_pos >= length:
            break

        # Find SOH (field delimiter)
        soh_pos = eq_pos + 1
        while soh_pos < length and mv[soh_pos] != SOH_INT:
            soh_pos += 1

        tag_bytes = bytes(mv[pos:eq_pos])
        value_bytes = bytes(mv[eq_pos + 1 : soh_pos])

        try:
            result[int(tag_bytes)] = value_bytes
        except ValueError:
            pass

        pos = soh_pos + 1

    return result


# ---------------------------------------------------------------------------
# Parser 3: Pre-compiled regex (for comparison)
# ---------------------------------------------------------------------------

_FIX_RE = re.compile(rb"(\d+)=([^\x01]*)\x01")


def parse_fix_regex(raw: bytes) -> dict[int, bytes]:
    """Parse a FIX message using a pre-compiled regex.

    Regex is convenient but typically slower than a hand-written scanner
    for fixed-format protocols.  Never use this in a hot path.

    Args:
        raw: Raw FIX message bytes.

    Returns:
        Dict mapping tag (int) to value (bytes).
    """
    return {int(m.group(1)): m.group(2) for m in _FIX_RE.finditer(raw)}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_parsers(n: int = 100_000) -> None:
    """Compare the three FIX parsers on n identical messages.

    Args:
        n: Number of parse iterations.
    """
    msg = make_fix_order("ORD001", "AAPL", 1, 100, 99.50)

    # Warm up
    for _ in range(1000):
        parse_fix_naive(msg)
        parse_fix_optimised(msg)
        parse_fix_regex(msg)

    parsers = [
        ("naive (split)", parse_fix_naive),
        ("optimised (scan)", parse_fix_optimised),
        ("regex", parse_fix_regex),
    ]

    print(f"FIX parser benchmark (n={n:,}, msg_len={len(msg)} bytes)\n")
    for label, fn in parsers:
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn(msg)
        elapsed_ns = time.perf_counter_ns() - t0
        ns_per_op = elapsed_ns / n
        print(f"  {label:<25}: {ns_per_op:>8.1f} ns/msg  ({elapsed_ns/1e6:>8.1f} ms total)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== FIX Protocol Parser ===\n")

    # Show sample message
    sample = make_fix_order("ORD001", "AAPL", 1, 100, 99.50)
    print(f"Sample FIX message:\n  {sample.decode().replace(chr(1), '|')}\n")

    # Parse with both parsers and show fields
    parsed = parse_fix_naive(sample)
    print("Parsed fields (naive):")
    for tag, value in sorted(parsed.items()):
        print(f"  Tag {tag:>4} = {value!r}")

    print()
    bench_parsers(100_000)

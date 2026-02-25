"""
src/market_data/binary_protocol.py
=====================================
Binary order message protocol using struct.pack/unpack.

HFT relevance
-------------
Binary protocols are 10–100× faster to parse than text protocols (FIX,
JSON) because:
  1. No character-by-character scanning or delimiter searching.
  2. No string-to-number conversions (values are natively typed).
  3. Fixed message sizes enable zero-copy parsing.
  4. CPU-friendly: sequential memory access, predictable branch patterns.

Real-world binary protocols:
  - NASDAQ ITCH 5.0:  Variable-length, tag-less, 40-byte add-order messages
  - CME MDP 3.0:      FlatBuffers-based, SBE (Simple Binary Encoding)
  - CBOE BOE:         Binary Order Entry, fixed-size messages
  - Eurex ETI:        Binary, big-endian, tag-less
  - OUCH 4.2:         NASDAQ order entry, 47-byte new order

This module defines a simplified binary protocol for order messages and
demonstrates the performance advantage over JSON.

Message format (48 bytes total):
  Offset | Size | Type    | Field
  -------|------|---------|-------
  0      | 1    | uint8   | msg_type (1=NewOrder, 2=Cancel, 3=Modify)
  1      | 8    | uint64  | order_id
  9      | 8    | uint64  | timestamp_ns
  17     | 4    | char[4] | symbol (padded with spaces)
  21     | 1    | uint8   | side (1=Buy, 2=Sell)
  22     | 4    | uint32  | qty
  26     | 8    | double  | price
  34     | 2    | uint16  | venue_id
  36     | 12   | bytes   | reserved (padding to 48 bytes)
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

class MsgType(IntEnum):
    """Order message types."""
    NEW_ORDER = 1
    CANCEL = 2
    MODIFY = 3


class Side(IntEnum):
    """Order side."""
    BUY = 1
    SELL = 2


# ---------------------------------------------------------------------------
# Message format
# ---------------------------------------------------------------------------

# Struct format: big-endian
# B=uint8, Q=uint64, 4s=4-byte char array, I=uint32, d=double, H=uint16, 12x=12 padding bytes
_ORDER_STRUCT = struct.Struct("!BQQ4sBId H 12x")
assert _ORDER_STRUCT.size == 48, f"Expected 48 bytes, got {_ORDER_STRUCT.size}"

MSG_SIZE = _ORDER_STRUCT.size  # 48 bytes


@dataclass(slots=True)
class OrderMessage:
    """Decoded binary order message.

    Args:
        msg_type:     Message type (MsgType enum).
        order_id:     Unique order identifier.
        timestamp_ns: Message creation timestamp (nanoseconds).
        symbol:       4-char instrument symbol.
        side:         Buy or Sell.
        qty:          Order quantity.
        price:        Limit price.
        venue_id:     Destination venue identifier.
    """

    msg_type: int
    order_id: int
    timestamp_ns: int
    symbol: str
    side: int
    qty: int
    price: float
    venue_id: int

    # Pre-allocated buffer for encoding (avoids per-call malloc)
    _encode_buf: ClassVar[bytearray] = bytearray(MSG_SIZE)

    def encode(self) -> bytes:
        """Serialise this message to 48 bytes.

        Returns:
            Fixed-size bytes representation.
        """
        sym_bytes = self.symbol.encode()[:4].ljust(4)
        return _ORDER_STRUCT.pack(
            self.msg_type,
            self.order_id,
            self.timestamp_ns,
            sym_bytes,
            self.side,
            self.qty,
            self.price,
            self.venue_id,
        )

    def encode_into(self, buf: bytearray, offset: int = 0) -> int:
        """Zero-copy encode directly into an existing buffer.

        This avoids allocating a new bytes object for every message,
        which is critical in a tight loop sending 1M messages/second.

        Args:
            buf:    Pre-allocated bytearray to write into.
            offset: Byte offset within buf.

        Returns:
            New offset after writing (offset + MSG_SIZE).
        """
        sym_bytes = self.symbol.encode()[:4].ljust(4)
        _ORDER_STRUCT.pack_into(
            buf,
            offset,
            self.msg_type,
            self.order_id,
            self.timestamp_ns,
            sym_bytes,
            self.side,
            self.qty,
            self.price,
            self.venue_id,
        )
        return offset + MSG_SIZE

    @staticmethod
    def decode(raw: bytes, offset: int = 0) -> "OrderMessage":
        """Deserialise 48 bytes into an OrderMessage.

        Args:
            raw:    Raw bytes to parse.
            offset: Byte offset within raw.

        Returns:
            Decoded OrderMessage.
        """
        (msg_type, order_id, ts_ns, sym_bytes, side, qty, price, venue_id) = \
            _ORDER_STRUCT.unpack_from(raw, offset)
        return OrderMessage(
            msg_type=msg_type,
            order_id=order_id,
            timestamp_ns=ts_ns,
            symbol=sym_bytes.rstrip().decode(),
            side=side,
            qty=qty,
            price=price,
            venue_id=venue_id,
        )

    def to_json(self) -> str:
        """Serialise to JSON (for comparison — never use in hot path).

        Returns:
            JSON string representation.
        """
        return json.dumps({
            "msg_type": self.msg_type,
            "order_id": self.order_id,
            "timestamp_ns": self.timestamp_ns,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "venue_id": self.venue_id,
        })

    @staticmethod
    def from_json(raw: str) -> "OrderMessage":
        """Deserialise from JSON.

        Args:
            raw: JSON string.

        Returns:
            Decoded OrderMessage.
        """
        d = json.loads(raw)
        return OrderMessage(**d)


# ---------------------------------------------------------------------------
# Benchmark: binary struct vs JSON
# ---------------------------------------------------------------------------

def bench_binary_vs_json(n: int = 1_000_000) -> None:
    """Compare binary struct and JSON encode/decode performance.

    Args:
        n: Number of iterations per benchmark.
    """
    msg = OrderMessage(
        msg_type=MsgType.NEW_ORDER,
        order_id=12345678,
        timestamp_ns=time.perf_counter_ns(),
        symbol="AAPL",
        side=Side.BUY,
        qty=100,
        price=99.50,
        venue_id=1,
    )

    # Pre-allocate a buffer for zero-copy encoding
    buf = bytearray(MSG_SIZE * 100)

    print(f"Message size comparison:")
    encoded = msg.encode()
    json_str = msg.to_json()
    print(f"  Binary struct: {len(encoded)} bytes")
    print(f"  JSON:          {len(json_str)} bytes")
    print(f"  JSON/binary ratio: {len(json_str)/len(encoded):.1f}×\n")

    benchmarks = [
        ("struct encode  (alloc)", lambda: msg.encode()),
        ("struct encode  (no-alloc)", lambda: msg.encode_into(buf, 0)),
        ("struct decode", lambda: OrderMessage.decode(encoded)),
        ("json encode", lambda: msg.to_json()),
        ("json decode", lambda: OrderMessage.from_json(json_str)),
    ]

    print(f"Encode/decode benchmark (n={n:,}):\n")
    for label, fn in benchmarks:
        # Warm up
        for _ in range(1000):
            fn()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        ns_per_op = (time.perf_counter_ns() - t0) / n
        print(f"  {label:<30}: {ns_per_op:>8.1f} ns/op")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Binary Protocol Demo ===\n")

    msg = OrderMessage(
        msg_type=MsgType.NEW_ORDER,
        order_id=1,
        timestamp_ns=time.perf_counter_ns(),
        symbol="MSFT",
        side=Side.SELL,
        qty=500,
        price=415.25,
        venue_id=2,
    )

    encoded = msg.encode()
    decoded = OrderMessage.decode(encoded)

    print(f"Original:  {msg}")
    print(f"Encoded:   {encoded.hex()}")
    print(f"Decoded:   {decoded}")
    print(f"Round-trip match: {msg.order_id == decoded.order_id and msg.price == decoded.price}")
    print()

    bench_binary_vs_json(500_000)

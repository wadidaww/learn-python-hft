"""
tests/test_networking.py
===========================
Unit tests for networking utilities and socket options.

Tests cover:
  - Socket option configuration (TCP_NODELAY, SO_REUSEADDR, etc.)
  - FIX message construction and parsing
  - Binary protocol encode/decode round-trip
  - Basic async TCP server connectivity
"""

from __future__ import annotations

import asyncio
import socket
import struct

import pytest

from src.market_data.binary_protocol import MSG_SIZE, MsgType, OrderMessage, Side
from src.market_data.fix_parser import (
    TAG_MSG_TYPE,
    TAG_PRICE,
    TAG_QTY,
    TAG_SIDE,
    TAG_SYMBOL,
    make_fix_order,
    parse_fix_naive,
    parse_fix_optimised,
    parse_fix_regex,
)
from src.networking.socket_options import (
    configure_socket_buffers,
    configure_tcp_low_latency,
)


# ---------------------------------------------------------------------------
# Binary protocol tests
# ---------------------------------------------------------------------------

class TestBinaryProtocol:
    """Tests for the binary order message protocol."""

    def _make_msg(self) -> OrderMessage:
        return OrderMessage(
            msg_type=MsgType.NEW_ORDER,
            order_id=42,
            timestamp_ns=1_000_000_000,
            symbol="AAPL",
            side=Side.BUY,
            qty=100,
            price=150.25,
            venue_id=1,
        )

    def test_encode_returns_correct_size(self) -> None:
        msg = self._make_msg()
        encoded = msg.encode()
        assert len(encoded) == MSG_SIZE

    def test_round_trip_preserves_fields(self) -> None:
        msg = self._make_msg()
        encoded = msg.encode()
        decoded = OrderMessage.decode(encoded)

        assert decoded.order_id == msg.order_id
        assert decoded.msg_type == msg.msg_type
        assert decoded.symbol == msg.symbol
        assert decoded.side == msg.side
        assert decoded.qty == msg.qty
        assert decoded.price == pytest.approx(msg.price, rel=1e-10)
        assert decoded.venue_id == msg.venue_id
        assert decoded.timestamp_ns == msg.timestamp_ns

    def test_encode_into_writes_to_buffer(self) -> None:
        msg = self._make_msg()
        buf = bytearray(MSG_SIZE * 2)
        new_offset = msg.encode_into(buf, 0)
        assert new_offset == MSG_SIZE

        # Verify the buffer contains the same data as encode()
        assert bytes(buf[:MSG_SIZE]) == msg.encode()

    def test_encode_into_with_offset(self) -> None:
        msg = self._make_msg()
        buf = bytearray(MSG_SIZE * 2)
        msg.encode_into(buf, MSG_SIZE)  # write at second slot

        decoded = OrderMessage.decode(buf, MSG_SIZE)
        assert decoded.order_id == msg.order_id

    def test_symbol_truncation(self) -> None:
        """Symbol longer than 4 chars should be truncated."""
        msg = OrderMessage(
            msg_type=MsgType.NEW_ORDER,
            order_id=1,
            timestamp_ns=0,
            symbol="TOOLONG",  # > 4 chars
            side=Side.SELL,
            qty=10,
            price=1.0,
            venue_id=0,
        )
        encoded = msg.encode()
        decoded = OrderMessage.decode(encoded)
        # Symbol should be at most 4 chars
        assert len(decoded.symbol) <= 4

    def test_different_msg_types(self) -> None:
        for mt in [MsgType.NEW_ORDER, MsgType.CANCEL, MsgType.MODIFY]:
            msg = OrderMessage(
                msg_type=mt,
                order_id=mt.value,
                timestamp_ns=0,
                symbol="MSFT",
                side=Side.BUY,
                qty=1,
                price=1.0,
                venue_id=0,
            )
            decoded = OrderMessage.decode(msg.encode())
            assert decoded.msg_type == mt


# ---------------------------------------------------------------------------
# FIX parser tests
# ---------------------------------------------------------------------------

class TestFIXParser:
    """Tests for FIX protocol parser."""

    def test_make_fix_order_is_bytes(self) -> None:
        raw = make_fix_order("ORD1", "AAPL", 1, 100, 99.5)
        assert isinstance(raw, bytes)

    def test_make_fix_order_contains_soh_delimiters(self) -> None:
        raw = make_fix_order("ORD1", "AAPL", 1, 100, 99.5)
        assert b"\x01" in raw

    def test_naive_parser_extracts_symbol(self) -> None:
        raw = make_fix_order("ORD1", "AAPL", 1, 100, 99.5)
        fields = parse_fix_naive(raw)
        assert fields[TAG_SYMBOL] == "AAPL"

    def test_naive_parser_extracts_price(self) -> None:
        raw = make_fix_order("ORD1", "AAPL", 1, 100, 99.5)
        fields = parse_fix_naive(raw)
        assert float(fields[TAG_PRICE]) == pytest.approx(99.5)

    def test_naive_parser_extracts_qty(self) -> None:
        raw = make_fix_order("ORD1", "AAPL", 1, 100, 99.5)
        fields = parse_fix_naive(raw)
        assert int(fields[TAG_QTY]) == 100

    def test_naive_parser_extracts_side(self) -> None:
        raw = make_fix_order("ORD1", "AAPL", 1, 100, 99.5)
        fields = parse_fix_naive(raw)
        assert int(fields[TAG_SIDE]) == 1  # Buy

    def test_optimised_parser_matches_naive(self) -> None:
        raw = make_fix_order("ORD1", "MSFT", 2, 500, 415.0)
        naive = parse_fix_naive(raw)
        optimised = parse_fix_optimised(raw)

        # Compare common tags (naive returns str, optimised returns bytes)
        for tag in [TAG_SYMBOL, TAG_SIDE, TAG_QTY]:
            assert str(naive[tag]) == optimised[tag].decode()

    def test_regex_parser_matches_naive(self) -> None:
        raw = make_fix_order("ORD2", "GOOG", 1, 200, 175.5)
        naive = parse_fix_naive(raw)
        regex = parse_fix_regex(raw)

        for tag in [TAG_SYMBOL, TAG_SIDE, TAG_QTY]:
            assert str(naive[tag]) == regex[tag].decode()

    def test_all_parsers_extract_msg_type(self) -> None:
        raw = make_fix_order("ORD1", "TSLA", 2, 50, 200.0)
        for parser in [parse_fix_naive, parse_fix_optimised, parse_fix_regex]:
            fields = parser(raw)
            # Tag 35 = MsgType = 'D' (New Order Single)
            msg_type = fields[TAG_MSG_TYPE]
            val = msg_type if isinstance(msg_type, str) else msg_type.decode()
            assert val == "D"

    def test_parser_handles_sell_side(self) -> None:
        raw = make_fix_order("ORD3", "NVDA", 2, 75, 500.0)  # side=2=Sell
        fields = parse_fix_naive(raw)
        assert int(fields[TAG_SIDE]) == 2


# ---------------------------------------------------------------------------
# Socket options tests
# ---------------------------------------------------------------------------

class TestSocketOptions:
    """Tests for socket configuration utilities."""

    def test_configure_tcp_low_latency_sets_nodelay(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            configure_tcp_low_latency(sock)
            nodelay = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            assert nodelay == 1
        finally:
            sock.close()

    def test_configure_tcp_low_latency_sets_reuseaddr(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            configure_tcp_low_latency(sock)
            reuse = sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
            assert reuse == 1
        finally:
            sock.close()

    def test_configure_socket_buffers_returns_actual_sizes(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = configure_socket_buffers(sock, rcvbuf_bytes=65536, sndbuf_bytes=65536)
            # Result should be a tuple of (actual_rcvbuf, actual_sndbuf)
            assert result is not None
            assert len(result) == 2
            # Actual sizes may differ from requested (kernel may double them)
            rcv_actual, snd_actual = result
            assert rcv_actual > 0
            assert snd_actual > 0
        finally:
            sock.close()


# ---------------------------------------------------------------------------
# Async TCP server tests
# ---------------------------------------------------------------------------

class TestTCPOrderServer:
    """Integration tests for the async TCP order server."""

    @pytest.mark.asyncio
    async def test_server_echoes_order(self) -> None:
        """Send an order to the server and verify it echoes back."""
        from src.networking.tcp_order_server import OrderClient, OrderServer

        server = OrderServer(host="127.0.0.1", port=19980)
        await server.start()

        client = OrderClient(host="127.0.0.1", port=19980)
        await client.connect()

        payload = b"NEW_ORDER|AAPL|BUY|100|99.50"
        ack = await client.send_order(payload)
        assert ack == payload

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_handles_multiple_orders(self) -> None:
        """Send multiple orders and verify all are echoed correctly."""
        from src.networking.tcp_order_server import OrderClient, OrderServer

        server = OrderServer(host="127.0.0.1", port=19981)
        await server.start()

        client = OrderClient(host="127.0.0.1", port=19981)
        await client.connect()

        orders = [
            b"NEW_ORDER|MSFT|SELL|200|415.00",
            b"NEW_ORDER|GOOG|BUY|50|175.00",
            b"CANCEL|ORD123",
        ]
        for order in orders:
            ack = await client.send_order(order)
            assert ack == order

        await client.close()
        await server.stop()
        assert server.stats["total_messages"] == len(orders)

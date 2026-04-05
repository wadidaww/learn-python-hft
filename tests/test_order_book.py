"""
tests/test_order_book.py
===========================
Unit tests for the Limit Order Book implementation.

Tests cover:
  - Order insertion and best-price tracking
  - Order cancellation
  - Immediate matching (aggressive orders)
  - Spread and mid-price calculations
  - Book depth queries
  - Edge cases: empty book, cross-spread, partial fills
"""

from __future__ import annotations

import pytest

from src.market_data.order_book import LimitOrderBook, Order, Side, Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def book() -> LimitOrderBook:
    """Return a fresh empty order book for AAPL."""
    return LimitOrderBook("AAPL")


@pytest.fixture
def oid_counter() -> list[int]:
    """Simple mutable counter for unique order IDs."""
    return [0]


def make_order(
    oid_counter: list[int],
    side: Side,
    price: float,
    qty: int,
    symbol: str = "AAPL",
) -> Order:
    """Helper to create an Order with an auto-incremented ID.

    Args:
        oid_counter: Mutable list holding [current_id].
        side:        BID or ASK.
        price:       Limit price.
        qty:         Quantity.
        symbol:      Instrument symbol.

    Returns:
        New Order instance.
    """
    oid_counter[0] += 1
    return Order(oid_counter[0], symbol, side, price, qty)


# ---------------------------------------------------------------------------
# Empty book tests
# ---------------------------------------------------------------------------

class TestEmptyBook:
    """Tests for an empty order book."""

    def test_best_bid_is_none(self, book: LimitOrderBook) -> None:
        assert book.best_bid() is None

    def test_best_ask_is_none(self, book: LimitOrderBook) -> None:
        assert book.best_ask() is None

    def test_spread_is_none(self, book: LimitOrderBook) -> None:
        assert book.spread() is None

    def test_mid_is_none(self, book: LimitOrderBook) -> None:
        assert book.mid_price() is None

    def test_depth_empty(self, book: LimitOrderBook) -> None:
        depth = book.depth(5)
        assert depth["bids"] == []
        assert depth["asks"] == []


# ---------------------------------------------------------------------------
# Order insertion tests
# ---------------------------------------------------------------------------

class TestOrderInsertion:
    """Tests for order insertion into the book."""

    def test_single_bid_sets_best_bid(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        order = make_order(oid_counter, Side.BID, 100.0, 100)
        book.add_order(order)
        assert book.best_bid() == 100.0

    def test_single_ask_sets_best_ask(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        order = make_order(oid_counter, Side.ASK, 101.0, 100)
        book.add_order(order)
        assert book.best_ask() == 101.0

    def test_best_bid_is_highest_price(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        book.add_order(make_order(oid_counter, Side.BID, 99.0, 100))
        book.add_order(make_order(oid_counter, Side.BID, 100.0, 100))
        book.add_order(make_order(oid_counter, Side.BID, 98.0, 100))
        assert book.best_bid() == 100.0

    def test_best_ask_is_lowest_price(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        book.add_order(make_order(oid_counter, Side.ASK, 101.0, 100))
        book.add_order(make_order(oid_counter, Side.ASK, 100.5, 100))
        book.add_order(make_order(oid_counter, Side.ASK, 102.0, 100))
        assert book.best_ask() == 100.5

    def test_spread_calculated_correctly(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        book.add_order(make_order(oid_counter, Side.BID, 99.5, 100))
        book.add_order(make_order(oid_counter, Side.ASK, 100.5, 100))
        assert book.spread() == pytest.approx(1.0, rel=1e-9)

    def test_mid_price_calculated_correctly(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        book.add_order(make_order(oid_counter, Side.BID, 99.0, 100))
        book.add_order(make_order(oid_counter, Side.ASK, 101.0, 100))
        assert book.mid_price() == pytest.approx(100.0, rel=1e-9)

    def test_no_trades_for_non_crossing_order(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        book.add_order(make_order(oid_counter, Side.ASK, 101.0, 100))
        trades = book.add_order(make_order(oid_counter, Side.BID, 100.0, 100))
        assert trades == []


# ---------------------------------------------------------------------------
# Cancellation tests
# ---------------------------------------------------------------------------

class TestCancellation:
    """Tests for order cancellation."""

    def test_cancel_existing_order(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        order = make_order(oid_counter, Side.BID, 100.0, 100)
        book.add_order(order)
        cancelled = book.cancel_order(order.order_id)
        assert cancelled is not None
        assert cancelled.order_id == order.order_id

    def test_cancel_removes_from_book(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        order = make_order(oid_counter, Side.BID, 100.0, 100)
        book.add_order(order)
        book.cancel_order(order.order_id)
        assert book.best_bid() is None

    def test_cancel_nonexistent_order_returns_none(
        self, book: LimitOrderBook
    ) -> None:
        result = book.cancel_order(999999)
        assert result is None

    def test_cancel_leaves_other_orders_intact(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        o1 = make_order(oid_counter, Side.BID, 100.0, 100)
        o2 = make_order(oid_counter, Side.BID, 99.0, 200)
        book.add_order(o1)
        book.add_order(o2)
        book.cancel_order(o1.order_id)
        assert book.best_bid() == 99.0

    def test_cancel_best_bid_updates_best_bid(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        o1 = make_order(oid_counter, Side.BID, 100.0, 100)
        o2 = make_order(oid_counter, Side.BID, 99.0, 100)
        book.add_order(o1)
        book.add_order(o2)
        book.cancel_order(o1.order_id)
        assert book.best_bid() == 99.0


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------

class TestMatching:
    """Tests for immediate matching of aggressive orders."""

    def test_full_match_buy_against_ask(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        ask = make_order(oid_counter, Side.ASK, 100.0, 100)
        book.add_order(ask)

        buy = make_order(oid_counter, Side.BID, 100.0, 100)
        trades = book.add_order(buy)

        assert len(trades) == 1
        assert trades[0].qty == 100
        assert trades[0].price == 100.0
        assert book.best_ask() is None  # ask fully consumed

    def test_partial_match_buy(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        ask = make_order(oid_counter, Side.ASK, 100.0, 50)
        book.add_order(ask)

        buy = make_order(oid_counter, Side.BID, 100.0, 100)
        trades = book.add_order(buy)

        assert len(trades) == 1
        assert trades[0].qty == 50
        # Remaining 50 qty of buy should rest in book
        assert book.best_bid() == 100.0

    def test_match_price_is_passive_price(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        """Match price = passive order's price (price-time priority)."""
        ask = make_order(oid_counter, Side.ASK, 100.0, 100)
        book.add_order(ask)

        # Aggressive buy willing to pay 101.0
        buy = make_order(oid_counter, Side.BID, 101.0, 100)
        trades = book.add_order(buy)

        assert trades[0].price == 100.0  # Executed at passive (ask) price

    def test_no_match_bid_below_ask(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        ask = make_order(oid_counter, Side.ASK, 101.0, 100)
        book.add_order(ask)

        buy = make_order(oid_counter, Side.BID, 100.0, 100)
        trades = book.add_order(buy)

        assert trades == []
        assert book.best_bid() == 100.0
        assert book.best_ask() == 101.0

    def test_fifo_within_price_level(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        """First-in, first-out at the same price level."""
        ask1 = make_order(oid_counter, Side.ASK, 100.0, 60)
        ask2 = make_order(oid_counter, Side.ASK, 100.0, 60)
        book.add_order(ask1)
        book.add_order(ask2)

        buy = make_order(oid_counter, Side.BID, 100.0, 60)
        trades = book.add_order(buy)

        assert len(trades) == 1
        assert trades[0].passive_id == ask1.order_id  # ask1 filled first

    def test_multi_level_match(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        """Buy sweeps through multiple ask price levels."""
        book.add_order(make_order(oid_counter, Side.ASK, 100.0, 50))
        book.add_order(make_order(oid_counter, Side.ASK, 100.5, 50))
        book.add_order(make_order(oid_counter, Side.ASK, 101.0, 50))

        buy = make_order(oid_counter, Side.BID, 101.0, 150)
        trades = book.add_order(buy)

        assert len(trades) == 3
        assert book.best_ask() is None  # All asks consumed


# ---------------------------------------------------------------------------
# Depth tests
# ---------------------------------------------------------------------------

class TestDepth:
    """Tests for book depth queries."""

    def test_depth_returns_correct_levels(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        book.add_order(make_order(oid_counter, Side.BID, 100.0, 100))
        book.add_order(make_order(oid_counter, Side.BID, 99.0, 200))
        book.add_order(make_order(oid_counter, Side.ASK, 101.0, 150))
        book.add_order(make_order(oid_counter, Side.ASK, 102.0, 250))

        depth = book.depth(5)
        assert depth["bids"][0] == (100.0, 100)
        assert depth["bids"][1] == (99.0, 200)
        assert depth["asks"][0] == (101.0, 150)
        assert depth["asks"][1] == (102.0, 250)

    def test_depth_aggregates_qty_at_same_price(
        self, book: LimitOrderBook, oid_counter: list[int]
    ) -> None:
        book.add_order(make_order(oid_counter, Side.ASK, 101.0, 100))
        book.add_order(make_order(oid_counter, Side.ASK, 101.0, 200))

        depth = book.depth(5)
        assert depth["asks"][0] == (101.0, 300)

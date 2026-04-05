# Order Routing System — System Design

## Interview Question

> **"Design an order routing system that routes orders to multiple exchanges simultaneously with Smart Order Routing (SOR) logic. The system must minimise market impact and execution cost."**

*Typical at: Jane Street, Optiver, Citadel, Virtu*

---

## 1. Requirements Clarification

| Question | Assumption |
|----------|-----------|
| Order types? | Market, Limit, IOC, FOK, Iceberg |
| Number of venues? | 5–15 (NYSE, NASDAQ, CBOE, BATS, IEX, dark pools) |
| Order rate? | 10k–100k orders/second peak |
| Latency target? | < 100 µs from decision to wire (co-located) |
| Regulatory? | MiFID II (EU) / Reg NMS (US) best execution obligation |
| Failure handling? | Venue downtime, partial fills, rejects |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORDER ROUTING SYSTEM (ORS)                          │
│                                                                              │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────────────────────────┐  │
│  │ Strategy │   │    Order    │   │           Smart Order Router         │  │
│  │ (alpha   │──▶│  Manager   │──▶│                                      │  │
│  │  signal) │   │  (risk,    │   │  ┌───────────┐  ┌────────────────┐   │  │
│  └──────────┘   │  validate) │   │  │ Venue     │  │ Algo           │   │  │
│                 └─────────────┘   │  │ Selector  │  │ (TWAP/VWAP/   │   │  │
│                                   │  └─────┬─────┘  │  Iceberg)     │   │  │
│                                   │        │         └───────┬───────┘   │  │
│                                   └────────┼─────────────────┼───────────┘  │
│                                            │                 │              │
│         ┌──────────────────────────────────┤                 │              │
│         │           ┌──────────────────────┘                 │              │
│  ┌──────▼──┐  ┌─────▼────┐  ┌──────────┐  ┌────────────────▼──────────┐   │
│  │ NYSE    │  │ NASDAQ   │  │  CBOE    │  │  Dark Pools (IEX, Lit)    │   │
│  │ Gateway │  │ Gateway  │  │ Gateway  │  │  Gateways                 │   │
│  └──────┬──┘  └─────┬────┘  └────┬─────┘  └───────────────┬───────────┘   │
│         │           │             │                         │               │
│         └───────────┴─────────────┴─────────────────────────┘               │
│                                   ↓                                          │
│                          Exchange Networks (FIX/OUCH/ETI)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Smart Order Router (SOR) Logic

### 3.1 Venue Selection Algorithm

The SOR must decide: which venues, in what order, what quantity on each?

```python
from dataclasses import dataclass

@dataclass
class VenueQuote:
    venue_id: str
    price: float
    available_qty: int
    fee_per_share: float  # exchange fees (make/take)
    latency_us: float     # co-located round-trip to this venue

def select_venues(
    order_side: str,      # 'B' or 'S'
    order_qty: int,
    venue_quotes: list[VenueQuote],
    max_venues: int = 3,
) -> list[tuple[str, int]]:
    """
    Smart venue selection: optimise for best net price including fees.

    Net price (buy) = ask_price + fee_per_share
    Net price (sell) = bid_price - fee_per_share

    Return list of (venue_id, qty) pairs.
    """
    if order_side == 'B':
        ranked = sorted(venue_quotes, key=lambda v: v.price + v.fee_per_share)
    else:
        ranked = sorted(venue_quotes, key=lambda v: -(v.price - v.fee_per_share))

    allocation = []
    remaining = order_qty
    for venue in ranked[:max_venues]:
        if remaining <= 0:
            break
        fill_qty = min(remaining, venue.available_qty)
        allocation.append((venue.venue_id, fill_qty))
        remaining -= fill_qty

    return allocation
```

### 3.2 Reg NMS / MiFID II Compliance

In the US (Reg NMS), you must not trade at a price inferior to the National Best Bid/Offer (NBBO):
- **Trade-through protection**: If NYSE shows 100.00 bid and you sell at 99.99, that's a violation.
- SOR must check consolidated quote before routing.

```python
def check_reg_nms(order_price: float, nbbo_bid: float, nbbo_ask: float, side: str) -> bool:
    """Return True if the order price is compliant with Reg NMS."""
    if side == 'S':
        return order_price >= nbbo_bid  # cannot sell below NBBO bid
    else:
        return order_price <= nbbo_ask  # cannot buy above NBBO ask
```

---

## 4. Order Lifecycle and State Machine

```
NEW → VALIDATED → ROUTED → PARTIAL_FILL → FILLED
                         ↘ REJECTED (by venue)
                         ↘ CANCELLED (by client)
                         ↘ EXPIRED (IOC/FOK timeout)
```

### 4.1 Order State Management

```python
from enum import Enum

class OrderStatus(Enum):
    NEW = "NEW"
    PENDING_ROUTE = "PENDING_ROUTE"
    SENT = "SENT"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderManager:
    """Tracks all live orders. Must be O(1) for status updates."""

    def __init__(self):
        self._orders: dict[int, dict] = {}  # order_id → order state

    def on_new_order(self, order_id: int, qty: int, price: float) -> None:
        self._orders[order_id] = {
            "status": OrderStatus.NEW,
            "qty": qty,
            "filled_qty": 0,
            "price": price,
        }

    def on_fill(self, order_id: int, filled_qty: int, fill_price: float) -> None:
        order = self._orders[order_id]
        order["filled_qty"] += filled_qty
        if order["filled_qty"] >= order["qty"]:
            order["status"] = OrderStatus.FILLED
        else:
            order["status"] = OrderStatus.PARTIAL_FILL
```

---

## 5. Gateway Design (Exchange Connectivity)

Each venue requires a dedicated gateway:

```
Strategy Order → Order Router → Venue Gateway → Exchange (FIX/OUCH/ETI)
                                      ↑
                              Execution Report
```

### 5.1 Protocol Selection

| Venue | Protocol | Latency | Notes |
|-------|---------|---------|-------|
| NASDAQ | OUCH 4.2 | < 50 µs | Binary, ultra-fast |
| NYSE | FIX 4.4 | ~100 µs | Text, robust |
| CME | iLink 3 | < 100 µs | Binary FIX variant |
| CBOE | BOE 2.0 | < 50 µs | Binary Order Entry |
| IEX | FIXATDL | ~350 µs | Speed bump (IEX crumple) |

### 5.2 Gateway Implementation (Python sketch)

```python
import asyncio
import struct

class OUCHGateway:
    """NASDAQ OUCH 4.2 order entry gateway (async)."""

    ENTER_ORDER_FMT = struct.Struct("!c14scc8sIIHcc")  # simplified

    async def send_order(self, order_id: str, symbol: str, side: str,
                         qty: int, price: int) -> None:
        """Send a New Order (type 'O') via OUCH protocol."""
        msg = self.ENTER_ORDER_FMT.pack(
            b'O',                          # msg type: Enter Order
            order_id.encode().ljust(14),   # order token (14 bytes)
            side.encode(),                 # B or S
            b'Y',                          # firm routed flag
            symbol.encode().ljust(8),      # symbol (8 bytes)
            qty,                           # shares (4 bytes)
            price,                         # price in 1/10000 cents (4 bytes)
            0,                             # time in force
            b'N',                          # display (N=non-displayed)
            b'1',                          # capacity (proprietary)
        )
        self._writer.write(msg)
        await self._writer.drain()
```

---

## 6. Latency Budget (Co-located System)

| Stage | Latency | Notes |
|-------|---------|-------|
| Signal → Order decision | 1–10 µs | Alpha model computation |
| Risk check | 0.5–2 µs | Pre-computed limits |
| SOR venue selection | 0.5–1 µs | In-memory NBBO lookup |
| Serialise to binary | 0.1–0.5 µs | struct.pack or C extension |
| TCP send (kernel path) | 5–20 µs | With TCP_NODELAY |
| TCP send (kernel bypass) | 0.5–2 µs | Solarflare / DPDK |
| Exchange processing | 10–50 µs | Varies by venue |
| **Total (kernel bypass)** | **~15–70 µs** | |
| **Total (standard)** | **~20–80 µs** | |

---

## 7. Risk Controls (Pre-Trade)

Every order must pass risk checks **before** routing. These checks must be O(1):

| Check | Implementation |
|-------|---------------|
| Order size limit | Compare qty to pre-configured max |
| Position limit | Maintain running position counter |
| Daily loss limit | Subtract P&L from daily budget |
| Symbol blacklist | Hash set lookup |
| Price reasonability | ±5% from last trade |
| Rate limit | Token bucket per venue |

---

## 8. Failure Handling

| Scenario | Handling |
|----------|----------|
| Venue down | Mark venue offline, reroute to next-best |
| Partial fill | Track remaining qty, optionally re-route |
| Reject | Log reason, alert ops, notify strategy |
| Duplicate order | Dedup by client_order_id (hash set) |
| Network partition | Queue orders, replay when reconnected |
| Clock skew | Use exchange timestamp in execution reports |

---

## 9. Python-Specific Notes

- Python is suitable for: order management logic, risk checks, monitoring.
- Python is NOT suitable for: sub-100 µs order submission (use C++ or Rust).
- If using Python for the gateway, use `asyncio` + `struct` — never `json`.
- Profile with `py-spy` to find hot paths; move to Cython if needed.
- Pre-allocate all order objects at startup; use an object pool.

---

## Sample Interview Answer Structure

1. **Clarify** (~2 min): order types, venues, latency targets, compliance.
2. **Sketch architecture** (~5 min): strategy → OMS → SOR → gateways.
3. **SOR algorithm** (~5 min): venue selection, Reg NMS/MiFID II compliance.
4. **Order lifecycle** (~3 min): state machine, partial fills.
5. **Latency budget** (~3 min): walk through each stage.
6. **Failure modes** (~3 min): venue down, partial fill, reject handling.
7. **Scaling** (~2 min): horizontal scale by symbol partitioning.

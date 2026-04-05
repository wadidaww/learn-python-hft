# Market Data Feed Handler — System Design

## Interview Question

> **"Design a system to ingest 100,000 messages per second from multiple exchange feeds, normalise them to a common format, and store them efficiently. Discuss backpressure handling."**

*Typical at: Citadel, Two Sigma, Jump Trading, HRT*

---

## 1. Requirements Clarification

Before designing, clarify:

| Question | Assumption |
|----------|-----------|
| How many simultaneous feeds? | 10–20 (NASDAQ, NYSE, CBOE, CME, ICE...) |
| Message types? | Quotes, Trades, Order book deltas |
| Latency target for normalisation? | < 1 µs tick-to-strategy |
| Storage type? | In-memory (hot) + disk (warm) + S3 (cold) |
| Consumers? | Strategy engines, risk system, UI, historical replay |
| Reliability? | Must detect and request gap-fills for missing sequences |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MARKET DATA PLATFORM                         │
│                                                                      │
│  ┌──────────┐   ┌─────────────────────────┐   ┌───────────────────┐ │
│  │ Multicast│   │   Feed Handler Process  │   │  Distribution Bus │ │
│  │ Receiver │──▶│  Decode | Normalise |   │──▶│  (Ring Buffer /   │ │
│  │ (per feed│   │  Seq Check | Gap Fill   │   │   Aeron / ZMQ)    │ │
│  └──────────┘   └─────────────────────────┘   └────────┬──────────┘ │
│                                                         │            │
│                          ┌──────────────────────────────┤            │
│                          │                              │            │
│               ┌──────────▼──┐              ┌───────────▼──┐         │
│               │  Strategy   │              │  Storage      │         │
│               │  Engines    │              │  (ArcticDB /  │         │
│               │ (per-symbol)│              │   InfluxDB)   │         │
│               └─────────────┘              └───────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feed Handler Design

### 3.1 Network Layer

```
NIC (kernel bypass preferred)
  ↓ Solarflare/DPDK bypass kernel TCP/IP stack
  ↓ Or: standard UDP socket with SO_BUSY_POLL + large SO_RCVBUF (8 MB)
  ↓
Receive thread (pinned to CPU core 0, SCHED_FIFO priority 90)
  ↓ Copies raw packet into pre-allocated ring buffer slot
  ↓ No heap allocation in this path
  ↓
Decode thread (CPU core 1)
  ↓ Reads from ring buffer, decodes binary protocol (ITCH/OPRA/MDP)
  ↓ Writes normalised NormalTick to output ring buffer
  ↓
Fan-out thread (CPU core 2)
  ↓ Reads normalised ticks, routes to per-symbol queues
```

### 3.2 Sequence Number Tracking

Every exchange feed has sequence numbers. **Gaps must be detected and recovered**:

```python
class SequenceTracker:
    """Track sequence numbers and detect gaps."""

    def __init__(self, feed_id: str) -> None:
        self.feed_id = feed_id
        self.expected_seq: int = 1
        self.gap_count: int = 0

    def process(self, seq: int) -> list[int]:
        """Process incoming sequence number, return list of missing seqs."""
        missing = []
        if seq > self.expected_seq:
            missing = list(range(self.expected_seq, seq))
            self.gap_count += len(missing)
        self.expected_seq = seq + 1
        return missing
```

### 3.3 Normalised Message Format

All exchange-specific formats (ITCH, OPRA, MDP3) are decoded into a common struct:

```
NormalTick (64 bytes):
  - feed_id:      uint8
  - msg_type:     uint8  (QUOTE=1, TRADE=2, STATUS=3)
  - symbol_hash:  uint32 (pre-computed hash for routing)
  - timestamp_ns: uint64 (exchange timestamp from feed header)
  - recv_ns:      uint64 (local receive timestamp)
  - bid:          double
  - ask:          double
  - bid_size:     uint32
  - ask_size:     uint32
  - last:         double
  - last_size:    uint32
  - reserved:     4 bytes padding
```

---

## 4. Backpressure Handling

Backpressure occurs when consumers are slower than producers. Strategies:

### 4.1 Drop Policy (for market data)

Market data is lossy by nature — a dropped quote is replaced by the next quote.
If the consumer falls behind, **drop old data, not new data**.

```python
def publish_with_drop(ring_buffer: RingBuffer, tick: NormalTick) -> bool:
    """Publish tick, dropping if buffer full (latest always wins)."""
    if ring_buffer.is_full():
        ring_buffer.drop_oldest()  # discard stale quote
    return ring_buffer.try_write(tick)
```

### 4.2 Slow Consumer Detection

Monitor queue depth. Alert if depth > threshold:

```python
WARN_DEPTH = 1000    # warn at 1k queued messages
CRITICAL_DEPTH = 5000  # circuit-break at 5k

if queue.depth > CRITICAL_DEPTH:
    # Strategy is too slow — pause feed, alert ops
    alert("FEED HANDLER: Consumer overloaded, circuit-breaking")
```

### 4.3 Time-based Aggregation

If a consumer cannot keep up at tick rate, aggregate to snapshots:
- Every 100 ms: publish best bid/ask snapshot (not every tick)
- This trades latency for throughput

---

## 5. Storage Design

| Tier | Technology | Latency | Retention | Use Case |
|------|-----------|---------|-----------|----------|
| Hot  | In-memory ring buffer | < 1 µs | Last 1M ticks | Strategy, real-time risk |
| Warm | ArcticDB / InfluxDB | < 1 ms | 30 days | Backtesting, T+1 analysis |
| Cold | S3 (Parquet) | 100 ms | Years | Research, model training |

### Write Path (hot tier)
```
NormalTick → SPSC Ring Buffer (shared memory)
  Consumer: Strategy engine reads directly (zero-copy)
  Consumer: Async writer flushes to ArcticDB every 100 ms
```

---

## 6. Performance Numbers

| Metric | Target | Notes |
|--------|--------|-------|
| Input rate | 100k msg/s | 10 feeds × 10k msg/s each |
| Normalisation latency | < 1 µs | CPU-pinned, no allocation |
| End-to-end (recv → strategy) | < 5 µs | Kernel-bypass = 1 µs |
| Storage throughput | > 200k msg/s | Async batched writes |
| Gap detection | < 1 message | Per-sequence tracking |

---

## 7. Failure Modes and Mitigations

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Feed disconnect | Heartbeat timeout (3 s) | Reconnect with gap-fill request |
| Sequence gap | Seq number jump | UDP retransmit channel (NASDAQ GapFill) |
| Consumer overload | Queue depth alarm | Circuit-break, drop, alert |
| Clock drift | NTP monitoring | Prefer exchange timestamps over local |
| Memory leak | RSS monitoring | Object pool, pre-allocation |

---

## 8. Python-Specific Recommendations

1. **Do not** use Python for the receive/decode path if targeting < 5 µs.
2. **Use Python** for: configuration, monitoring, gap-fill logic, storage writes.
3. For normalisation in Python, use `struct.unpack` — not JSON, not FIX text.
4. Use `multiprocessing` with `shared_memory` for zero-copy consumer handoff.
5. Profile with `perf stat` / `perf record` to find kernel time vs user time.

---

## Sample Interview Answer Structure

1. **Clarify requirements** (~2 min): rates, latency targets, consumers.
2. **Sketch architecture** (~5 min): draw the box diagram.
3. **Deep-dive on critical path** (~5 min): recv → decode → distribute.
4. **Discuss backpressure** (~3 min): drop policy, queue depth monitoring.
5. **Storage design** (~3 min): hot/warm/cold tiers.
6. **Failure modes** (~2 min): reconnect, gap-fill, circuit-break.

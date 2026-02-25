# Python HFT Mastery

> **Disclaimer:** Python is commonly used for research, tooling, and infrastructure in HFT firms; C++ (and sometimes Rust) is used for the critical, ultra-low-latency execution paths. This repository teaches you *where Python fits*, *how to squeeze performance out of it*, and *how to reason about low-latency systems* — exactly what HFT firms test in interviews.

A structured, technically rigorous learning path taking a proficient Python engineer to mastery of **Low-Latency Systems** and **HFT Interview Preparation**.

---

## Table of Contents

- [Repository Goal](#repository-goal)
- [Prerequisites](#prerequisites)
- [Learning Path](#learning-path)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Key Concepts Covered](#key-concepts-covered)
- [Target Firms & Interview Style](#target-firms--interview-style)

---

## Repository Goal

This repository covers six pillars of Python for HFT:

1. **Python Internals** — Memory management, CPython object model, GIL implications.
2. **Concurrency** — `asyncio`, multiprocessing, threading, and lock-free structures.
3. **Low-Latency Networking** — Socket programming, UDP multicast, kernel bypass concepts.
4. **Data Structures** — Efficient order books, market data parsing, binary protocols.
5. **Optimization** — Profiling, Cython, Numba, vectorization, avoiding GC pauses.
6. **Interview Prep** — Coding challenges and system design questions from Jane Street, Citadel, HRT, and Optiver.

---

## Prerequisites

### Operating System Knowledge
- Understand process vs thread model (Linux `clone(2)` syscall semantics)
- Familiarity with CPU affinity (`taskset`, `numactl`) and NUMA topology
- Basic understanding of Linux scheduler (CFS) and real-time priorities (`SCHED_FIFO`)
- Knowledge of memory-mapped files (`mmap`) and huge pages (THP / explicit HugeTLBfs)

### Networking Basics
- TCP/IP stack: the cost of the kernel network path (soft IRQ → socket buffer → userspace copy)
- UDP vs TCP trade-offs in a market-data context (multicast feeds, TCP order entry)
- Understand what a NIC does: DMA, RSS (Receive Side Scaling), interrupt coalescing
- Familiarity with `ethtool`, `ss`, and `/proc/net/` diagnostics

### Python Knowledge
- Comfortable with Python 3.10+ features: structural pattern matching, `match`, `ParamSpec`
- Understanding of the CPython bytecode (`dis` module)
- Familiarity with `ctypes` and the C-extension ABI

---

## Learning Path

| Week | Topic | Directory |
|------|-------|-----------|
| 1 | CPython Internals: memory model, GIL, object lifecycle | `src/internals/` |
| 2 | Concurrency primitives: asyncio, threads, processes | `src/concurrency/` |
| 3 | Networking: sockets, UDP multicast, TCP options | `src/networking/` |
| 4 | Market data structures: order book, FIX, binary protocols | `src/market_data/` |
| 5 | Optimization: Cython, NumPy, GC tuning, profiling | `src/optimization/` |
| 6 | Benchmarking: latency histograms, throughput tests | `benchmarks/` |
| 7–8 | Interview preparation: coding challenges & system design | `interviews/` |

---

## Directory Structure

```
python-hft-mastery/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── benchmarks/
│   ├── __init__.py
│   ├── latency_test.py
│   └── throughput_test.py
├── src/
│   ├── __init__.py
│   ├── internals/
│   │   ├── memory_layout.py
│   │   ├── gil_behavior.py
│   │   └── object_pool.py
│   ├── concurrency/
│   │   ├── async_io_engine.py
│   │   ├── multiproc_queue.py
│   │   └── lock_free_ring_buffer.py
│   ├── networking/
│   │   ├── udp_multicast_listener.py
│   │   ├── tcp_order_server.py
│   │   └── socket_options.py
│   ├── market_data/
│   │   ├── order_book.py
│   │   ├── fix_parser.py
│   │   └── binary_protocol.py
│   └── optimization/
│       ├── cython_examples.pyx
│       ├── numpy_vectors.py
│       └── gc_tuning.py
├── interviews/
│   ├── coding_challenges/
│   │   ├── latency_sensitive_algo.py
│   │   └── concurrency_bug_fix.py
│   └── system_design/
│       ├── market_data_feed_handler.md
│       └── order_routing_system.md
└── tests/
    ├── __init__.py
    ├── test_order_book.py
    └── test_networking.py
```

---

## Quick Start

```bash
# 1. Create a virtual environment (Python 3.10+)
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode
pip install -e .

# 4. Run the internals demo
python src/internals/memory_layout.py

# 5. Run the order book demo
python src/market_data/order_book.py

# 6. Run the benchmarks
python benchmarks/latency_test.py
python benchmarks/throughput_test.py

# 7. Run tests
pytest tests/ -v
```

---

## Key Concepts Covered

### The GIL (Global Interpreter Lock)
The GIL is a mutex in CPython that prevents multiple native threads from executing Python bytecode simultaneously. For **CPU-bound** tasks, threads provide no parallelism. Use `multiprocessing` or process-based architectures instead. For **I/O-bound** tasks, the GIL is released during syscalls, so `asyncio` or threads work fine.

See: `src/internals/gil_behavior.py`

### Memory Locality
Python objects are heap-allocated PyObjects scattered across memory. A list of floats stores 64-bit *pointers* to boxed `float` objects — terrible for CPU cache. Use `array.array`, `numpy.ndarray`, or C structs via `ctypes`/`struct` for cache-friendly access patterns.

See: `src/internals/memory_layout.py`

### Context Switching
OS thread context switches cost ~1–10 µs. Async coroutines switch in ~100 ns (cooperative, no kernel involvement). For market-data fan-out with thousands of streams, `asyncio` wins. For CPU-bound parallel computation, use `multiprocessing`.

See: `src/concurrency/async_io_engine.py`

### Serialization Performance
| Format | Encode | Decode | Notes |
|--------|--------|--------|-------|
| JSON | ~5 µs | ~10 µs | Human-readable, slow |
| FIX (text) | ~2 µs | ~3 µs | Industry standard, tag=value |
| Protobuf | ~0.5 µs | ~0.5 µs | Schema-based, compact |
| `struct.pack` | ~0.05 µs | ~0.05 µs | Zero-copy, no schema overhead |
| FlatBuffers | ~0.01 µs | ~0 µs | Zero-copy, no parse needed |

See: `src/market_data/binary_protocol.py`, `src/market_data/fix_parser.py`

### Kernel Bypass
Technologies like **DPDK** (Data Plane Development Kit) and **Solarflare/OpenOnload** bypass the Linux kernel TCP/IP stack entirely, cutting network latency from ~10 µs to ~1 µs RTT. Python can interact with these via CFFI/ctypes bindings, but the hot path should be C/C++. Python is used for configuration, monitoring, and research.

See: `src/networking/socket_options.py`

### Clocks
| Clock | Resolution | Monotonic | Use Case |
|-------|-----------|-----------|----------|
| `time.time()` | ~1 µs | No | Wall clock, NTP-adjusted |
| `time.perf_counter()` | ~1 ns | Yes | Benchmarking (process lifetime) |
| `time.perf_counter_ns()` | 1 ns | Yes | Benchmarking (integer, no float error) |
| `clock_gettime(CLOCK_MONOTONIC)` | ~1 ns | Yes | Cross-language benchmarking |
| `clock_gettime(CLOCK_REALTIME)` | ~1 ns | No | Timestamping with wall time |

See: `benchmarks/latency_test.py`

---

## Target Firms & Interview Style

| Firm | Style | Focus Areas |
|------|-------|-------------|
| **Jane Street** | Functional reasoning, OCaml mindset | Algorithms, type systems, reasoning under uncertainty |
| **Citadel** | Systems depth, scalability | Distributed systems, low-latency, market microstructure |
| **HRT (Hudson River Trading)** | Pure performance | Lock-free structures, kernel internals, profiling |
| **Optiver** | Market making, numerical | Options pricing, numerical stability, speed |
| **Two Sigma** | Data-driven, ML infra | Pipeline efficiency, distributed compute |
| **Jump Trading** | Low-level systems | Network stacks, FPGA awareness, C++ depth |

See: `interviews/` for specific challenges and system design questions.

---

## Contributing

This is a learning resource. If you find errors or have improvements, PRs are welcome. Focus on correctness, performance rationale, and educational value.

---

## License

MIT License — see `LICENSE` for details.

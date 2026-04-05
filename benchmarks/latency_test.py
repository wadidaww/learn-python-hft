"""
benchmarks/latency_test.py
==========================
Micro-benchmark harness measuring:
  - Raw Python function-call overhead
  - Context-switch cost estimate (thread wake-up round-trip)
  - Network loopback RTT via localhost UDP

HFT relevance
-------------
Latency is measured in *nanoseconds* on real trading systems.
`time.perf_counter_ns()` is the correct clock: it is monotonic and
returns an integer (no floating-point rounding), matching what you
would use in a C++ `std::chrono::steady_clock` benchmark.

Key insight: a single Python function call costs ~100 ns; a C function
called via ctypes costs ~300 ns (call overhead dominates).  Hot paths
must be tight loops in C/Cython/NumPy, not Python-level dispatches.
"""

from __future__ import annotations

import socket
import threading
import time
from statistics import mean, median, stdev
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_samples(fn: Callable[[], None], n: int = 10_000) -> list[int]:
    """Collect *n* latency samples (nanoseconds) for *fn*.

    Args:
        fn: Zero-argument callable to benchmark.
        n:  Number of samples to collect.

    Returns:
        Sorted list of elapsed times in nanoseconds.
    """
    samples: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    samples.sort()
    return samples


def _print_stats(label: str, samples: list[int]) -> None:
    """Print a human-readable latency summary.

    Args:
        label:   Descriptive name for the benchmark.
        samples: Sorted list of elapsed times in nanoseconds.
    """
    p50 = samples[len(samples) // 2]
    p99 = samples[int(len(samples) * 0.99)]
    p999 = samples[int(len(samples) * 0.999)]
    print(
        f"[{label}]"
        f"  min={samples[0]} ns"
        f"  p50={p50} ns"
        f"  p99={p99} ns"
        f"  p99.9={p999} ns"
        f"  max={samples[-1]} ns"
        f"  mean={mean(samples):.1f} ns"
        f"  stdev={stdev(samples):.1f} ns"
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_function_call_overhead(n: int = 100_000) -> None:
    """Measure the overhead of a minimal Python function call.

    A no-op function call in CPython takes roughly 50-150 ns depending
    on the Python version and hardware.  This sets the baseline for any
    higher-level dispatch logic.
    """
    def noop() -> None:
        pass

    samples = _collect_samples(noop, n)
    _print_stats("function_call_overhead", samples)


def bench_thread_wakeup_latency(n: int = 1_000) -> None:
    """Estimate the cost of waking a sleeping thread.

    Uses a Condition variable: the main thread signals, the worker
    thread wakes up and records the round-trip time.

    This approximates the *best-case* latency for a producer→consumer
    design using OS threads (no busy-poll).  Typical values: 2–20 µs
    on Linux depending on scheduler configuration.

    TODO: Compare with busy-poll (spinning on a shared atomic flag
          using ctypes) to see the ~100 ns spin-wait latency.
    """
    cond = threading.Condition()
    latencies: list[int] = []
    ready = threading.Event()
    done = threading.Event()

    def worker() -> None:
        ready.set()
        for _ in range(n):
            with cond:
                cond.wait()
            latencies.append(time.perf_counter_ns())

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ready.wait()

    send_times: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        send_times.append(t0)
        with cond:
            cond.notify()
        time.sleep(0)  # yield to let worker run

    t.join(timeout=5)

    rtts = [r - s for r, s in zip(latencies[:len(send_times)], send_times)]
    if rtts:
        rtts.sort()
        _print_stats("thread_wakeup_latency", rtts)


def bench_udp_loopback_rtt(n: int = 1_000, port: int = 19876) -> None:
    """Measure UDP loopback round-trip time on localhost.

    Sends a 64-byte datagram to itself and measures the RTT.
    Typical loopback RTT: 20–100 µs (kernel path).

    HFT note: With kernel bypass (DPDK / Solarflare OpenOnload), RTT
    drops to 1–3 µs.  Python cannot achieve this natively, but it is
    important to understand *where* the latency lives.

    TODO: Repeat with `SO_BUSY_POLL` socket option (Linux 3.11+) to
          reduce interrupt latency.
    """
    payload = b"x" * 64
    addr = ("127.0.0.1", port)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind(addr)
    server_sock.settimeout(1.0)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_sock.settimeout(1.0)

    stop_event = threading.Event()

    def echo_server() -> None:
        while not stop_event.is_set():
            try:
                data, src = server_sock.recvfrom(256)
                server_sock.sendto(data, src)
            except socket.timeout:
                pass

    thread = threading.Thread(target=echo_server, daemon=True)
    thread.start()

    rtts: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        client_sock.sendto(payload, addr)
        client_sock.recvfrom(256)
        t1 = time.perf_counter_ns()
        rtts.append(t1 - t0)

    stop_event.set()
    server_sock.close()
    client_sock.close()

    rtts.sort()
    _print_stats("udp_loopback_rtt", rtts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Python HFT Latency Benchmarks")
    print("=" * 60)
    bench_function_call_overhead()
    bench_thread_wakeup_latency()
    bench_udp_loopback_rtt()

"""
src/optimization/numpy_vectors.py
====================================
NumPy vectorisation vs Python loops for financial calculations.

HFT relevance
-------------
NumPy wraps BLAS/LAPACK and uses CPU SIMD instructions (SSE2, AVX2, AVX-512).
A vectorised NumPy operation on 1M elements completes in microseconds;
a pure Python loop takes milliseconds (100–1000× slower).

Rules for HFT Python:
  1. NEVER loop over price/volume arrays in Python — use NumPy operations.
  2. Avoid creating temporary arrays in hot paths (use `out=` parameter).
  3. Use `np.float32` where precision allows — halves memory, doubles throughput.
  4. Prefer contiguous (C-order) arrays for cache efficiency.
  5. Use `numba.jit` for non-trivial loops that NumPy can't vectorise directly.

This module benchmarks:
  - Python loop vs NumPy for VWAP, moving average, z-score
  - In-place operations to avoid allocation
  - Float64 vs Float32 throughput
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. VWAP: Python loop vs NumPy
# ---------------------------------------------------------------------------

def vwap_python_loop(prices: list[float], volumes: list[float]) -> float:
    """Compute VWAP using a pure Python loop.

    This is the naive implementation — correct but slow.
    DO NOT use this on tick data in production.

    Args:
        prices:  List of trade prices.
        volumes: List of trade volumes.

    Returns:
        Volume-weighted average price.
    """
    pv_sum = 0.0
    v_sum = 0.0
    for p, v in zip(prices, volumes):
        pv_sum += p * v
        v_sum += v
    return pv_sum / v_sum if v_sum else 0.0


def vwap_numpy(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Compute VWAP using NumPy vectorised operations.

    `np.dot(prices, volumes)` is a single C-level BLAS daxpy call.
    On 1M elements: ~1 ms Python loop → ~10 µs NumPy (100× faster).

    Args:
        prices:  1D float64 array of trade prices.
        volumes: 1D float64 array of trade volumes.

    Returns:
        Volume-weighted average price.
    """
    v_sum = volumes.sum()
    if v_sum == 0.0:
        return 0.0
    return float(np.dot(prices, volumes) / v_sum)


# ---------------------------------------------------------------------------
# 2. Moving average: Python loop vs NumPy
# ---------------------------------------------------------------------------

def moving_average_python(prices: list[float], window: int) -> list[float]:
    """Compute simple moving average using Python loop.

    Args:
        prices: List of prices.
        window: Look-back window.

    Returns:
        List of SMA values (NaN for first window-1 elements).
    """
    n = len(prices)
    result = [float("nan")] * n
    if window <= 0 or window > n:
        return result
    window_sum = sum(prices[:window])
    result[window - 1] = window_sum / window
    for i in range(window, n):
        window_sum += prices[i] - prices[i - window]
        result[i] = window_sum / window
    return result


def moving_average_numpy(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute simple moving average using np.convolve.

    `np.convolve` uses FFT or direct convolution depending on size.
    For large arrays, `np.cumsum` is faster and avoids the full convolution.

    Args:
        prices: 1D float64 price array.
        window: Look-back window size.

    Returns:
        1D array of SMA values; first (window-1) elements are NaN.
    """
    result = np.full(len(prices), np.nan)
    if window <= 0 or window > len(prices):
        return result
    cumsum = np.cumsum(prices)
    result[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0.0], cumsum[:-window]])) / window
    return result


# ---------------------------------------------------------------------------
# 3. Z-score (rolling normalisation)
# ---------------------------------------------------------------------------

def zscore_numpy(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-score: (price - rolling_mean) / rolling_std.

    Used in mean-reversion strategies to identify statistically
    extreme prices.  NumPy rolling operations are much faster than
    pandas for raw calculation (no index overhead).

    Args:
        prices: 1D float64 price array.
        window: Rolling window size.

    Returns:
        1D array of z-scores; first (window-1) elements are NaN.
    """
    n = len(prices)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_slice = prices[i - window + 1 : i + 1]
        std = window_slice.std()
        if std > 0:
            result[i] = (prices[i] - window_slice.mean()) / std
    return result


def zscore_numpy_vectorised(prices: np.ndarray, window: int) -> np.ndarray:
    """Faster z-score using pre-computed rolling statistics.

    Uses stride tricks to avoid explicit Python loops.
    Suitable for batch recalculation, not real-time streaming.

    Args:
        prices: 1D float64 price array.
        window: Rolling window size.

    Returns:
        1D array of z-scores.
    """
    n = len(prices)
    result = np.full(n, np.nan)
    if window > n:
        return result

    # Use as_strided to create rolling windows view (zero-copy)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(prices, window)  # shape: (n-window+1, window)
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    valid = stds > 0
    z = np.where(valid, (prices[window - 1:] - means) / np.where(valid, stds, 1.0), np.nan)
    result[window - 1:] = z
    return result


# ---------------------------------------------------------------------------
# 4. In-place operations (avoid allocation)
# ---------------------------------------------------------------------------

def normalise_inplace(arr: np.ndarray, out: np.ndarray) -> None:
    """Normalise array to [0, 1] range, writing to a pre-allocated output.

    Using `out=` avoids creating a temporary array.  In a loop running
    1M iterations, this eliminates 1M malloc/free pairs.

    Args:
        arr: Input array.
        out: Pre-allocated output array (same shape and dtype as arr).
    """
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        out[:] = 0.0
        return
    np.subtract(arr, min_val, out=out)
    out /= (max_val - min_val)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_vwap(n: int = 1_000_000) -> None:
    """Benchmark VWAP: Python loop vs NumPy.

    Args:
        n: Array size.
    """
    prices_np = np.random.uniform(95.0, 105.0, n).astype(np.float64)
    volumes_np = np.random.uniform(100.0, 10_000.0, n).astype(np.float64)
    prices_list = prices_np.tolist()
    volumes_list = volumes_np.tolist()

    print(f"\n--- VWAP (n={n:,}) ---")

    t0 = time.perf_counter_ns()
    result_py = vwap_python_loop(prices_list, volumes_list)
    py_ns = time.perf_counter_ns() - t0
    print(f"  Python loop: {py_ns/1e6:.1f} ms  (result={result_py:.4f})")

    t0 = time.perf_counter_ns()
    result_np = vwap_numpy(prices_np, volumes_np)
    np_ns = time.perf_counter_ns() - t0
    print(f"  NumPy:       {np_ns/1e6:.1f} ms  (result={result_np:.4f})")
    print(f"  Speedup:     {py_ns / np_ns:.0f}×")


def bench_moving_average(n: int = 1_000_000, window: int = 50) -> None:
    """Benchmark SMA: Python loop vs NumPy.

    Args:
        n:      Array size.
        window: SMA window size.
    """
    prices_np = np.random.uniform(95.0, 105.0, n).astype(np.float64)
    prices_list = prices_np.tolist()

    print(f"\n--- Moving Average (n={n:,}, window={window}) ---")

    t0 = time.perf_counter_ns()
    result_py = moving_average_python(prices_list, window)
    py_ns = time.perf_counter_ns() - t0
    print(f"  Python loop: {py_ns/1e6:.1f} ms")

    t0 = time.perf_counter_ns()
    result_np = moving_average_numpy(prices_np, window)
    np_ns = time.perf_counter_ns() - t0
    print(f"  NumPy:       {np_ns/1e6:.1f} ms")
    print(f"  Speedup:     {py_ns / np_ns:.0f}×")


def bench_float32_vs_float64(n: int = 10_000_000) -> None:
    """Compare float32 vs float64 throughput.

    Float32 uses half the memory → better cache utilisation → higher throughput.
    Use float32 for large arrays where precision > 1e-6 is sufficient.

    Args:
        n: Array size.
    """
    arr64 = np.random.random(n).astype(np.float64)
    arr32 = arr64.astype(np.float32)

    print(f"\n--- float32 vs float64 (n={n:,}) ---")
    for dtype, arr in [("float64", arr64), ("float32", arr32)]:
        t0 = time.perf_counter_ns()
        _ = arr.sum()
        ns = time.perf_counter_ns() - t0
        print(f"  {dtype}: {ns/1e6:.2f} ms  (memory={arr.nbytes/1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== NumPy Vectorisation Benchmarks ===")
    bench_vwap()
    bench_moving_average()
    bench_float32_vs_float64()
    print(
        "\nKey takeaways:"
        "\n  1. NumPy is 10-100× faster than Python loops for numerical work."
        "\n  2. Use float32 to double throughput on large arrays."
        "\n  3. Use out= to avoid temporary allocations."
        "\n  4. For custom loops, use Cython or Numba."
    )

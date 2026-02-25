# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
src/optimization/cython_examples.pyx
=======================================
Cython examples demonstrating type declarations for HFT-relevant calculations.

HFT relevance
-------------
Cython compiles Python-like code to C extensions.  With type declarations,
a tight numerical loop runs at C speed (~10–100× faster than pure Python).

Key Cython concepts for HFT:
  1. `cdef double x` — C-typed local variable (no PyObject boxing)
  2. `cdef double[:] arr` — typed memoryview (C array access, no GIL needed)
  3. `with nogil:` — release the GIL during C computation
  4. `@cython.boundscheck(False)` — disable array bounds checking in hot loops

Build command:
    cythonize -i src/optimization/cython_examples.pyx
Or via setup.py:
    pip install -e ".[cython]"

Usage::
    from src.optimization.cython_examples import moving_average_cython
    result = moving_average_cython(prices, window=20)
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs


# ---------------------------------------------------------------------------
# 1. Moving average (C-speed tight loop)
# ---------------------------------------------------------------------------

def moving_average_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] prices,
    int window,
) -> cnp.ndarray:
    """Compute simple moving average using Cython typed memoryviews.

    This is 10–50× faster than the pure-Python loop equivalent because:
      - No PyObject boxing/unboxing for each array element
      - No bounds-checking overhead (disabled at top of file)
      - Compiled to tight C loop with potential SIMD vectorisation

    Args:
        prices: 1D array of price values.
        window: Look-back window size.

    Returns:
        1D numpy array of moving average values.  First `window-1`
        elements are NaN (insufficient history).
    """
    cdef int n = len(prices)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.full(n, np.nan)
    cdef double window_sum = 0.0
    cdef int i

    if window <= 0 or window > n:
        return result

    # Seed the first window
    for i in range(window):
        window_sum += prices[i]
    result[window - 1] = window_sum / window

    # Slide the window
    for i in range(window, n):
        window_sum += prices[i] - prices[i - window]
        result[i] = window_sum / window

    return result


# ---------------------------------------------------------------------------
# 2. VWAP calculation (type-declared, GIL-released)
# ---------------------------------------------------------------------------

def vwap_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] prices,
    cnp.ndarray[cnp.float64_t, ndim=1] volumes,
) -> double:
    """Compute Volume-Weighted Average Price (VWAP).

    VWAP = Σ(price_i × volume_i) / Σ(volume_i)

    With `with nogil:`, the GIL is released during the C loop, allowing
    other Python threads to run concurrently (useful in a multithreaded
    strategy engine).

    Args:
        prices:  Array of trade prices.
        volumes: Array of trade volumes (same length as prices).

    Returns:
        VWAP as a C double.
    """
    cdef double pv_sum = 0.0
    cdef double v_sum = 0.0
    cdef int n = len(prices)
    cdef int i
    cdef double[:] p_view = prices
    cdef double[:] v_view = volumes

    with nogil:
        for i in range(n):
            pv_sum += p_view[i] * v_view[i]
            v_sum += v_view[i]

    if v_sum == 0.0:
        return 0.0
    return pv_sum / v_sum


# ---------------------------------------------------------------------------
# 3. Order book price level aggregation (realistic hot-path example)
# ---------------------------------------------------------------------------

def aggregate_bid_qty(
    cnp.ndarray[cnp.float64_t, ndim=1] prices,
    cnp.ndarray[cnp.float64_t, ndim=1] qtys,
    double min_price,
) -> double:
    """Sum bid quantities at or above min_price.

    In a real system this scans the bid side of the order book to compute
    depth (total available liquidity above a price threshold).

    Args:
        prices:    Array of bid price levels.
        qtys:      Array of quantities at each price level.
        min_price: Include only prices >= min_price.

    Returns:
        Total bid quantity above min_price.
    """
    cdef double total = 0.0
    cdef int n = len(prices)
    cdef int i

    for i in range(n):
        if prices[i] >= min_price:
            total += qtys[i]

    return total

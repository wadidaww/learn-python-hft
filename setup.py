"""Setup script for python-hft-mastery.

Uses setuptools with optional Cython extension compilation.
Run `pip install -e .` for development mode.
Run `pip install -e ".[cython]"` to also compile the Cython extensions.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import find_packages, setup

# Attempt to build Cython extensions if Cython is available.
try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        "src/optimization/cython_examples.pyx",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    )
except ImportError:
    ext_modules = []

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="python-hft-mastery",
    version="0.1.0",
    description="Low-latency Python patterns and HFT interview preparation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    install_requires=[
        "numpy>=1.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "line_profiler>=4.1.0",
            "memory_profiler>=0.61.0",
        ],
        "cython": ["Cython>=3.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

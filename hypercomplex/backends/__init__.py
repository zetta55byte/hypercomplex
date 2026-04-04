"""
hypercomplex.backends
---------------------
Backend registry for hypercomplex arithmetic.

Available backends
------------------
numpy  : Pure NumPy (default). No extra dependencies.
jax    : JAX/XLA. Requires ``pip install "hcderiv[jax]"``. JIT-compiled.

Usage
-----
    from hypercomplex import hessian

    H = hessian(f, x)                          # NumPy (default)
    H = hessian(f, x, backend="numpy")         # NumPy explicit
    H = hessian(f, x, backend="jax")           # JAX, JIT on
    H = hessian(f, x, backend="jax", jit=False) # JAX, JIT off
"""

from __future__ import annotations
from typing import Literal

BackendName = Literal["numpy", "jax"]


def get_backend(name: BackendName = "numpy"):
    """
    Return the array module for the requested backend.

    Parameters
    ----------
    name : {"numpy", "jax"}

    Returns
    -------
    module
        numpy or jax.numpy

    Raises
    ------
    ImportError
        If JAX is requested but not installed.
    ValueError
        If the name is not recognised.
    """
    if name == "numpy":
        import numpy as xp

        return xp
    elif name == "jax":
        try:
            import jax.numpy as xp

            return xp
        except ImportError:
            raise ImportError(
                "JAX backend requires jax. " "Install with:  pip install 'hcderiv[jax]'"
            )
    else:
        raise ValueError(f"Unknown backend {name!r}. Choose 'numpy' or 'jax'.")


def is_jax(name: BackendName) -> bool:
    return name == "jax"


__all__ = ["get_backend", "is_jax", "BackendName"]

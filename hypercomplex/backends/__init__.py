"""
hypercomplex.backends
---------------------
Backend registry for hypercomplex arithmetic.

Available backends
------------------
numpy  : Pure NumPy (default). No extra dependencies.
jax     : JAX backend (Python Hyper dispatch). Requires jax.
jax-xla : XLA-compiled backend. Coefficient-level JAX arrays, jax.jit traceable.

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

BackendName = Literal["numpy", "jax", "jax-xla"]  # type: ignore[assignment]


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
    elif name == "jax-xla":
        try:
            import jax  # noqa: F401
        except ImportError:
            raise ImportError(
                "jax-xla backend requires jax. " "Install with:  pip install 'hcderiv[jax]'"
            )
        return None  # jax-xla uses JAXHyperArray, not an array module
    else:
        raise ValueError(f"Unknown backend {name!r}. Choose 'numpy', 'jax', or 'jax-xla'.")


def is_jax(name: BackendName) -> bool:
    return name in ("jax", "jax-xla")


def is_xla(name: BackendName) -> bool:
    return name == "jax-xla"


__all__ = ["get_backend", "is_jax", "is_xla", "BackendName"]

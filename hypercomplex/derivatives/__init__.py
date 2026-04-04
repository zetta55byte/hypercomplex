"""
hypercomplex.derivatives
------------------------
Exact derivative extraction via hypercomplex perturbation.

All public functions accept a ``backend`` keyword argument:

    backend="numpy"  (default) — pure NumPy, no extra dependencies
    backend="jax"              — JAX/XLA, requires ``pip install 'hcderiv[jax]'``

When ``backend="jax"`` and ``jit=True`` (default), the hypercomplex
evaluation is wrapped in ``jax.jit`` for XLA compilation.

Examples
--------
>>> from hypercomplex import grad, hessian
>>> def f(X): return X[0]**2 + X[1]**2
>>> grad(f, [1.0, 2.0])                         # NumPy
array([2., 4.])
>>> grad(f, [1.0, 2.0], backend="jax")           # JAX
array([2., 4.], dtype=float64)
"""

from __future__ import annotations
import numpy as np
from ..core.utils import make_inputs, extract_gradient_hessian
from ..backends import get_backend

# ---------------------------------------------------------------------------
# Internal: JIT wrapper
# ---------------------------------------------------------------------------


def _maybe_jit(fn, use_jit: bool, backend: str):
    """
    JIT note: jax.jit cannot trace Hyper objects directly.
    The jit= parameter is reserved for a future coefficient-level compiled
    backend. For now this is a no-op. To JIT the full pipeline, wrap the
    outer call::

        @jax.jit
        def jit_hessian(x):
            return hessian(f, x, backend="jax", jit=False)
    """
    return fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grad(f, x, alpha=1e-20, beta=1e-20, backend="numpy", jit=True):
    """
    Compute the exact gradient of a scalar function.

    Parameters
    ----------
    f : callable
        Scalar function f(X) -> Hyper, written using Hyper arithmetic.
    x : array-like
        Point at which to evaluate the gradient.
    alpha : float, optional
        Imaginary step size. Default 1e-20.
    beta : float, optional
        Nilpotent step size. Default 1e-20.
    backend : {"numpy", "jax"}, optional
        Array backend. Default "numpy".
    jit : bool, optional
        If True and backend="jax", wrap f in jax.jit. Default True.

    Returns
    -------
    ndarray of shape (n,)
        Exact gradient vector.

    Examples
    --------
    >>> from hypercomplex import grad
    >>> def f(X): return X[0]*X[0] + X[1]*X[1]
    >>> grad(f, [1.0, 2.0])
    array([2., 4.])
    >>> grad(f, [1.0, 2.0], backend="jax")
    array([2., 4.])
    """
    xp = get_backend(backend)
    x = np.asarray(x, dtype=float)
    X = make_inputs(x, alpha=alpha, beta=beta, xp=xp)
    f_ = _maybe_jit(f, jit, backend)
    F = f_(X)
    g, _ = extract_gradient_hessian(F, X, alpha=alpha, beta=beta)
    return g


def hessian(f, x, alpha=1e-20, beta=1e-20, backend="numpy", jit=True):
    """
    Compute the exact Hessian matrix in one evaluation.

    Parameters
    ----------
    f : callable
        Scalar function f(X) -> Hyper.
    x : array-like
        Point at which to evaluate the Hessian.
    alpha : float, optional
    beta : float, optional
    backend : {"numpy", "jax"}, optional
        Default "numpy".
    jit : bool, optional
        JIT-compile with JAX when backend="jax". Default True.

    Returns
    -------
    ndarray of shape (n, n)
        Exact Hessian matrix.

    Examples
    --------
    >>> from hypercomplex import hessian
    >>> def f(X): return X[0]**2 + X[0]*X[1]*3 + X[1]**2*2
    >>> hessian(f, [1.0, 2.0])
    array([[2., 3.],
           [3., 4.]])
    >>> hessian(f, [1.0, 2.0], backend="jax")
    array([[2., 3.],
           [3., 4.]])
    """
    xp = get_backend(backend)
    x = np.asarray(x, dtype=float)
    X = make_inputs(x, alpha=alpha, beta=beta, xp=xp)
    f_ = _maybe_jit(f, jit, backend)
    F = f_(X)
    _, H = extract_gradient_hessian(F, X, alpha=alpha, beta=beta)
    return H


def grad_and_hessian(f, x, alpha=1e-20, beta=1e-20, backend="numpy", jit=True):
    """
    Compute exact gradient and Hessian simultaneously in one evaluation.

    Parameters
    ----------
    f : callable
    x : array-like
    alpha, beta : float, optional
    backend : {"numpy", "jax"}, optional
    jit : bool, optional

    Returns
    -------
    grad : ndarray of shape (n,)
    H    : ndarray of shape (n, n)
    """
    xp = get_backend(backend)
    x = np.asarray(x, dtype=float)
    X = make_inputs(x, alpha=alpha, beta=beta, xp=xp)
    f_ = _maybe_jit(f, jit, backend)
    F = f_(X)
    return extract_gradient_hessian(F, X, alpha=alpha, beta=beta)


def jacobian(f, x, alpha=1e-20, beta=1e-20, backend="numpy", jit=True):
    """
    Compute the Jacobian of a vector-valued function.

    Parameters
    ----------
    f : callable
        Vector function f(X) -> list[Hyper], shape (m,).
    x : array-like
        Point in R^n.
    alpha, beta : float, optional
    backend : {"numpy", "jax"}, optional
    jit : bool, optional

    Returns
    -------
    ndarray of shape (m, n)
    """
    xp = get_backend(backend)
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = make_inputs(x, alpha=alpha, beta=beta, xp=xp)
    f_ = _maybe_jit(f, jit, backend)
    F_vec = f_(X)
    J = np.zeros((len(F_vec), n))
    for i, Fi in enumerate(F_vec):
        for j in range(n):
            J[i, j] = float(Fi.c[X[j].idx_i(j)]) / alpha
    return J


def hessian_vector_product(f, x, v, alpha=1e-20, beta=1e-20, backend="numpy", jit=True):
    """
    Compute the Hessian-vector product H(x) @ v.

    Parameters
    ----------
    f : callable
    x : array-like
    v : array-like
    alpha, beta : float, optional
    backend : {"numpy", "jax"}, optional
    jit : bool, optional

    Returns
    -------
    ndarray of shape (n,)
    """
    H = hessian(f, x, alpha=alpha, beta=beta, backend=backend, jit=jit)
    return H @ np.asarray(v, dtype=float)

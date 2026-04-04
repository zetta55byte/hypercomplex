"""
hypercomplex.core.utils
-----------------------
Utilities for constructing hypercomplex perturbations.
"""

import numpy as np
from .hyper import Hyper


def make_inputs(x, alpha=1e-20, beta=1e-20, xp=None):
    """
    Build hypercomplex input vector for one-pass Hessian extraction.

    Parameters
    ----------
    x : array-like
        Base point in R^n.
    alpha : float
        Imaginary step size (i_j channel).
    beta : float
        Nilpotent step size (eps_j channel).
    xp : module, optional
        Array module (numpy or jax.numpy). Default: numpy.

    Returns
    -------
    list of Hyper
        List of n hypercomplex numbers X_j = x_j + alpha*i_j + beta*eps_j.
    """
    if xp is None:
        xp = np
    x = np.asarray(x, dtype=float)  # always read input as numpy
    n = len(x)
    X = []
    for j in range(n):
        h = Hyper.zero(n, xp=xp)
        # Use .at[] for JAX, direct index for NumPy
        if xp is np:
            h.c[0] = x[j]
            h.c[h.idx_i(j)] = alpha
            h.c[h.idx_eps(j)] = beta
        else:
            h.c = h.c.at[0].set(x[j])
            h.c = h.c.at[h.idx_i(j)].set(alpha)
            h.c = h.c.at[h.idx_eps(j)].set(beta)
        X.append(h)
    return X


def extract_gradient_hessian(F, X, alpha=1e-20, beta=1e-20):
    """
    Extract gradient and Hessian from a hypercomplex evaluation result.

    Parameters
    ----------
    F : Hyper
        Result of evaluating a scalar function on hypercomplex inputs.
    X : list of Hyper
        Hypercomplex inputs as returned by make_inputs.
    alpha : float
        Imaginary step size used in make_inputs.
    beta : float
        Nilpotent step size used in make_inputs.

    Returns
    -------
    grad : ndarray of shape (n,)
        Exact gradient.
    H : ndarray of shape (n, n)
        Exact Hessian matrix.
    """
    n = len(X)
    ab = alpha * beta

    # Always return plain NumPy arrays (backend-agnostic output)
    grad = np.array([float(F.c[X[j].idx_i(j)]) / alpha for j in range(n)])

    H = np.zeros((n, n))
    for j in range(n):
        H[j, j] = float(F.c[X[j].idx_diag_mix(j)]) / ab
    for j in range(n):
        for k in range(j + 1, n):
            val = float(F.c[X[j].idx_mix(j, k)]) / ab
            H[j, k] = val
            H[k, j] = val

    return grad, H

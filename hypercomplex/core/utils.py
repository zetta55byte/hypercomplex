"""
hypercomplex.core.utils
-----------------------
Utilities for constructing hypercomplex perturbations.
"""

import numpy as np
from .hyper import Hyper


def make_inputs(x, alpha=1e-20, beta=1e-20):
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

    Returns
    -------
    list of Hyper
        List of n hypercomplex numbers X_j = x_j + alpha*i_j + beta*eps_j.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = []
    for j in range(n):
        h = Hyper.zero(n)
        h.c[0] = x[j]
        h.c[h.idx_i(j)] = alpha
        h.c[h.idx_eps(j)] = beta
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

    # gradient: coeff(i_j) / alpha
    grad = np.array([F.c[X[j].idx_i(j)] / alpha for j in range(n)])

    # diagonal Hessian: coeff(i_j eps_j) / (alpha * beta)
    H = np.zeros((n, n))
    ab = alpha * beta
    for j in range(n):
        H[j, j] = F.c[X[j].idx_diag_mix(j)] / ab

    # off-diagonal Hessian: coeff(i_j eps_k) / (alpha * beta)
    for j in range(n):
        for k in range(j + 1, n):
            val = F.c[X[j].idx_mix(j, k)] / ab
            H[j, k] = val
            H[k, j] = val

    return grad, H

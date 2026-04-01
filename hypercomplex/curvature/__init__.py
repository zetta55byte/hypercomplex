"""
hypercomplex.curvature
----------------------
Ridge and principal curvature extraction for scalar fields.
"""

import numpy as np
from ..derivatives import grad_and_hessian


def ridge_curvature(f, x, alpha=1e-20, beta=1e-20):
    """
    Compute ridge curvature of a scalar field at a point.

    Ridge curvature is defined as the largest eigenvalue of the Hessian,
    which characterizes the curvature transverse to the gradient flow.

    Parameters
    ----------
    f : callable
        Scalar function f(X) -> Hyper.
    x : array-like
        Point in R^n.
    alpha : float, optional
    beta : float, optional

    Returns
    -------
    float
        Ridge curvature kappa = lambda_max(H(x)).

    Examples
    --------
    >>> from hypercomplex import ridge_curvature
    >>> def U(X): return -(X[0]*X[0] + X[1]*X[1])
    >>> ridge_curvature(U, [0.0, 0.0])
    -2.0
    """
    _, H = grad_and_hessian(f, x, alpha=alpha, beta=beta)
    return float(np.linalg.eigvalsh(H)[-1])


def principal_curvatures(f, x, alpha=1e-20, beta=1e-20):
    """
    Compute principal curvatures and directions of a scalar field.

    Parameters
    ----------
    f : callable
        Scalar function f(X) -> Hyper.
    x : array-like
        Point in R^n.
    alpha : float, optional
    beta : float, optional

    Returns
    -------
    eigenvalues : ndarray of shape (n,)
        Principal curvatures (eigenvalues of Hessian), ascending order.
    eigenvectors : ndarray of shape (n, n)
        Principal directions (columns are eigenvectors).

    Notes
    -----
    The largest eigenvalue corresponds to the ridge direction.
    """
    _, H = grad_and_hessian(f, x, alpha=alpha, beta=beta)
    return np.linalg.eigh(H)


def curvature_map(f, xs, ys, alpha=1e-20, beta=1e-20):
    """
    Compute ridge curvature over a 2D grid.

    Parameters
    ----------
    f : callable
        Scalar function f(X) -> Hyper.
    xs : array-like
        x-coordinates of the grid.
    ys : array-like
        y-coordinates of the grid.
    alpha : float, optional
    beta : float, optional

    Returns
    -------
    ndarray of shape (len(ys), len(xs))
        Ridge curvature at each grid point.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    kappa = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            kappa[i, j] = ridge_curvature(f, [x, y], alpha=alpha, beta=beta)
    return kappa


def shape_operator(f, x, alpha=1e-20, beta=1e-20):
    """
    Compute the shape operator (Weingarten map) of a level set at x.

    Parameters
    ----------
    f : callable
        Scalar function f(X) -> Hyper.
    x : array-like
        Point in R^n.
    alpha : float, optional
    beta : float, optional

    Returns
    -------
    ndarray of shape (n, n)
        Shape operator matrix S = H / |grad f|.

    Notes
    -----
    The shape operator encodes the curvature of level sets of f.
    Its eigenvalues are the principal curvatures of the level set.
    """
    g, H = grad_and_hessian(f, x, alpha=alpha, beta=beta)
    gnorm = np.linalg.norm(g)
    if gnorm < 1e-300:
        raise ValueError("Gradient is zero at x; shape operator is undefined.")
    return H / gnorm

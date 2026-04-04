"""
hypercomplex.systems
--------------------
Tools for attractor-ridge analysis of dynamical systems.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from ..curvature import (
    curvature_map as curvature_map,
    ridge_curvature as ridge_curvature,
)


def make_ridge_potential(f_ode):
    """
    Construct the ridge potential U(x) = -||f(x)||^2 from an ODE vector field.

    Parameters
    ----------
    f_ode : callable
        ODE vector field f(x) -> array of shape (n,). Real-valued.

    Returns
    -------
    callable
        Function U(X) -> Hyper, compatible with hypercomplex evaluation.

    Notes
    -----
    The returned function uses Hyper arithmetic internally. The ODE function
    f_ode must be re-implemented in Hyper arithmetic separately for
    hypercomplex evaluation.
    """

    def U(x):
        fv = np.asarray(f_ode(x))
        return -float(np.dot(fv, fv))

    return U


def find_fixed_points(f_ode, n, grid_bounds, grid_points=10):
    """
    Find all fixed points of an ODE system by grid search + Newton refinement.

    Parameters
    ----------
    f_ode : callable
        Real-valued ODE vector field f(x) -> array of shape (n,).
    n : int
        State space dimension.
    grid_bounds : tuple of (low, high)
        Search bounds for each dimension.
    grid_points : int
        Number of initial conditions per dimension.

    Returns
    -------
    list of ndarray
        Unique fixed points, sorted by first coordinate.
    """
    low, high = grid_bounds
    fps = []
    from itertools import product

    grid = np.linspace(low, high, grid_points)
    for x0 in product(*[grid] * n):
        x0 = np.array(x0)
        try:
            sol = fsolve(f_ode, x0, full_output=True)
            xf, _, ier, _ = sol
            if ier == 1 and np.linalg.norm(f_ode(xf)) < 1e-10:
                if np.all(xf > low) and np.all(xf < high):
                    if not any(np.linalg.norm(xf - fp) < 1e-4 for fp in fps):
                        fps.append(xf)
        except Exception:
            pass
    fps.sort(key=lambda x: x[0])
    return fps


def classify_fixed_points(f_ode, fps, h=1e-6):
    """
    Classify fixed points as stable, unstable, or saddle.

    Parameters
    ----------
    f_ode : callable
        Real ODE vector field.
    fps : list of ndarray
        Fixed points.
    h : float
        Step size for numerical Jacobian.

    Returns
    -------
    list of dict
        Each dict has keys 'point', 'eigenvalues', 'type'.
    """
    results = []
    for fp in fps:
        n = len(fp)
        J = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            J[:, i] = (f_ode(fp + h * ei) - f_ode(fp - h * ei)) / (2 * h)
        evals = np.linalg.eigvals(J).real
        if np.all(evals < 0):
            ftype = "stable"
        elif np.all(evals > 0):
            ftype = "unstable"
        else:
            ftype = "saddle"
        results.append({"point": fp, "eigenvalues": evals, "type": ftype})
    return results


def basin_map(f_ode, xs, ys, t_end=50, tol=1e-8):
    """
    Classify each grid point by its attractor basin via forward integration.

    Parameters
    ----------
    f_ode : callable
        Real ODE vector field f(x) -> array.
    xs : array-like
        x-coordinates.
    ys : array-like
        y-coordinates.
    t_end : float
        Integration time.
    tol : float
        Solver tolerance.

    Returns
    -------
    ndarray of shape (len(ys), len(xs))
        Basin index: 1 if x[0] > x[1] at final time, 0 otherwise.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    result = np.zeros((len(ys), len(xs)))

    def rhs(t, y):
        return f_ode(y)

    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            sol = solve_ivp(
                rhs, [0, t_end], [x0, y0], method="RK45", rtol=tol, atol=tol
            )
            yf = sol.y[:, -1]
            result[i, j] = 1 if yf[0] > yf[1] else 0
    return result


def separatrix(f_ode, saddle, t_back=20):
    """
    Extract the separatrix by backward integration from the saddle.

    Parameters
    ----------
    f_ode : callable
        Real ODE vector field.
    saddle : array-like
        Saddle point coordinates.
    t_back : float
        Backward integration time.

    Returns
    -------
    tuple of (ndarray, ndarray)
        Two branches of the separatrix.
    """
    saddle = np.asarray(saddle)
    n = len(saddle)
    h = 1e-6
    J = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        J[:, i] = (f_ode(saddle + h * ei) - f_ode(saddle - h * ei)) / (2 * h)
    evals, evecs = np.linalg.eig(J)
    stable_idx = np.argmin(evals.real)
    v = evecs[:, stable_idx].real
    v = v / np.linalg.norm(v)

    branches = []

    def rhs_back(t, y):
        return -f_ode(y)

    for sign in [1, -1]:
        x0 = saddle + sign * 0.01 * v
        sol = solve_ivp(
            rhs_back,
            [0, t_back],
            x0,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            max_step=0.1,
        )
        pts = sol.y.T
        mask = np.all((pts > 0) & (pts < 10), axis=1)
        branches.append(pts[mask])
    return branches[0], branches[1]

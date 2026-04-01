"""
hypercomplex.examples.lambda_switch
------------------------------------
Canonical lambda-switch example demonstrating one-pass ridge curvature extraction.

Usage
-----
    python -m hypercomplex.examples.lambda_switch
"""

import numpy as np
from ..core.hyper import Hyper
from ..derivatives import grad_and_hessian
from ..curvature import ridge_curvature, curvature_map
from ..systems import find_fixed_points, classify_fixed_points, basin_map, separatrix

# ── Parameters ────────────────────────────────────────────────────────────────

PARAMS = dict(
    alpha_C=5.0, alpha_R=5.0,
    K_C=1.0,     K_R=1.0,
    n_C=3,       n_R=3,
    beta_C=1.0,  beta_R=1.0,
)


def f_ode(x, p=PARAMS):
    """Real-valued lambda-switch vector field."""
    C, R = x
    dC = p['alpha_C'] / (1 + (R / p['K_R'])**p['n_R']) - p['beta_C'] * C
    dR = p['alpha_R'] / (1 + (C / p['K_C'])**p['n_C']) - p['beta_R'] * R
    return np.array([dC, dR])


def U_hyper(X, p=PARAMS):
    """
    Ridge potential U(x) = -||f(x)||^2 in Hyper arithmetic.

    Parameters
    ----------
    X : list of Hyper
        Hypercomplex inputs [C, R].

    Returns
    -------
    Hyper
        Hypercomplex evaluation of U.
    """
    C, R = X
    R_scaled = R * (1.0 / p['K_R'])
    C_scaled = C * (1.0 / p['K_C'])
    denom_C = Hyper.real(2, 1.0) + R_scaled ** p['n_R']
    denom_R = Hyper.real(2, 1.0) + C_scaled ** p['n_C']
    fC = p['alpha_C'] * (1.0 / denom_C) - C * p['beta_C']
    fR = p['alpha_R'] * (1.0 / denom_R) - R * p['beta_R']
    return (fC * fC + fR * fR) * (-1.0)


def run(verbose=True):
    """Run the full lambda-switch analysis."""

    # Fixed points
    fps = find_fixed_points(f_ode, n=2, grid_bounds=(0.01, 6.0))
    classified = classify_fixed_points(f_ode, fps)
    saddle = next(r['point'] for r in classified if r['type'] == 'saddle')

    if verbose:
        print("Lambda-switch fixed points:")
        for r in classified:
            print(f"  {np.round(r['point'], 4)}  [{r['type']}]  "
                  f"evals={np.round(r['eigenvalues'], 3)}")
        print(f"\nSaddle: {saddle}")

    # Ridge curvature at saddle
    kappa = ridge_curvature(U_hyper, saddle)
    g, H = grad_and_hessian(U_hyper, saddle)

    if verbose:
        print(f"\nHessian of U at saddle:\n{H}")
        print(f"Ridge curvature kappa = {kappa:.8f}")

    return {
        'fixed_points': classified,
        'saddle': saddle,
        'hessian': H,
        'gradient': g,
        'kappa': kappa,
    }


if __name__ == '__main__':
    run()

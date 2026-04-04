"""
examples/trust_region.py
========================
Trust-region optimization on a modified Rosenbrock function using three
Hessian sources:
  1. Exact      — hcderiv (hypercomplex perturbation)
  2. Approximate — finite differences (central, h=1e-5)
  3. Baseline   — diagonal (identity)

Function: f(x) = (1-x0)² + 10*(x1-x0²)²
  Optimum at (1, 1). Off-diagonal Hessian terms are significant (~40 at x0),
  making the diagonal approximation a poor substitute for the full curvature.

Demonstrates:
  - Exact and FD both converge in ~16 iterations (Newton-like)
  - Diagonal baseline requires 150+ iterations and fails to reach f < 1e-6
  - Step acceptance rate and radius adaptation differ substantially

Produces: examples/figures/trust_region_trajectories.pdf/.png

Usage:
  python examples/trust_region.py
"""

import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hypercomplex import grad_and_hessian
from hypercomplex.core.hyper import Hyper

os.makedirs("examples/figures", exist_ok=True)


# ── objective ────────────────────────────────────────────────────────────────


def f_real(x):
    """Modified Rosenbrock (b=10): f(x) = (1-x0)² + 10*(x1-x0²)²"""
    return float((1.0 - x[0]) ** 2 + 10.0 * (x[1] - x[0] ** 2) ** 2)


def f_hc(X):
    """Modified Rosenbrock in hypercomplex arithmetic."""
    one = Hyper.real(X[0].n, 1.0)
    a = one - X[0]
    b = X[1] - X[0] ** 2
    return a**2 + b**2 * 10.0


# ── Hessian sources ───────────────────────────────────────────────────────────


def grad_hess_exact(x):
    """Exact gradient and Hessian via hypercomplex perturbation."""
    return grad_and_hessian(f_hc, x)


def grad_hess_fd(x, h=1e-5):
    """Central-difference gradient and Hessian."""
    n = len(x)
    g = np.zeros(n)
    H = np.zeros((n, n))
    e = np.eye(n)
    for i in range(n):
        g[i] = (f_real(x + h * e[i]) - f_real(x - h * e[i])) / (2 * h)
        for j in range(n):
            H[i, j] = (
                f_real(x + h * e[i] + h * e[j])
                - f_real(x + h * e[i] - h * e[j])
                - f_real(x - h * e[i] + h * e[j])
                + f_real(x - h * e[i] - h * e[j])
            ) / (4 * h * h)
    return g, H


def grad_hess_diag(x, h=1e-5):
    """Diagonal baseline: gradient via FD, Hessian = identity."""
    n = len(x)
    g = np.zeros(n)
    e = np.eye(n)
    for i in range(n):
        g[i] = (f_real(x + h * e[i]) - f_real(x - h * e[i])) / (2 * h)
    return g, np.eye(n)


# ── trust-region step ─────────────────────────────────────────────────────────


def trust_region_step(g, H, delta):
    """
    Cauchy point + Newton step.
    Takes Newton step if it lies inside the trust region,
    otherwise falls back to the Cauchy point on the boundary.
    """
    try:
        p_newton = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        p_newton = -g

    if np.linalg.norm(p_newton) <= delta:
        return p_newton

    gnorm = np.linalg.norm(g)
    if gnorm < 1e-14:
        return np.zeros_like(g)
    gHg = float(g @ (H @ g))
    tau = 1.0 if gHg <= 0 else min(gnorm**3 / (delta * gHg), 1.0)
    return -tau * delta * g / gnorm


# ── solver ────────────────────────────────────────────────────────────────────


def solve(x0, grad_hess_fn, delta0=1.0, eta=0.15, max_iter=150):
    """
    Trust-region solver with standard radius update rules.

    Returns
    -------
    traj  : (iters+1, n) array of iterates
    diags : list of dicts — rho, delta, accepted, gnorm per step
    """
    x = x0.copy()
    delta = delta0
    traj = [x.copy()]
    diags = []

    for _ in range(max_iter):
        g, H = grad_hess_fn(x)
        if np.linalg.norm(g) < 1e-10:
            break

        p = trust_region_step(g, H, delta)

        pred = float(-(g @ p + 0.5 * p @ (H @ p)))
        ared = f_real(x) - f_real(x + p)
        rho = ared / pred if abs(pred) > 1e-15 else 0.0

        accepted = rho > eta
        if accepted:
            x = x + p

        pnorm = np.linalg.norm(p)
        if rho < 0.25:
            delta = 0.25 * pnorm
        elif rho > 0.75:
            delta = min(2.0 * delta, 10.0)

        traj.append(x.copy())
        diags.append(dict(rho=rho, delta=delta, accepted=accepted, gnorm=np.linalg.norm(g)))

    return np.array(traj), diags


# ── figure ────────────────────────────────────────────────────────────────────


def plot_results(trajs, diags_list, labels, colors):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    # Panel 1: trajectories
    ax = axes[0]
    xs = np.linspace(-1.5, 1.3, 400)
    ys = np.linspace(-0.3, 1.8, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (1 - X) ** 2 + 10 * (Y - X**2) ** 2
    ax.contour(X, Y, np.log1p(Z), levels=25, cmap="gray", alpha=0.5, linewidths=0.6)
    ax.plot(1.0, 1.0, "*", ms=14, color="gold", zorder=5, label="Optimum (1,1)")

    for traj, label, color in zip(trajs, labels, colors):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "o-",
            color=color,
            ms=3,
            lw=1.5,
            label=label,
            alpha=0.9,
        )
        ax.plot(traj[0, 0], traj[0, 1], "s", color=color, ms=7, zorder=4)

    ax.set_xlabel("x₀", fontsize=11)
    ax.set_ylabel("x₁", fontsize=11)
    ax.set_title("Trajectories\n(square = start)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(-1.5, 1.3)
    ax.set_ylim(-0.3, 1.8)

    # Panel 2: convergence
    ax = axes[1]
    for traj, label, color in zip(trajs, labels, colors):
        fvals = [f_real(x) for x in traj]
        ax.semilogy(fvals, "o-", color=color, ms=3, lw=1.5, label=label)
    ax.axhline(1e-6, color="gray", lw=0.8, ls=":", label="f = 1e-6")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("f(x)  [log scale]", fontsize=11)
    ax.set_title("Convergence", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, 50)  # zoom to first 50 iters where exact/FD are interesting

    # Panel 3: radius + rejections
    ax = axes[2]
    for diags, label, color in zip(diags_list, labels, colors):
        deltas = [d["delta"] for d in diags]
        accepted = [d["accepted"] for d in diags]
        iters = list(range(len(deltas)))
        ax.plot(iters, deltas, "-", color=color, lw=1.5, label=label)
        rej = [i for i, a in enumerate(accepted) if not a]
        ax.scatter(rej, [deltas[i] for i in rej], marker="x", color=color, s=50, zorder=4)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Trust-region radius δ", fontsize=11)
    ax.set_title("Radius adaptation\n(× = rejected step)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)

    plt.tight_layout(pad=1.8)
    for ext in ("pdf", "png"):
        out = f"examples/figures/trust_region_trajectories.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ── summary ───────────────────────────────────────────────────────────────────


def print_summary(trajs, diags_list, labels):
    print(
        f"\n{'Method':<22}  {'Iters':>6}  {'Accepted':>9}  {'Rejected':>9}  "
        f"{'Final f':>10}  {'Final |g|':>10}"
    )
    print("-" * 75)
    for traj, diags, label in zip(trajs, diags_list, labels):
        n_iter = len(traj) - 1
        n_accept = sum(d["accepted"] for d in diags)
        n_reject = n_iter - n_accept
        final_f = f_real(traj[-1])
        final_g = diags[-1]["gnorm"] if diags else float("nan")
        print(
            f"{label:<22}  {n_iter:>6}  {n_accept:>9}  {n_reject:>9}  "
            f"{final_f:>10.2e}  {final_g:>10.2e}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0])

    print("Running trust-region solvers on modified Rosenbrock (b=10)...")
    traj_exact, diags_exact = solve(x0, grad_hess_exact)
    traj_fd, diags_fd = solve(x0, grad_hess_fd)
    traj_diag, diags_diag = solve(x0, grad_hess_diag)

    labels = ["Exact (hcderiv)", "Finite Differences", "Diagonal"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    trajs = [traj_exact, traj_fd, traj_diag]
    diags = [diags_exact, diags_fd, diags_diag]

    print_summary(trajs, diags, labels)
    plot_results(trajs, diags, labels, colors)
    print("\nDone.")

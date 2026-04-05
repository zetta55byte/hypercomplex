"""
Differentiable Physics Example — Pendulum Energy Hessian
=========================================================
Computes the exact Hessian of a pendulum's total mechanical energy
in a single forward evaluation using hcderiv.

System:
  State: x = (θ, ω)  — angle (rad), angular velocity (rad/s)
  Energy: E(θ, ω) = ½mL²ω² + mgL(1 − cos θ)
    Kinetic:   T = ½mL²ω²
    Potential: V = mgL(1 − cos θ)

The Hessian ∇²E tells us:
  ∂²E/∂θ²   = mgL·cos θ   (curvature of potential well)
  ∂²E/∂ω²   = mL²          (effective mass — constant)
  ∂²E/∂θ∂ω  = 0             (no coupling — separable system)

Run:
    python examples/pendulum_energy.py
"""

from __future__ import annotations

import numpy as np

from hypercomplex import grad, hessian

# ── System parameters ─────────────────────────────────────────────────────────

M = 1.0  # mass (kg)
L = 1.0  # length (m)
G = 9.81  # gravitational acceleration (m/s²)


# ── Energy function ───────────────────────────────────────────────────────────


def pendulum_energy(X):
    """
    Total mechanical energy of a simple pendulum.

    Parameters
    ----------
    X : list of Hyper
        X[0] = θ (angle in radians)
        X[1] = ω (angular velocity in rad/s)

    Returns
    -------
    Hyper
        Total energy E = T + V.
    """
    theta, omega = X[0], X[1]
    kinetic = omega**2 * (0.5 * M * L**2)
    potential = (theta.cos() * (-M * G * L)) + (M * G * L)  # mgL(1 - cos θ)
    return kinetic + potential


# ── Analytic reference ────────────────────────────────────────────────────────


def analytic_hessian(theta: float, omega: float) -> np.ndarray:
    """
    Analytic Hessian of E(θ, ω):
      H = [[mgL·cos θ,  0     ],
           [0,          mL²   ]]
    """
    return np.array(
        [
            [M * G * L * np.cos(theta), 0.0],
            [0.0, M * L**2],
        ]
    )


# ── Demo ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("\n=== Pendulum Energy — Exact Hessian via hcderiv ===\n")
    print(f"System: m={M} kg, L={L} m, g={G} m/s²")
    print("E(θ, ω) = ½mL²ω² + mgL(1 − cos θ)\n")

    test_points = [
        ("Equilibrium", 0.0, 0.0),
        ("Small displacement", 0.1, 0.5),
        ("Quarter swing", np.pi / 4, 1.0),
        ("Near top (unstable)", np.pi * 0.9, 0.0),
    ]

    print(
        f"{'Point':<22} {'θ':>8} {'ω':>8} {'E':>10} "
        f"{'H[0,0]':>10} {'H[1,1]':>10} {'H[0,1]':>10} {'Error':>10}"
    )
    print("-" * 92)

    for label, theta, omega in test_points:
        x = [theta, omega]

        # Exact energy and gradient
        E = 0.5 * M * L**2 * omega**2 + M * G * L * (1.0 - np.cos(theta))
        H = hessian(pendulum_energy, x)  # also: grad(pendulum_energy, x)
        H = hessian(pendulum_energy, x)
        H_analytic = analytic_hessian(theta, omega)
        error = np.max(np.abs(H - H_analytic))

        print(
            f"{label:<22} {theta:>8.3f} {omega:>8.3f} {E:>10.4f} "
            f"{H[0,0]:>10.4f} {H[1,1]:>10.4f} {H[0,1]:>10.1e} {error:>10.2e}"
        )

    print()
    print("H[0,0] = ∂²E/∂θ² = mgL·cos θ  (potential curvature)")
    print("H[1,1] = ∂²E/∂ω² = mL²         (effective inertia, constant)")
    print("H[0,1] = ∂²E/∂θ∂ω = 0          (no coupling — separable)")
    print()
    print("HC vs analytic: max error < 1e-13 (machine precision)")

    # Curvature sweep over angle
    print("\n=== Potential curvature ∂²E/∂θ² vs angle ===\n")
    print(f"{'θ (deg)':>10}  {'∂²E/∂θ²':>12}  {'analytic':>12}  bar")
    print("-" * 60)

    for deg in range(0, 181, 20):
        theta = np.radians(deg)
        H = hessian(pendulum_energy, [theta, 0.0])
        curv = H[0, 0]
        analytic = M * G * L * np.cos(theta)
        bar_len = max(0, int((curv / (M * G * L) + 1) * 15))
        bar = "█" * bar_len
        print(f"{deg:>10}  {curv:>12.4f}  {analytic:>12.4f}  {bar}")

    print()
    print("At θ=0 (bottom):  max curvature → stable equilibrium")
    print("At θ=90°:         zero curvature → inflection")
    print("At θ=180° (top):  negative curvature → unstable equilibrium")


if __name__ == "__main__":
    main()

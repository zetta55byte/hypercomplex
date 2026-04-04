"""
benchmarks/run_scaling.py
=========================
HC (vectorized v0.2.0) vs FD (central differences) vs JAX (compiled)

Functions tested:
  - quadratic:  f(x) = sum_i x_i^2 + sum_{i<j} x_i*x_j
  - rosenbrock: f(x) = sum_i [100*(x_{i+1}-x_i^2)^2 + (x_i-1)^2]
  - nonlinear:  f(x) = sum_i sin(x_i)*exp(-x_i^2/2)  [for FD error sweep]

Outputs:
  benchmarks/results/scaling_results.csv
  benchmarks/results/fd_error_sweep.csv
"""

import numpy as np
import time
import csv
import os

os.makedirs("benchmarks/results", exist_ok=True)

# ── optional JAX ─────────────────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not installed — skipping JAX column")

# ── HC ───────────────────────────────────────────────────────────────────────
from hypercomplex import hessian as hc_hessian  # noqa: E402
from hypercomplex.core.hyper import Hyper  # noqa: E402


def f_quad_hc(X):
    n = len(X)
    acc = Hyper.real(X[0].n, 0.0)
    for i in range(n):
        acc = acc + X[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            acc = acc + X[i] * X[j]
    return acc


def f_rosen_hc(X):
    n = len(X)
    acc = Hyper.real(X[0].n, 0.0)
    for i in range(n - 1):
        a = X[i + 1] - X[i] ** 2
        b = X[i] - Hyper.real(X[i].n, 1.0)
        acc = acc + a**2 * 100.0 + b**2
    return acc


# ── FD ───────────────────────────────────────────────────────────────────────
def fd_hessian(f, x, h=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy()
            xpp[i] += h
            xpp[j] += h
            xpm = x.copy()
            xpm[i] += h
            xpm[j] -= h
            xmp = x.copy()
            xmp[i] -= h
            xmp[j] += h
            xmm = x.copy()
            xmm[i] -= h
            xmm[j] -= h
            v = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
            H[i, j] = H[j, i] = v
    return H


def f_quad_real(x):
    return float(
        np.sum(x**2) + sum(x[i] * x[j] for i in range(len(x)) for j in range(i + 1, len(x)))
    )


def f_rosen_real(x):
    return float(
        sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))
    )


# ── JAX ───────────────────────────────────────────────────────────────────────
if HAS_JAX:

    def f_quad_jax(x):
        return jnp.sum(x**2) + jnp.sum(jnp.tril(jnp.outer(x, x), -1))

    def f_rosen_jax(x):
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

    jax_hess_quad = jax.jit(jax.hessian(f_quad_jax))
    jax_hess_rosen = jax.jit(jax.hessian(f_rosen_jax))

# ── benchmark loop ────────────────────────────────────────────────────────────
DIMS = [2, 4, 8, 16, 32, 64]
REPS = {2: 30, 4: 20, 8: 15, 16: 10, 32: 5, 64: 3}

scaling_rows = []

for fname, f_hc, f_real, jax_fn in [
    ("quadratic", f_quad_hc, f_quad_real, jax_hess_quad if HAS_JAX else None),
    ("rosenbrock", f_rosen_hc, f_rosen_real, jax_hess_rosen if HAS_JAX else None),
]:
    print(f"\n{'=' * 60}")
    print(f"Function: {fname}")
    print(f"{'n':>4}  {'HC(ms)':>8}  {'FD(ms)':>8}  {'JAX(ms)':>8}  {'FD_err%':>9}")
    print("-" * 48)

    x_warm = np.ones(8)
    if HAS_JAX:
        _ = jax_fn(jnp.array(x_warm))  # warm-up

    for n in DIMS:
        reps = REPS[n]
        x = np.ones(n) * 0.5

        # HC
        t0 = time.perf_counter()
        for _ in range(reps):
            H_hc = hc_hessian(f_hc, x)
        t_hc = (time.perf_counter() - t0) / reps * 1000

        # FD
        if n <= 32:
            t0 = time.perf_counter()
            for _ in range(reps):
                H_fd = fd_hessian(f_real, x)
            t_fd = (time.perf_counter() - t0) / reps * 1000
        else:
            H_fd = fd_hessian(f_real, x)
            t_fd = None

        fd_err = np.max(np.abs(H_fd - H_hc)) / (np.max(np.abs(H_hc)) + 1e-30) * 100

        # JAX
        if HAS_JAX:
            xj = jnp.array(x)
            t0 = time.perf_counter()
            for _ in range(reps):
                H_jax = jax_fn(xj)
            t_jax = (time.perf_counter() - t0) / reps * 1000
        else:
            t_jax = None

        fd_str = f"{t_fd:.2f}" if t_fd is not None else "---"
        jax_str = f"{t_jax:.2f}" if t_jax is not None else "---"
        print(f"{n:>4}  {t_hc:>8.2f}  {fd_str:>8}  {jax_str:>8}  {fd_err:>8.4f}%")

        scaling_rows.append(
            dict(
                function=fname,
                n=n,
                t_hc_ms=round(t_hc, 4),
                t_fd_ms=round(t_fd, 4) if t_fd is not None else "",
                t_jax_ms=round(t_jax, 4) if t_jax is not None else "",
                fd_err_pct=round(fd_err, 6),
            )
        )

with open("benchmarks/results/scaling_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=scaling_rows[0].keys())
    w.writeheader()
    w.writerows(scaling_rows)
print("\nSaved: benchmarks/results/scaling_results.csv")

# ── FD step-size error sweep ──────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("FD step-size error sweep (nonlinear: sin·exp)")
print(f"{'h':>10}  {'max_abs_err':>14}  {'rel_err_%':>12}")
print("-" * 42)


def f_nl_hc(X):
    n = len(X)
    acc = Hyper.real(n, 0.0)
    for i in range(n):
        acc = acc + X[i].sin() * (X[i] ** 2 * (-0.5)).exp()
    return acc


def f_nl_real(x):
    return float(np.sum(np.sin(x) * np.exp(-(x**2) / 2)))


x4 = np.array([0.5, 1.0, -0.3, 0.8])
H_hc_nl = hc_hessian(f_nl_hc, x4)

fd_error_rows = []
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    H_fd_nl = fd_hessian(f_nl_real, x4, h)
    err = float(np.max(np.abs(H_fd_nl - H_hc_nl)))
    rel = err / (float(np.max(np.abs(H_hc_nl))) + 1e-30) * 100
    print(f"{h:>10.0e}  {err:>14.2e}  {rel:>11.4f}%")
    fd_error_rows.append(dict(h=h, max_abs_err=err, rel_err_pct=rel))

with open("benchmarks/results/fd_error_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fd_error_rows[0].keys())
    w.writeheader()
    w.writerows(fd_error_rows)
print("Saved: benchmarks/results/fd_error_sweep.csv")

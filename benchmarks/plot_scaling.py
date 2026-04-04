"""
benchmarks/plot_scaling.py
==========================
Reads benchmarks/results/scaling_results.csv and
benchmarks/results/fd_error_sweep.csv, produces:

  benchmarks/results/figure_scaling.pdf   (paper quality)
  benchmarks/results/figure_scaling.png   (README embed)

Run after run_scaling.py.
"""

import csv
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs("benchmarks/results", exist_ok=True)

# ── load scaling results ──────────────────────────────────────────────────────
scaling = {}
with open("benchmarks/results/scaling_results.csv") as f:
    for row in csv.DictReader(f):
        fn = row["function"]
        if fn not in scaling:
            scaling[fn] = dict(n=[], hc=[], fd=[], jax=[])
        scaling[fn]["n"].append(int(row["n"]))
        scaling[fn]["hc"].append(float(row["t_hc_ms"]))
        scaling[fn]["fd"].append(float(row["t_fd_ms"]) if row["t_fd_ms"] else None)
        scaling[fn]["jax"].append(float(row["t_jax_ms"]) if row["t_jax_ms"] else None)

# ── load FD error sweep ───────────────────────────────────────────────────────
fd_hs, fd_errs = [], []
with open("benchmarks/results/fd_error_sweep.csv") as f:
    for row in csv.DictReader(f):
        fd_hs.append(float(row["h"]))
        fd_errs.append(float(row["rel_err_pct"]))

# ── style ─────────────────────────────────────────────────────────────────────
BLUE = "#1f77b4"
ORANGE = "#d62728"
GREEN = "#2ca02c"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    }
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.patch.set_facecolor("white")


def _plot_runtime(ax, fn_name, title):
    d = scaling[fn_name]
    ns = d["n"]
    hcs = d["hc"]

    fd_ns = [n for n, v in zip(ns, d["fd"]) if v is not None]
    fd_vs = [v for v in d["fd"] if v is not None]
    jax_ns = [n for n, v in zip(ns, d["jax"]) if v is not None]
    jax_vs = [v for v in d["jax"] if v is not None]

    ax.plot(ns, hcs, "o-", color=BLUE, lw=2, ms=7, label="HC (vectorized)")
    ax.plot(fd_ns, fd_vs, "s--", color=ORANGE, lw=2, ms=7, label="FD (central diff)")
    ax.plot(jax_ns, jax_vs, "^:", color=GREEN, lw=2, ms=7, label="JAX (compiled)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Dimension  n", fontsize=12)
    ax.set_ylabel("Runtime  (ms)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which="both")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))


_plot_runtime(axes[0], "quadratic", "Quadratic — runtime vs n")
_plot_runtime(axes[1], "rosenbrock", "Rosenbrock — runtime vs n")

# ── Panel 3: FD step-size error ───────────────────────────────────────────────
ax = axes[2]
ax.semilogx(fd_hs, fd_errs, "o-", color=ORANGE, lw=2, ms=8, label="FD relative error  (%)")
ax.axhline(0.0, color=BLUE, lw=1.8, ls="--", label="HC  (machine precision, 4×10⁻¹⁶)")
ax.axvspan(1e-5, 8e-9, alpha=0.10, color="red", label="Cancellation regime")
ax.axvspan(2e-1, 8e-3, alpha=0.10, color="gold", label="Truncation regime")

ax.set_xlabel("FD step size  h", fontsize=12)
ax.set_ylabel("Relative Hessian error  (%)", fontsize=12)
ax.set_title(
    "FD step-size sensitivity\n" r"$f(x) = \sum_i \sin(x_i)\,e^{-x_i^2/2}$",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, which="both")
ax.set_ylim(-2, 40)

plt.tight_layout(pad=1.8)

for ext in ("pdf", "png"):
    out = f"benchmarks/results/figure_scaling.{ext}"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close()

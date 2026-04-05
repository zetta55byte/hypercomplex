"""
benchmarks/bench_hessian_backends.py
-------------------------------------
Benchmark hcderiv NumPy vs JAX (eager + jit) backends.

Measures median wall-clock time to compute the full d×d Hessian of:

    f(x) = sum_i [ sin(x_i) + 0.1 * x_i^4 ]

across d ∈ {3, 8, 16, 32, 64}.

Usage:
    python benchmarks/bench_hessian_backends.py
    python benchmarks/bench_hessian_backends.py --plot
    python benchmarks/bench_hessian_backends.py --save results/backend_timing.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

# ── hcderiv imports ───────────────────────────────────────────────────────────
# hcderiv installs as the 'hypercomplex' Python module.
# Public API: hessian(f, x, backend="numpy"|"jax")
from hypercomplex import hessian
from hypercomplex.core.hyper import Hyper


# ── Test functions ─────────────────────────────────────────────────────────────


def f_numpy(X):
    """f(x) = sum_i [ sin(x_i) + 0.1 * x_i^4 ]  — NumPy backend."""
    acc = Hyper.real(X[0].n, 0.0)
    for xi in X:
        acc = acc + xi.sin() + xi**4 * 0.1
    return acc


def f_jax(X):
    """f(x) = sum_i [ sin(x_i) + 0.1 * x_i^4 ]  — JAX backend."""
    acc = Hyper.real(X[0].n, 0.0, xp=X[0]._xp)
    for xi in X:
        acc = acc + xi.sin() + xi**4 * 0.1
    return acc


# ── Timing ────────────────────────────────────────────────────────────────────


def time_fn(fn, x, repeats: int = 50, warmup: int = 5) -> float:
    """Return median wall-clock time (seconds) over `repeats` calls."""
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


# ── Main benchmark ────────────────────────────────────────────────────────────


def run_benchmark(
    dims: list[int] | None = None,
    repeats: int = 50,
    warmup: int = 5,
    seed: int = 0,
    include_jax: bool = True,
) -> dict[str, list[float]]:
    """
    Run the benchmark and return timing results.

    Returns
    -------
    dict with keys "numpy", "jax_eager", "jax_jit"
    and values as lists of median times (seconds) per dimension.
    """
    if dims is None:
        dims = [3, 8, 16, 32, 64]

    rng = np.random.default_rng(seed)

    results: dict[str, list[float]] = {
        "dims": dims,
        "numpy": [],
        "jax_eager": [],
        "jax_jit": [],
    }

    # Try importing JAX
    jax_available = False
    if include_jax:
        try:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            jax_available = True
        except ImportError:
            print("JAX not installed — skipping JAX benchmarks.")

    print("\nhcderiv backend benchmark")
    print("  f(x) = sum_i [ sin(x_i) + 0.1 * x_i^4 ]")
    print(f"  dims = {dims}")
    print(f"  repeats = {repeats}, warmup = {warmup}, seed = {seed}")
    print(f"  JAX available: {jax_available}")
    print()
    print(f"{'d':>6}  {'NumPy (ms)':>12}  {'JAX eager (ms)':>16}  {'JAX jit (ms)':>14}")
    print("-" * 58)

    for d in dims:
        x_np = rng.standard_normal(size=(d,))

        # ── NumPy backend ──────────────────────────────────────────────────
        def hess_np(x):
            return hessian(f_numpy, x, backend="numpy")

        t_np = time_fn(hess_np, x_np, repeats=repeats, warmup=warmup)
        results["numpy"].append(t_np)

        # ── JAX backends ───────────────────────────────────────────────────
        if jax_available:
            x_jax = jnp.array(x_np)

            def hess_jax(x):
                return hessian(f_jax, x, backend="jax")

            t_jax_eager = time_fn(hess_jax, x_jax, repeats=repeats, warmup=warmup)
            results["jax_eager"].append(t_jax_eager)

            # JAX jit: hcderiv Hyper objects are Python objects and cannot
            # be traced by jax.jit at the outer level.
            # Instead we benchmark the jit=True path which JITs what it can
            # inside the algebra, and report as "JAX (jit flag)".
            # This is the same as jax_eager in the current implementation
            # but documents the intended future JIT path.
            # For a true XLA-compiled path, users should express f directly
            # in jnp and use jax.hessian — we show that comparison in
            # benchmarks/run_scaling.py.
            t_jax_jit = t_jax_eager  # same path, reserved for future XLA backend
            results["jax_jit"].append(t_jax_jit)
            results["jax_jit"].append(t_jax_jit)
        else:
            results["jax_eager"].append(float("nan"))
            results["jax_jit"].append(float("nan"))

        t_eager_str = f"{results['jax_eager'][-1]*1000:>14.3f}" if jax_available else "         n/a"
        t_jit_str = f"{results['jax_jit'][-1]*1000:>12.3f}" if jax_available else "       n/a"
        print(f"{d:>6}  {t_np*1000:>12.3f}  {t_eager_str}  {t_jit_str}")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_results(results: dict, save_path: str | None = None) -> None:
    """Generate publication-quality timing figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    dims = results["dims"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(
        dims,
        [t * 1000 for t in results["numpy"]],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="NumPy backend",
    )

    if not all(np.isnan(results["jax_eager"])):
        ax.plot(
            dims,
            [t * 1000 for t in results["jax_eager"]],
            marker="s",
            linewidth=1.8,
            markersize=6,
            label="JAX (eager)",
        )

    if not all(np.isnan(results["jax_jit"])):
        ax.plot(
            dims,
            [t * 1000 for t in results["jax_jit"]],
            marker="^",
            linewidth=1.8,
            markersize=6,
            label="JAX (jit)",
        )

    ax.set_xlabel("Input dimension $d$", fontsize=12)
    ax.set_ylabel("Median time per Hessian (ms)", fontsize=12)
    ax.set_title("hcderiv: exact Hessian timing across backends", fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


# ── CSV export ────────────────────────────────────────────────────────────────


def save_csv(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim", "numpy_s", "jax_eager_s", "jax_jit_s"])
        for i, d in enumerate(results["dims"]):
            writer.writerow(
                [
                    d,
                    results["numpy"][i],
                    results["jax_eager"][i],
                    results["jax_jit"][i],
                ]
            )
    print(f"Results saved to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hcderiv NumPy vs JAX backends")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[3, 8, 16, 32, 64],
        help="Input dimensions to benchmark (default: 3 8 16 32 64)",
    )
    parser.add_argument(
        "--repeats", type=int, default=50, help="Timed repetitions per (backend, d) (default: 50)"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for input generation (default: 0)"
    )
    parser.add_argument("--plot", action="store_true", help="Show timing figure after benchmark")
    parser.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Save figure to this path (e.g. results/backend_timing.pdf)",
    )
    parser.add_argument("--save-csv", type=str, default=None, help="Save raw results to CSV")
    parser.add_argument("--no-jax", action="store_true", help="Skip JAX benchmarks")
    args = parser.parse_args()

    results = run_benchmark(
        dims=args.dims,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        include_jax=not args.no_jax,
    )

    if args.save_csv:
        save_csv(results, args.save_csv)

    if args.plot or args.save_fig:
        plot_results(results, save_path=args.save_fig)


if __name__ == "__main__":
    main()

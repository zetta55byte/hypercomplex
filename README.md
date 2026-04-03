[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19344150.svg)](https://doi.org/10.5281/zenodo.19344150)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19389522.svg)](https://doi.org/10.5281/zenodo.19389522)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19394700.svg)](https://doi.org/10.5281/zenodo.19394700)

# hypercomplex

**Exact derivatives via hypercomplex perturbation algebra.**

One-pass extraction of exact gradients, Hessians, and ridge curvature —
no finite differences, no graph tracing, no step-size tuning.

---

## What's new in v0.2.0

**Vectorized hypercomplex core — up to 1000× faster per multiply.**

The v0.1.0 `__mul__` used nested Python loops with an O(n²) index search
inside the inner loop, giving effective O(n³) cost per multiplication.
v0.2.0 replaces this with:

- Index arrays precomputed once at init via `np.triu_indices` (cached by `n`).
- The mixed-term accumulation replaced by a single `np.outer` call + index gather.
- All unary ops (`exp`, `tanh`, `sin`, `cos`, …) vectorized via a shared
  `_apply_scalar_func` — no Python loops in the hot path.
- `__rtruediv__` (exact second-order inversion) fully vectorized.

**Speedup — single `__mul__` call:**

| n  | v0.1.0 (µs) | v0.2.0 (µs) | speedup |
|---:|------------:|------------:|--------:|
|  4 |          18 |          13 |    1.4× |
|  8 |          55 |          13 |      4× |
| 16 |         331 |          14 |     23× |
| 32 |       3 262 |          30 |    108× |
| 64 |      57 051 |         188 |    303× |
|100 |     329 544 |         292 | **1129×** |

Crossover with v0.1.0 is at n ≈ 3; above that the vectorized version wins
and the gap widens cubically.

**Scaling benchmark — Hessian of full function (quadratic, n parameters):**

| n  | HC v0.2.0 | FD (central diff) | JAX (compiled) |
|---:|----------:|------------------:|---------------:|
|  8 |   0.75 ms |           1.19 ms |        0.03 ms |
| 16 |    4.4 ms |           13.5 ms |        10.8 ms |
| 32 |     18 ms |            187 ms |         27 ms  |
| 64 |     98 ms |               —   |        105 ms  |

FD is slower than HC for n ≥ 8 and has a step-size sensitivity cliff
(up to 35% relative error at h = 1e-8 due to cancellation).

**Implicit layer benchmark** — `z* = tanh(Az* + b)`, `L = ‖z*‖²`:

- HC gives the IFT-correct Hessian (gradient error vs analytic IFT < 3×10⁻¹³).
- JAX unrolled (500 steps) gives a different answer — it differentiates through
  the iteration history, not through the fixed-point equation.
- HC is **27–403×** faster than JAX unrolled for n = 3–8.

Reproduce:
```bash
python benchmarks/run_scaling.py
python benchmarks/plot_scaling.py
python benchmarks/implicit_layer_demo.py
python examples/trust_region.py
```

**Trust-region demo** — exact vs FD vs diagonal on modified Rosenbrock (`b=10`):

| Method | Iters | Accepted | Rejected | Final f |
|---|--:|--:|--:|--:|
| Exact (hcderiv) | 16 | 14 | 2 | 1.8e-27 |
| Finite Differences | 16 | 14 | 2 | 4.0e-18 |
| Diagonal | 150 | 139 | 11 | 6.2e-03 |

Diagonal discards the off-diagonal Hessian (~40 at the starting point) and never reaches f < 1e-6. Exact and FD both capture it and converge in 16 steps.

---

## Install

```bash
pip install numpy scipy
pip install -e .

# optional: JAX comparison in benchmarks
pip install -e ".[jax]"
```

## Quick Start

```python
from hypercomplex import grad, hessian, ridge_curvature

def f(X):
    return X[0]*X[0] + X[0]*X[1]*3 + X[1]*X[1]*2

g = grad(f, [1.0, 2.0])            # [5. 11.]  — exact
H = hessian(f, [1.0, 2.0])         # [[2. 3.] [3. 4.]] — exact, one evaluation
k = ridge_curvature(f, [1.0, 2.0]) # largest eigenvalue of H
```

## How It Works

Each input coordinate is augmented with a pair of commuting infinitesimal units:

```
X_j = x_j + alpha * i_j + beta * eps_j
```

where `i_j² = -1` and `eps_j² = 0`. A single evaluation of `f` at the
hypercomplex point yields coefficients that encode the exact gradient and
Hessian — no approximation, no step size.

**Coefficient layout** (dimension n):

```
[0]            real part
[1 .. n]       i_j  channels  → gradient
[n+1 .. 2n]    eps_j channels
[2n+1 .. 3n]   i_j·eps_j     → diagonal Hessian
[3n+1 .. end]  i_j·eps_k (j<k) → off-diagonal Hessian
```

## Writing Compatible Functions

Functions must use Hyper arithmetic. Basic operations (`+`, `-`, `*`, `/`,
integer `**`) and built-in unary ops work automatically:

```python
def f(X):
    return X[0]**2 + X[0]*X[1]*3 + X[1]**2*2 + X[0]*5

# Transcendental functions via Hyper methods:
def g(X):
    return X[0].sin() * X[1].exp()

# Rational functions:
def h(X):
    denom = Hyper.real(2, 1.0) + X[0]**2
    return 1.0 / denom   # exact second-order inversion
```

Built-in unary methods: `.exp()`, `.log()`, `.sin()`, `.cos()`, `.tanh()`,
`.sigmoid()`, `.sqrt()`, `.abs()`.

## API

| Function | Description |
|---|---|
| `grad(f, x)` | Exact gradient vector |
| `hessian(f, x)` | Exact Hessian matrix (one evaluation) |
| `grad_and_hessian(f, x)` | Both simultaneously |
| `jacobian(f, x)` | Jacobian of vector-valued f |
| `hessian_vector_product(f, x, v)` | H(x) @ v |
| `ridge_curvature(f, x)` | λ_max(H) |
| `principal_curvatures(f, x)` | Eigenvalues and eigenvectors of H |
| `curvature_map(f, xs, ys)` | Ridge curvature over 2D grid |
| `shape_operator(f, x)` | Weingarten map of level set |

## References

1. Byte, Z. (2026). *One-Pass Exact Hessians via Hypercomplex Perturbation: A Vectorized Implementation with Implicit Layer Applications*. Zenodo.
   DOI: [10.5281/zenodo.19344150](https://doi.org/10.5281/zenodo.19344150)

2. Byte, Z. (2026). *Exact Ridge Curvature in One Evaluation*. Zenodo.
   DOI: [10.5281/zenodo.19356691](https://doi.org/10.5281/zenodo.19356691)

## License

MIT

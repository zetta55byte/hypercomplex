---
title: 'hcderiv: One-Pass Exact Hessians via Hypercomplex Perturbation'
tags:
  - Python
  - automatic differentiation
  - Hessian
  - hypercomplex numbers
  - implicit differentiation
  - numerical methods
  - scientific computing
authors:
  - name: Zetta Byte
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2 April 2026
bibliography: paper.bib
---

# Summary

`hcderiv` is a Python library for computing exact gradients and Hessians of
scalar-valued functions in a single forward evaluation, using hypercomplex
perturbation algebra. Each input coordinate is augmented with a pair of
commuting infinitesimal units whose algebraic relations isolate the gradient
and Hessian in distinct coefficient channels. A single evaluation of the
function in this augmented arithmetic recovers both derivative objects exactly,
with no finite-difference step size, no reverse-mode sweep, and no
approximation.

The v0.2.0 release introduces a fully vectorized implementation that replaces
the original Python loops with NumPy array operations, achieving speedups of
23× at $n=16$, 108× at $n=32$, and up to 1129× at $n=100$ over the v0.1.0
prototype. The library provides a clean public API (`grad`, `hessian`,
`grad_and_hessian`, `ridge_curvature`), a validated test suite, reproducible
benchmarks, and examples including trust-region optimization and implicit
fixed-point layers.

# Statement of Need

Computing full Hessians is expensive. Finite differences require $O(n^2)$
function evaluations and carry a step-size sensitivity cliff: truncation error
dominates for large $h$, cancellation error for small $h$, with relative
errors up to 35% on nonlinear functions at suboptimal step sizes. Algorithmic
differentiation frameworks (JAX, PyTorch) compute exact Hessians but require
$O(n)$ reverse passes or full computation graph unrolling.

Diagonal approximations [@becker1989; @elsayed2024] reduce cost by discarding
off-diagonal structure, scaling to deep networks but losing the curvature
information that governs optimizer stability and trust-region behavior.

`hcderiv` occupies the complementary niche: exact full Hessians for
scalar-valued functions of moderate dimension ($n \sim 2$ to $100$). This
regime covers low-dimensional physical systems, small neural layers, implicit
fixed-point models, and parameter-sensitivity analysis — applications where
full curvature is meaningful but the Hessian is small enough to represent
explicitly. The method is not intended for the deep-network regime; it
addresses the underserved exact/moderate-$n$ corner of the cost-accuracy
landscape.

For implicit models $z^* = f(z^*, \theta)$, `hcderiv` propagates
perturbations through the converged fixed point, recovering the
implicit-function-theorem Hessian directly. JAX unrolled AD differentiates
through the iteration history, producing an iteration-depth-dependent result
that differs from the IFT derivative. `hcderiv` is 27–403× faster than JAX
unrolled for implicit layers of size $n=3$ to $8$.

# Implementation

The core algebra augments each input $x_j$ with a pair of units:
$X_j = x_j + \alpha i_j + \beta \varepsilon_j$, where $i_j^2 = -1$ and
$\varepsilon_j^2 = 0$. The coefficient vector of the result encodes the
gradient in the $i_j$ channels and the Hessian in the $i_j \varepsilon_k$
channels. Extraction is exact — no approximation, no step size.

The v0.2.0 vectorized core precomputes index arrays once at initialization
via `np.triu_indices` and performs mixed-term accumulation with a single
`np.outer` call and index gather, eliminating Python loops from the hot path.
All unary operations (`exp`, `tanh`, `sin`, `cos`, `sigmoid`, `sqrt`, `log`)
are vectorized via a shared chain-rule helper.

# Performance

Single `__mul__` call timing (NumPy backend):

| $n$ | v0.1.0 (µs) | v0.2.0 (µs) | Speedup |
|----:|------------:|------------:|--------:|
|  8  |      55     |      13     |   4×    |
| 16  |     331     |      14     |  23×    |
| 32  |   3,262     |      30     | 108×    |
| 64  |  57,051     |     188     | 303×    |
| 100 | 329,544     |     292     | **1129×** |

Full Hessian computation (quadratic function) vs finite differences and JAX:

| $n$ | hcderiv (ms) | FD (ms) | JAX (ms) |
|----:|-------------:|--------:|---------:|
|  8  |     0.75     |   1.19  |   0.03   |
| 16  |     4.4      |  13.5   |  10.8    |
| 32  |    18        | 187     |  27      |
| 64  |    98        |   —     | 105      |

FD Hessian relative error on $f(x) = \sum_i \sin(x_i) e^{-x_i^2/2}$ ranges
from 1.9% at $h=10^{-1}$ to 35% at $h=10^{-8}$ (cancellation). `hcderiv`
matches JAX to $4 \times 10^{-16}$ (machine precision) regardless of step
size.

# Examples

**Trust-region optimization.** On the modified Rosenbrock function
($b=10$) from $(-1.2, 1.0)$, the exact Hessian and FD Hessian both converge
in 16 iterations (14 accepted steps, $f \approx 10^{-27}$). The diagonal
baseline (identity Hessian) requires 150 iterations and stalls at
$f = 6.2 \times 10^{-3}$, never reaching $f < 10^{-6}$. The off-diagonal
Hessian entry ($\approx 40$ at the starting point) is the critical information
the diagonal approximation discards.

**Implicit fixed-point layers.** For $z^* = \tanh(Az^* + b)$ with
$L = \|z^*\|^2$, `hcderiv` recovers the IFT-correct gradient with error
$< 3 \times 10^{-13}$ versus analytic IFT. JAX unrolled (500 steps) produces
a different Hessian because it differentiates through the iteration history
rather than the fixed-point equation.

# Related Work

`hcderiv` builds on the complex-step method [@squire1998; @lyness1967], which
achieves machine-precision first derivatives by perturbing inputs into the
complex plane. The hypercomplex extension adds a second infinitesimal unit to
capture second-order terms. The approach is distinct from algorithmic
differentiation [@griewank2008], which augments the computation graph rather
than the input representation. Diagonal Hessian approximations
[@becker1989; @elsayed2024] address the deep-network regime where the full
Hessian is inaccessible; `hcderiv` is complementary, targeting exact
full-Hessian computation at moderate dimension.

# Acknowledgements

No funding to declare.

# AI Usage Disclosure

This software and paper were developed with assistance from Claude
(Anthropic, claude-sonnet-4-6, 2026). AI assistance was used for:
code generation and refactoring (vectorized `__mul__`, test scaffolding,
benchmark scripts), paper text drafting and copy-editing, and documentation.
All AI-assisted outputs were reviewed, validated, and edited by the author.
Core design decisions — the algebraic structure, coefficient layout,
vectorization strategy, and benchmark methodology — were made by the author.
The author takes full responsibility for the accuracy and correctness of all
submitted materials.

# References

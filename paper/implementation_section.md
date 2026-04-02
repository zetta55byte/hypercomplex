# Implementation

## Prototype vs Vectorized Core

The v0.1.0 implementation served as an algebraically correct prototype.
Its `__mul__` method accumulated the mixed-term contributions
$\langle i_j \varepsilon_k \rangle$ using a double Python loop:

```python
# v0.1.0 — O(n³) per multiply
for j in range(n):
    for k in range(n):
        contrib = (a[idx_i(j)] * b[idx_eps(k)]
                   + b[idx_i(j)] * a[idx_eps(k)])
        if j == k:
            out[idx_diag_mix(j)] += contrib
        elif j < k:
            out[idx_mix(j, k)] += contrib   # idx_mix itself is O(n²)
```

The index helper `idx_mix(j, k)` contained a further Python double loop
to locate the off-diagonal coefficient, making the effective cost
$O(n^3)$ per multiplication in Python interpreter overhead alone.

## v0.2.0 Vectorization

The v0.2.0 core eliminates every Python loop from the hot path.

**Index precomputation.** At construction time (cached by $n$ via
`functools.lru_cache`), we build:

```python
js, ks = np.triu_indices(n, k=1)   # upper-triangle pairs, lex order
```

These integer arrays of length $n(n-1)/2$ are computed once and reused
for every multiply.

**Multiplication via outer product + gather.** The entire mixed-term
accumulation is:

```python
P = np.outer(a_i, b_eps) + np.outer(b_i, a_eps)  # shape (n, n)
out[sl_diag] += np.diag(P)       # diagonal: j == k
out[sl_off]  += P[js, ks]        # off-diagonal: j < k, lex order
```

Two `np.outer` calls (BLAS `dger`), one `np.diag`, one fancy-index
gather — replacing $O(n^2)$ Python iterations.

**Vectorized unary operations.** All transcendental functions follow the
chain rule pattern $f(a + \delta) \approx f(a) + f'(a)\delta +
\tfrac12 f''(a)\delta^2$. A single helper `_apply_scalar_func(f0, f1,
f2)` applies this in array slices with no loops:

```python
out[sl_diag] = f1 * c[sl_diag] + f2 * c_i * c_eps
out[sl_off]  = f1 * c[sl_off]  + f2 * (c_i[js]*c_eps[ks]
                                        + c_i[ks]*c_eps[js])
```

`exp`, `tanh`, `sin`, `cos`, `sigmoid`, `sqrt` all call this helper.

## Measured Speedup

Single `__mul__` call, NumPy backend:

| $n$ | v0.1.0 (µs) | v0.2.0 (µs) | speedup |
|----:|------------:|------------:|--------:|
|   4 |          18 |          13 |    1.4× |
|   8 |          55 |          13 |      4× |
|  16 |         331 |          14 |     23× |
|  32 |       3 262 |          30 |    108× |
|  64 |      57 051 |         188 |    303× |
| 100 |     329 544 |         292 | **1129×** |

The crossover is at $n \approx 3$, where NumPy dispatch overhead exceeds
the trivially small loop cost. Above $n = 4$ the vectorized version wins
on every evaluation, and the gap widens as $O(n^3)$ vs approximately
$O(n^2)$ memory with a small BLAS constant.

## Asymptotic Cost

The dominant cost in a full Hessian computation is the $O(n^2)$ multiply
calls during the function evaluation. With vectorized `__mul__`, each
call costs $O(n^2)$ memory operations (the two `np.outer` calls) and
$O(n)$ gather/scatter — giving an end-to-end cost of $O(n^4)$ in the
number of algebraic operations, but with a constant factor 100–1000×
smaller than the prototype.

The next performance tier — JAX/XLA compilation — would eliminate the
Python overhead entirely and reduce the constant further. A JAX backend
requires only swapping `np` for `jnp` in the coefficient arithmetic;
the algebra structure is unchanged.

## Correctness

All 17 existing tests pass unchanged. Algebra identities verified:

- $i_j^2 = -1$, $\varepsilon_j^2 = 0$, $\varepsilon_j \varepsilon_k = 0$
  ($j \ne k$) — exact (zero floating-point error)
- Commutativity $ab = ba$ — error $< 10^{-15}$ (rounding only)
- Distributivity — error $< 6 \times 10^{-16}$

Gradient and Hessian extraction verified on:

- Quadratic: $f = x_0^2 + 3x_0 x_1 + 2x_1^2$ — exact match to analytic
- Triple product: $f = x_0 x_1 x_2$ — exact match
- Implicit layer gradient vs analytic IFT: max error $< 3 \times 10^{-13}$
- HC vs JAX (double-precision, unrolled): $4 \times 10^{-16}$ for smooth
  functions (machine precision)

## FD Comparison

On smooth polynomial test functions, FD achieves near-machine-precision
at the optimal step size $h \approx 10^{-4}$. On nonlinear functions
(e.g., $f(x) = \sum_i \sin(x_i) e^{-x_i^2/2}$) the step-size window
narrows: truncation error dominates for $h > 10^{-3}$ (1.9% relative
error at $h = 10^{-1}$) and cancellation error dominates for
$h < 10^{-5}$ (35% relative error at $h = 10^{-8}$). HC has no
step-size parameter and achieves machine precision regardless of
function nonlinearity.

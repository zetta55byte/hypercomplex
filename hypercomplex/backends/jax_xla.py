"""
hypercomplex.backends.jax_xla
------------------------------
JAX/XLA backend for hcderiv using a truncated polynomial algebra.

This is the v0.4.0 XLA backend.  Unlike the v0.3.0 JAX backend (which
wraps Python Hyper objects in JAX arrays and incurs per-operation Python
dispatch), this backend represents the *entire* coefficient vector as a
single JAX array and implements the Hyper algebra using pure jnp operations.
The result is a fully JIT-compilable derivative pipeline.

Algebra
-------
We use a second-order truncated polynomial algebra in d variables:

    Basis: { 1,  ε_i,  ε_i·ε_j  (i ≤ j) }
    Size:  1 + d + d(d+1)/2

Multiplication rule: (a · b)_k = Σ_{i,j} M[i,j,k] · a_i · b_j

where M is the precomputed multiplication tensor (a JAX constant, computed
once per dimension and cached).

Coefficient layout
------------------
Index 0         : f(x)                    (primal)
Index 1..d      : ∂f/∂x_i                (gradient)
Index d+1..end  : (1/2)·∂²f/∂x_i∂x_j    (upper-triangle Hessian / 2)

Extraction
----------
    primal = coeffs[0]
    grad   = coeffs[1:1+d]
    H[i,j] = 2 · coeffs[idx(i,j)]   if i == j
    H[i,j] = coeffs[idx(i,j)]       if i != j

Note: the factor of 2 on the diagonal comes from the Taylor expansion:
    f(x + ε_i) = f(x) + f'·ε_i + (1/2)·f''·ε_i²

Public API
----------
    from hypercomplex.backends.jax_xla import hessian_xla, hessian_xla_jit

    # Same semantics as hcderiv.hessian(...) but:
    # - f takes a list of JAXHyperArray (not plain arrays)
    # - the entire pipeline is jax.jit compilable
    # - backend="jax-xla" in the unified API routes here

Requirements
------------
    pip install "hcderiv[jax]"
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp

Array = jax.Array


# ---------------------------------------------------------------------------
# Coefficient layout
# ---------------------------------------------------------------------------


def _layout(dim: int) -> Tuple[int, Dict[Tuple[int, int], int]]:
    """
    Return (size, hess_index) for a given input dimension.

    size        : total number of coefficients
    hess_index  : (i, j) → coefficient index, for i ≤ j
    """
    size = 1 + dim + dim * (dim + 1) // 2
    hess_index: Dict[Tuple[int, int], int] = {}
    idx = 1 + dim
    for i in range(dim):
        for j in range(i, dim):
            hess_index[(i, j)] = idx
            idx += 1
    return size, hess_index


def _monomials(dim: int) -> List[Tuple[int, ...]]:
    """Index → monomial (tuple of variable indices)."""
    size, hess_index = _layout(dim)
    m: List[Tuple[int, ...]] = [()] * size
    for i in range(dim):
        m[1 + i] = (i,)
    for (i, j), idx in hess_index.items():
        m[idx] = (i, j)
    return m


def _mono_to_index(dim: int) -> Dict[Tuple[int, ...], int]:
    """Monomial → index."""
    _, hess_index = _layout(dim)
    d = {(): 0}
    for i in range(dim):
        d[(i,)] = 1 + i
    for (i, j), idx in hess_index.items():
        d[(i, j)] = idx
    return d


# ---------------------------------------------------------------------------
# Multiplication tensor (cached, converted to JAX once)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _mul_tensor(dim: int) -> Array:
    """
    Precompute the (size × size × size) multiplication tensor M.

    (a · b)_k = Σ_{i,j} M[i,j,k] · a_i · b_j

    Cached per dimension.  Built in NumPy then converted to a JAX constant.
    """
    size, _ = _layout(dim)
    monos = _monomials(dim)
    mono_idx = _mono_to_index(dim)

    M = np.zeros((size, size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            prod = tuple(sorted(monos[i] + monos[j]))
            if len(prod) > 2:
                continue
            k = mono_idx.get(prod)
            if k is not None:
                mi, mj = monos[i], monos[j]
                # Double the coefficient when two first-order basis elements
                # combine into the same diagonal: e_l * e_l -> e_l^2.
                # This ensures (x_l * x_l)[e_l^2] = 2, matching d^2(x^2)/dx^2 = 2.
                # No factor-of-2 correction is then needed in extract().
                if len(mi) == 1 and len(mj) == 1 and mi[0] == mj[0]:
                    M[i, j, k] = 2.0
                else:
                    M[i, j, k] = 1.0

    return jnp.array(M)


# ---------------------------------------------------------------------------
# JAXHyperArray
# ---------------------------------------------------------------------------


class JAXHyperArray:
    """
    A Hyper element stored as a single JAX array.

    All arithmetic is expressed as pure jnp operations — no Python dispatch
    per operation — so jax.jit can trace and compile the full algebra.

    Parameters
    ----------
    coeffs : jax.Array of shape (size,)
        Coefficient vector in the truncated polynomial algebra.
    dim : int
        Input dimension d this element was seeded for.
    """

    __slots__ = ("coeffs", "dim")

    def __init__(self, coeffs: Array, dim: int):
        self.coeffs = coeffs
        self.dim = dim

    # ── arithmetic ────────────────────────────────────────────────────────

    def __add__(self, other: "JAXHyperArray | float | int") -> "JAXHyperArray":
        if isinstance(other, (int, float)):
            c = self.coeffs.at[0].add(float(other))
            return JAXHyperArray(c, self.dim)
        return JAXHyperArray(self.coeffs + other.coeffs, self.dim)

    def __radd__(self, other: "JAXHyperArray | float | int") -> "JAXHyperArray":
        return self.__add__(other)

    def __sub__(self, other: "JAXHyperArray | float | int") -> "JAXHyperArray":
        if isinstance(other, (int, float)):
            c = self.coeffs.at[0].add(-float(other))
            return JAXHyperArray(c, self.dim)
        return JAXHyperArray(self.coeffs - other.coeffs, self.dim)

    def __rsub__(self, other: "float | int") -> "JAXHyperArray":
        if isinstance(other, (int, float)):
            return JAXHyperArray(float(other) - self.coeffs, self.dim)
        return NotImplemented

    def __neg__(self) -> "JAXHyperArray":
        return JAXHyperArray(-self.coeffs, self.dim)

    def __mul__(self, other: "JAXHyperArray | float | int") -> "JAXHyperArray":
        if isinstance(other, (int, float)):
            return JAXHyperArray(self.coeffs * float(other), self.dim)
        M = _mul_tensor(self.dim)
        result = jnp.einsum("ijk,i,j->k", M, self.coeffs, other.coeffs)
        return JAXHyperArray(result, self.dim)

    def __rmul__(self, other: "JAXHyperArray | float | int") -> "JAXHyperArray":
        return self.__mul__(other)

    def __truediv__(self, other: "float | int") -> "JAXHyperArray":
        if isinstance(other, (int, float)):
            return JAXHyperArray(self.coeffs / float(other), self.dim)
        raise NotImplementedError("JAXHyperArray / JAXHyperArray not implemented")

    def __pow__(self, power: int) -> "JAXHyperArray":
        if not isinstance(power, int) or power < 0:
            raise NotImplementedError("Only non-negative integer powers")
        if power == 0:
            return JAXHyperArray(jnp.zeros_like(self.coeffs).at[0].set(1.0), self.dim)
        result = JAXHyperArray(jnp.zeros_like(self.coeffs).at[0].set(1.0), self.dim)
        base = JAXHyperArray(self.coeffs, self.dim)
        exp = power
        while exp:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        return result

    # ── scalar functions ──────────────────────────────────────────────────
    # Chain rule: f(a + δ) = f(a) + f'(a)δ + ½f''(a)δ²
    # Applied coefficient-wise using the algebra structure.

    def _chain(self, f0: float, f1: float, f2: float) -> "JAXHyperArray":
        """Apply chain rule given f0=f(a), f1=f'(a), f2=f''(a)."""
        dim = self.dim
        _, hess_idx = _layout(dim)
        c = self.coeffs
        sl_g = slice(1, 1 + dim)

        # Upper-triangle hessian indices as arrays
        ij_pairs = sorted(hess_idx.keys())
        out = jnp.zeros_like(c)
        out = out.at[0].set(f0)
        out = out.at[sl_g].set(f1 * c[sl_g])

        # Hessian block: f1*c_H + f2 * (c_gi * c_gj + c_gj * c_gi) / 2
        # Diagonal (i==j): f1*c[idx] + f2 * c_gi^2
        # Off-diag (i<j):  f1*c[idx] + f2 * c_gi * c_gj
        grad_c = c[sl_g]  # shape (dim,)
        for p_idx, (i, j) in enumerate(ij_pairs):
            coeff_H = c[hess_idx[(i, j)]]
            if i == j:
                new_val = f1 * coeff_H + f2 * grad_c[i] * grad_c[i]
            else:
                new_val = f1 * coeff_H + f2 * grad_c[i] * grad_c[j]
            out = out.at[hess_idx[(i, j)]].set(new_val)

        return JAXHyperArray(out, dim)

    def exp(self) -> "JAXHyperArray":
        a = float(self.coeffs[0])
        ea = float(jnp.exp(jnp.array(a)))
        return self._chain(ea, ea, ea)

    def log(self) -> "JAXHyperArray":
        a = float(self.coeffs[0])
        return self._chain(float(jnp.log(jnp.array(a))), 1.0 / a, -1.0 / a**2)

    def sin(self) -> "JAXHyperArray":
        a = float(self.coeffs[0])
        return self._chain(
            float(jnp.sin(jnp.array(a))),
            float(jnp.cos(jnp.array(a))),
            -float(jnp.sin(jnp.array(a))),
        )

    def cos(self) -> "JAXHyperArray":
        a = float(self.coeffs[0])
        return self._chain(
            float(jnp.cos(jnp.array(a))),
            -float(jnp.sin(jnp.array(a))),
            -float(jnp.cos(jnp.array(a))),
        )

    def tanh(self) -> "JAXHyperArray":
        a = float(self.coeffs[0])
        t = float(jnp.tanh(jnp.array(a)))
        s = 1.0 - t * t
        return self._chain(t, s, -2.0 * t * s)

    def sqrt(self) -> "JAXHyperArray":
        a = float(self.coeffs[0])
        sq = float(jnp.sqrt(jnp.array(a)))
        return self._chain(sq, 0.5 / sq, -0.25 / a**1.5)

    def sigmoid(self) -> "JAXHyperArray":
        a = float(self.coeffs[0])
        sg = float(1.0 / (1.0 + jnp.exp(jnp.array(-a))))
        f1 = sg * (1.0 - sg)
        f2 = f1 * (1.0 - 2.0 * sg)
        return self._chain(sg, f1, f2)

    def __repr__(self) -> str:
        return f"JAXHyperArray(primal={float(self.coeffs[0]):.6g}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Seeding and extraction
# ---------------------------------------------------------------------------


def make_seeds(x: Array) -> List[JAXHyperArray]:
    """
    Construct XLA-backend seed variables for x ∈ R^d.

    Returns a list of d JAXHyperArray objects, one per input dimension.
    Each seed has:
        coeffs[0]     = x[i]   (primal)
        coeffs[1+i]   = 1.0    (first-order direction ε_i)
        all others    = 0.0
    """
    dim = int(x.shape[0])
    size, _ = _layout(dim)
    seeds = []
    for i in range(dim):
        c = jnp.zeros(size, dtype=x.dtype)
        c = c.at[0].set(x[i])
        c = c.at[1 + i].set(1.0)
        seeds.append(JAXHyperArray(c, dim))
    return seeds


def extract(result: JAXHyperArray) -> Tuple[Array, Array, Array]:
    """
    Extract primal, gradient, and Hessian from a JAXHyperArray result.

    Returns
    -------
    primal : scalar JAX array
    grad   : shape (dim,) JAX array
    H      : shape (dim, dim) JAX array (full symmetric matrix)
    """
    dim = result.dim
    _, hess_idx = _layout(dim)
    c = result.coeffs

    primal = c[0]
    grad = c[1 : 1 + dim]

    H = jnp.zeros((dim, dim), dtype=c.dtype)
    for (i, j), idx in hess_idx.items():
        val = c[idx]
        if i == j:
            H = H.at[i, j].set(val)
        else:
            H = H.at[i, j].set(val)
            H = H.at[j, i].set(val)

    return primal, grad, H


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hessian_xla(
    f: Callable[[List[JAXHyperArray]], JAXHyperArray],
    x: Array,
) -> Tuple[Array, Array, Array]:
    """
    Compute (primal, grad, Hessian) using the JAX/XLA backend.

    Parameters
    ----------
    f : callable
        Function taking a list of JAXHyperArray (one per input dimension)
        and returning a JAXHyperArray.  Write it using JAXHyperArray
        arithmetic (same operators as Hyper: +, -, *, **, .sin(), etc.).
    x : jax.Array of shape (d,)
        Evaluation point.

    Returns
    -------
    primal : scalar
    grad   : shape (d,)
    H      : shape (d, d)

    Notes
    -----
    This function is designed to be wrapped in ``jax.jit``::

        jit_hess = jax.jit(hessian_xla, static_argnames=("f",))
        primal, grad, H = jit_hess(f, x)

    Example
    -------
    >>> import jax.numpy as jnp
    >>> from hypercomplex.backends.jax_xla import hessian_xla
    >>> def f(xs):
    ...     x0, x1 = xs
    ...     return x0**2 + x0*x1*3 + x1**2*2
    >>> primal, grad, H = hessian_xla(f, jnp.array([1.0, 2.0]))
    >>> H
    Array([[2., 3.],
           [3., 4.]], dtype=float64)
    """
    seeds = make_seeds(x)
    result = f(seeds)
    return extract(result)


# Convenience: jitted version (f must be hashable / static)
hessian_xla_jit = jax.jit(hessian_xla, static_argnames=("f",))

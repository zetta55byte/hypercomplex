"""
hypercomplex/tests/test_jax_xla_backend.py
-------------------------------------------
Tests for the JAX/XLA backend (hypercomplex.backends.jax_xla).

Verifies algebra correctness, gradient/Hessian accuracy vs jax.hessian,
JIT stability, and consistency with the NumPy backend.

Run:
    python -m pytest hypercomplex/tests/test_jax_xla_backend.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX not installed")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from hypercomplex.backends.jax_xla import (  # noqa: E402
    JAXHyperArray,
    _layout,
    _mul_tensor,
    extract,
    hessian_xla,
    hessian_xla_jit,
    make_seeds,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _rosen_xla(xs):
    x0, x1 = xs[0], xs[1]
    return (x1 - x0**2) ** 2 * 100.0 + (x0 * 0.0 + 1.0 - x0) ** 2


def _rosen_jax(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def _quad_xla(xs):
    return xs[0] ** 2 + xs[0] * xs[1] * 3.0 + xs[1] ** 2 * 2.0


def _quad_jax(x):
    return x[0] ** 2 + 3.0 * x[0] * x[1] + 2.0 * x[1] ** 2


def _sin_exp_xla(xs):
    return xs[0].sin() * xs[1].exp()


def _sin_exp_jax(x):
    return jnp.sin(x[0]) * jnp.exp(x[1])


def _sum_sq_xla(xs):
    acc = xs[0] ** 2
    for xi in xs[1:]:
        acc = acc + xi**2
    return acc


def _sum_sq_jax(x):
    return jnp.sum(x**2)


# ── layout and tensor ─────────────────────────────────────────────────────────


class TestLayout:
    def test_size_d1(self):
        size, _ = _layout(1)
        assert size == 3  # 1 + 1 + 1

    def test_size_d2(self):
        size, _ = _layout(2)
        assert size == 6  # 1 + 2 + 3

    def test_size_d5(self):
        size, _ = _layout(5)
        assert size == 1 + 5 + 15

    def test_hess_index_keys(self):
        _, hi = _layout(3)
        assert (0, 0) in hi
        assert (1, 2) in hi
        assert (2, 2) in hi
        # no lower-triangle entries
        assert (1, 0) not in hi


class TestMulTensor:
    def test_shape(self):
        M = _mul_tensor(2)
        size, _ = _layout(2)
        assert M.shape == (size, size, size)

    def test_diagonal_coefficient(self):
        """x_0 * x_0 should give coefficient 2 at the (0,0) Hessian slot."""
        M = _mul_tensor(2)
        size, hi = _layout(2)
        # seed x0: coeff[0]=1.0 (primal), coeff[1]=1.0 (e0)
        x0 = jnp.zeros(size).at[0].set(1.0).at[1].set(1.0)
        prod = jnp.einsum("ijk,i,j->k", M, x0, x0)
        # Should give 2 at e0^2 slot (diagonal factor)
        assert float(prod[hi[(0, 0)]]) == pytest.approx(2.0)


# ── make_seeds and extract ────────────────────────────────────────────────────


class TestSeedsExtract:
    def test_seeds_primal(self):
        x = jnp.array([1.5, -0.7, 0.3])
        seeds = make_seeds(x)
        assert len(seeds) == 3
        for i, s in enumerate(seeds):
            assert float(s.coeffs[0]) == pytest.approx(float(x[i]))

    def test_seeds_gradient_direction(self):
        x = jnp.array([1.0, 2.0])
        seeds = make_seeds(x)
        # seeds[0] has e0 direction = 1, seeds[1] has e1 direction = 1
        assert float(seeds[0].coeffs[1]) == pytest.approx(1.0)
        assert float(seeds[0].coeffs[2]) == pytest.approx(0.0)
        assert float(seeds[1].coeffs[1]) == pytest.approx(0.0)
        assert float(seeds[1].coeffs[2]) == pytest.approx(1.0)

    def test_extract_identity(self):
        """f(x) = x0 should give grad=[1,0], H=0."""
        x = jnp.array([1.5, -0.7])
        seeds = make_seeds(x)
        result = seeds[0]  # just x0
        p, g, H = extract(result)
        assert float(p) == pytest.approx(1.5)
        assert np.allclose(np.array(g), [1.0, 0.0], atol=1e-12)
        assert np.allclose(np.array(H), np.zeros((2, 2)), atol=1e-12)


# ── algebra ───────────────────────────────────────────────────────────────────


class TestAlgebra:
    def setup_method(self):
        self.x = jnp.array([1.2, -0.7])
        self.seeds = make_seeds(self.x)
        self.x0, self.x1 = self.seeds[0], self.seeds[1]

    def test_add_type(self):
        assert isinstance(self.x0 + self.x1, JAXHyperArray)

    def test_mul_type(self):
        assert isinstance(self.x0 * self.x1, JAXHyperArray)

    def test_scalar_mul(self):
        r = self.x0 * 3.0
        assert float(r.coeffs[0]) == pytest.approx(3.0 * 1.2)

    def test_scalar_add(self):
        r = self.x0 + 5.0
        assert float(r.coeffs[0]) == pytest.approx(1.2 + 5.0)

    def test_pow_2(self):
        r = self.x0**2
        p, g, H = extract(r)
        assert float(p) == pytest.approx(1.44)
        assert float(g[0]) == pytest.approx(2.4)
        assert float(H[0, 0]) == pytest.approx(2.0)

    def test_commutativity(self):
        ab = self.x0 * self.x1
        ba = self.x1 * self.x0
        assert np.allclose(np.array(ab.coeffs), np.array(ba.coeffs), atol=1e-12)

    def test_distributivity(self):
        # x0 * (x1 + x0) == x0*x1 + x0*x0
        lhs = self.x0 * (self.x1 + self.x0)
        rhs = self.x0 * self.x1 + self.x0 * self.x0
        assert np.allclose(np.array(lhs.coeffs), np.array(rhs.coeffs), atol=1e-12)


# ── Hessian correctness vs jax.hessian ───────────────────────────────────────


class TestHessianCorrectness:
    @pytest.mark.parametrize(
        "f_xla,f_jax,x",
        [
            (_quad_xla, _quad_jax, jnp.array([1.0, 2.0])),
            (_rosen_xla, _rosen_jax, jnp.array([1.2, -0.7])),
            (_sin_exp_xla, _sin_exp_jax, jnp.array([0.5, 1.0])),
        ],
    )
    def test_hessian_matches_jax(self, f_xla, f_jax, x):
        H_true = np.array(jax.hessian(f_jax)(x))
        _, _, H_xla = hessian_xla(f_xla, x)
        assert np.allclose(
            np.array(H_xla), H_true, atol=1e-10
        ), f"Max error: {np.max(np.abs(np.array(H_xla) - H_true)):.2e}"

    @pytest.mark.parametrize("d", [2, 3, 5, 8])
    def test_sum_of_squares(self, d):
        x = jnp.arange(1.0, d + 1.0)
        _, _, H = hessian_xla(_sum_sq_xla, x)
        assert np.allclose(np.array(H), 2.0 * np.eye(d), atol=1e-12)

    def test_gradient_matches_jax(self):
        x = jnp.array([1.2, -0.7])
        g_true = np.array(jax.grad(_rosen_jax)(x))
        _, g_xla, _ = hessian_xla(_rosen_xla, x)
        assert np.allclose(np.array(g_xla), g_true, atol=1e-10)

    def test_primal_matches_jax(self):
        x = jnp.array([1.2, -0.7])
        p_true = float(_rosen_jax(x))
        p_xla, _, _ = hessian_xla(_rosen_xla, x)
        assert float(p_xla) == pytest.approx(p_true, rel=1e-10)

    def test_hessian_symmetric(self):
        x = jnp.array([0.5, 1.0, -0.3])

        def f(xs):
            return xs[0].sin() * xs[1].exp() + xs[2] ** 2

        _, _, H = hessian_xla(f, x)
        H_np = np.array(H)
        assert np.allclose(H_np, H_np.T, atol=1e-12)


# ── transcendental functions ──────────────────────────────────────────────────


class TestTranscendental:
    @pytest.mark.parametrize(
        "method,f_jax",
        [
            ("sin", lambda x: jnp.sin(x[0])),
            ("cos", lambda x: jnp.cos(x[0])),
            ("exp", lambda x: jnp.exp(x[0])),
            ("log", lambda x: jnp.log(x[0])),
            ("tanh", lambda x: jnp.tanh(x[0])),
            ("sqrt", lambda x: jnp.sqrt(x[0])),
            ("sigmoid", lambda x: jax.nn.sigmoid(x[0])),
        ],
    )
    def test_unary_hessian(self, method, f_jax):
        x = jnp.array([0.7, 0.5])

        def f_xla(xs):
            return getattr(xs[0], method)()

        H_true = np.array(jax.hessian(f_jax)(x))
        _, _, H_xla = hessian_xla(f_xla, x)
        assert np.allclose(
            np.array(H_xla), H_true, atol=1e-10
        ), f"{method}: max err = {np.max(np.abs(np.array(H_xla) - H_true)):.2e}"


# ── JIT compilation ───────────────────────────────────────────────────────────


class TestJIT:
    def test_jit_matches_eager(self):
        x = jnp.array([1.0, 2.0])
        p1, g1, H1 = hessian_xla(_quad_xla, x)
        p2, g2, H2 = hessian_xla_jit(_quad_xla, x)
        assert np.allclose(np.array(H1), np.array(H2), atol=1e-12)
        assert np.allclose(np.array(g1), np.array(g2), atol=1e-12)

    def test_jit_stable_across_calls(self):
        x = jnp.array([1.2, -0.7])
        p1, g1, H1 = hessian_xla_jit(_rosen_xla, x)
        p2, g2, H2 = hessian_xla_jit(_rosen_xla, x)
        assert np.allclose(np.array(H1), np.array(H2), atol=1e-12)

    def test_jit_different_inputs(self):
        x1 = jnp.array([1.0, 2.0])
        x2 = jnp.array([0.5, -1.0])
        _, _, H1 = hessian_xla_jit(_quad_xla, x1)
        _, _, H2 = hessian_xla_jit(_quad_xla, x2)
        # Both should give same H (quadratic has constant Hessian)
        assert np.allclose(np.array(H1), np.array(H2), atol=1e-12)


# ── XLA vs NumPy backend consistency ─────────────────────────────────────────


class TestXLAvsNumPy:
    def test_matches_numpy_backend(self):
        """XLA Hessian must match the NumPy backend to machine precision."""
        from hypercomplex import hessian

        def f_np(X):
            return X[0] ** 2 + X[0] * X[1] * 3.0 + X[1] ** 2 * 2.0

        x = np.array([1.0, 2.0])
        H_np = hessian(f_np, x, backend="numpy")

        x_jax = jnp.array(x)
        _, _, H_xla = hessian_xla(_quad_xla, x_jax)

        assert np.allclose(H_np, np.array(H_xla), atol=1e-12)

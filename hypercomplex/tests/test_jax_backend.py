"""
hypercomplex/tests/test_jax_backend.py
---------------------------------------
Tests for the JAX backend (backend="jax").

Skipped automatically if JAX is not installed.

Run with:  python -m pytest hypercomplex/tests/test_jax_backend.py -v
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX not installed")
jax.config.update("jax_enable_x64", True)

from hypercomplex import grad, hessian, grad_and_hessian  # noqa: E402
from hypercomplex.core.hyper import Hyper  # noqa: E402
from hypercomplex.backends import get_backend  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────


def f_quad(X):
    return X[0] ** 2 + X[0] * X[1] * 3 + X[1] ** 2 * 2


def f_triple(X):
    return X[0] * X[1] * X[2]


def f_sin_exp(X):
    return X[0].sin() * X[1].exp()


def f_tanh(X):
    return X[0].tanh() + X[1] ** 2


def f_sum_sq(X):
    acc = Hyper.real(X[0].n, 0.0, xp=X[0]._xp)
    for xi in X:
        acc = acc + xi**2
    return acc


# ── backend import ────────────────────────────────────────────────────────────


class TestBackendRegistry:

    def test_get_jax_backend(self):
        xp = get_backend("jax")
        import jax.numpy as jnp

        assert xp is jnp

    def test_get_numpy_backend(self):
        import numpy as np2

        assert get_backend("numpy") is np2

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("torch")


# ── correctness: JAX vs NumPy ────────────────────────────────────────────────


class TestJAXvsNumPy:
    """JAX results must match NumPy results to machine precision."""

    @pytest.mark.parametrize(
        "x,f",
        [
            ([1.0, 2.0], f_quad),
            ([0.5, 1.0], f_sin_exp),
            ([0.3, -0.7], f_tanh),
        ],
    )
    def test_grad_matches_numpy(self, x, f):
        g_np = grad(f, x, backend="numpy")
        g_jx = grad(f, x, backend="jax")
        assert np.allclose(g_np, g_jx, atol=1e-12), f"grad mismatch: numpy={g_np} jax={g_jx}"

    @pytest.mark.parametrize(
        "x,f",
        [
            ([1.0, 2.0], f_quad),
            ([0.5, 1.0], f_sin_exp),
            ([0.3, -0.7], f_tanh),
        ],
    )
    def test_hessian_matches_numpy(self, x, f):
        H_np = hessian(f, x, backend="numpy")
        H_jx = hessian(f, x, backend="jax")
        assert np.allclose(
            H_np, H_jx, atol=1e-12
        ), f"hessian mismatch:\nnumpy=\n{H_np}\njax=\n{H_jx}"

    def test_triple_product_n3(self):
        x = [1.0, 2.0, 3.0]
        g_np = grad(f_triple, x)
        H_np = hessian(f_triple, x)
        g_jx = grad(f_triple, x, backend="jax")
        H_jx = hessian(f_triple, x, backend="jax")
        assert np.allclose(g_np, g_jx, atol=1e-12)
        assert np.allclose(H_np, H_jx, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_sum_of_squares(self, n):
        x = np.arange(1.0, n + 1.0).tolist()
        H_np = hessian(f_sum_sq, x, backend="numpy")
        H_jx = hessian(f_sum_sq, x, backend="jax")
        assert np.allclose(H_np, H_jx, atol=1e-12)
        assert np.allclose(H_jx, 2 * np.eye(n), atol=1e-12)

    def test_grad_and_hessian_simultaneous(self):
        x = [1.0, 2.0]
        g_np, H_np = grad_and_hessian(f_quad, x, backend="numpy")
        g_jx, H_jx = grad_and_hessian(f_quad, x, backend="jax")
        assert np.allclose(g_np, g_jx, atol=1e-12)
        assert np.allclose(H_np, H_jx, atol=1e-12)


# ── analytic correctness ──────────────────────────────────────────────────────


class TestJAXAnalytic:

    def test_quadratic_analytic(self):
        """f = x0^2 + 3*x0*x1 + 2*x1^2, grad=[2x0+3x1, 3x0+4x1], H=[[2,3],[3,4]]"""
        g = grad(f_quad, [1.0, 2.0], backend="jax")
        H = hessian(f_quad, [1.0, 2.0], backend="jax")
        assert np.allclose(g, [8.0, 11.0], atol=1e-12)
        assert np.allclose(H, [[2.0, 3.0], [3.0, 4.0]], atol=1e-12)

    def test_triple_product_analytic(self):
        """f = x0*x1*x2, grad=[x1x2, x0x2, x0x1], H off-diag from permutations"""
        g = grad(f_triple, [1.0, 2.0, 3.0], backend="jax")
        H = hessian(f_triple, [1.0, 2.0, 3.0], backend="jax")
        assert np.allclose(g, [6.0, 3.0, 2.0], atol=1e-12)
        assert np.allclose(H, [[0, 3, 2], [3, 0, 1], [2, 1, 0]], atol=1e-12)


# ── Hyper object with JAX backend ────────────────────────────────────────────


class TestHyperJAX:

    def test_hyper_zero_jax(self):
        import jax.numpy as jnp

        h = Hyper.zero(3, xp=jnp)
        assert h._xp is jnp
        assert float(h.c[0]) == 0.0

    def test_hyper_real_jax(self):
        import jax.numpy as jnp

        h = Hyper.real(3, 5.0, xp=jnp)
        assert float(h.c[0]) == 5.0

    def test_mul_jax_matches_numpy(self):
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        n = 4
        a_np = Hyper(rng.standard_normal(Hyper.size(n)), n)
        b_np = Hyper(rng.standard_normal(Hyper.size(n)), n)
        a_jx = Hyper(jnp.array(a_np.c), n, xp=jnp)
        b_jx = Hyper(jnp.array(b_np.c), n, xp=jnp)
        prod_np = a_np * b_np
        prod_jx = a_jx * b_jx
        assert np.allclose(np.array(prod_jx.c), prod_np.c, atol=1e-13)

    def test_unary_ops_jax(self):
        import jax.numpy as jnp

        for method in ["exp", "tanh", "sin", "cos", "sigmoid", "sqrt", "log"]:
            h_np = Hyper.real(2, 0.7)
            h_jx = Hyper.real(2, 0.7, xp=jnp)
            r_np = getattr(h_np, method)()
            r_jx = getattr(h_jx, method)()
            assert np.allclose(np.array(r_jx.c), r_np.c, atol=1e-12), f"{method} mismatch"


# ── jit=False path ────────────────────────────────────────────────────────────


class TestJITFlag:

    def test_jit_false_same_result(self):
        x = [1.0, 2.0]
        H_jit = hessian(f_quad, x, backend="jax", jit=True)
        H_nojit = hessian(f_quad, x, backend="jax", jit=False)
        H_numpy = hessian(f_quad, x, backend="numpy")
        assert np.allclose(H_jit, H_numpy, atol=1e-12)
        assert np.allclose(H_nojit, H_numpy, atol=1e-12)

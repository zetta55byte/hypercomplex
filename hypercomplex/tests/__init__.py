"""
hypercomplex tests
------------------
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
from hypercomplex.core.hyper import Hyper
from hypercomplex.derivatives import grad, hessian, grad_and_hessian, jacobian
from hypercomplex.curvature import ridge_curvature


# ── Algebra tests ─────────────────────────────────────────────────────────────

class TestHyperAlgebra:

    def test_real_addition(self):
        h = Hyper.real(2, 3.0)
        result = h + 2.0
        assert result.c[0] == 5.0

    def test_multiplication_real(self):
        h = Hyper.real(2, 3.0)
        result = h * 4.0
        assert result.c[0] == 12.0

    def test_i_squared(self):
        """i_j^2 should contribute -1 to real part."""
        n = 2
        h = Hyper.zero(n)
        h.c[h.idx_i(0)] = 1.0
        result = h * h
        assert abs(result.c[0] - (-1.0)) < 1e-14

    def test_eps_squared(self):
        """eps_j^2 should be zero."""
        n = 2
        h = Hyper.zero(n)
        h.c[h.idx_eps(0)] = 1.0
        result = h * h
        assert abs(result.c[0]) < 1e-14
        assert np.all(np.abs(result.c[1:]) < 1e-14)

    def test_i_eps_product(self):
        """i_j * eps_j should land in diag_mix channel."""
        n = 2
        hi = Hyper.zero(n); hi.c[hi.idx_i(0)] = 1.0
        he = Hyper.zero(n); he.c[he.idx_eps(0)] = 1.0
        result = hi * he
        assert abs(result.c[result.idx_diag_mix(0)] - 1.0) < 1e-14

    def test_division_scalar(self):
        n = 2
        h = Hyper.real(n, 4.0)
        result = h / 2.0
        assert abs(result.c[0] - 2.0) < 1e-14

    def test_rtruediv(self):
        n = 2
        h = Hyper.real(n, 4.0)
        result = 8.0 / h
        assert abs(result.c[0] - 2.0) < 1e-14


# ── Gradient tests ────────────────────────────────────────────────────────────

class TestGradient:

    def test_quadratic_gradient(self):
        """f(x) = x0^2 + x1^2, grad = [2x0, 2x1]."""
        def f(X): return X[0]*X[0] + X[1]*X[1]
        x = np.array([1.0, 2.0])
        g = grad(f, x)
        assert np.allclose(g, [2.0, 4.0], atol=1e-12)

    def test_linear_gradient(self):
        """f(x) = 3x0 + 5x1, grad = [3, 5]."""
        def f(X): return X[0]*3.0 + X[1]*5.0
        g = grad(f, [1.0, 1.0])
        assert np.allclose(g, [3.0, 5.0], atol=1e-12)

    def test_cross_term_gradient(self):
        """f(x) = x0*x1, grad = [x1, x0]."""
        def f(X): return X[0]*X[1]
        g = grad(f, [2.0, 3.0])
        assert np.allclose(g, [3.0, 2.0], atol=1e-12)


# ── Hessian tests ─────────────────────────────────────────────────────────────

class TestHessian:

    def test_quadratic_hessian(self):
        """f = x0^2 + 3*x0*x1 + 2*x1^2, H = [[2,3],[3,4]]."""
        def f(X):
            return X[0]*X[0] + X[0]*X[1]*3.0 + X[1]*X[1]*2.0
        H = hessian(f, [1.0, 2.0])
        expected = np.array([[2.0, 3.0], [3.0, 4.0]])
        assert np.allclose(H, expected, atol=1e-12)

    def test_hessian_symmetry(self):
        """Hessian must be symmetric."""
        def f(X):
            return X[0]*X[0]*2 + X[0]*X[1]*3 + X[1]*X[1]*4 + X[0]*5
        H = hessian(f, [1.0, -1.0])
        assert np.allclose(H, H.T, atol=1e-14)

    def test_pure_quadratic(self):
        """f = x0^2, H[0,0]=2, rest zero."""
        def f(X): return X[0]*X[0]
        H = hessian(f, [3.0, 1.0])
        assert abs(H[0, 0] - 2.0) < 1e-12
        assert abs(H[0, 1]) < 1e-12
        assert abs(H[1, 1]) < 1e-12

    def test_hessian_3d(self):
        """Test in 3 dimensions."""
        def f(X):
            return X[0]*X[0] + X[1]*X[1]*2 + X[2]*X[2]*3 + X[0]*X[1]
        H = hessian(f, [1.0, 1.0, 1.0])
        assert abs(H[0, 0] - 2.0) < 1e-12
        assert abs(H[1, 1] - 4.0) < 1e-12
        assert abs(H[2, 2] - 6.0) < 1e-12
        assert abs(H[0, 1] - 1.0) < 1e-12
        assert abs(H[0, 2]) < 1e-12


# ── Curvature tests ───────────────────────────────────────────────────────────

class TestCurvature:

    def test_ridge_curvature_bowl(self):
        """U = -(x0^2 + x1^2), kappa at origin = -2."""
        def U(X): return (X[0]*X[0] + X[1]*X[1]) * (-1.0)
        kappa = ridge_curvature(U, [0.0, 0.0])
        assert abs(kappa - (-2.0)) < 1e-10

    def test_ridge_curvature_anisotropic(self):
        """U = -(2*x0^2 + x1^2), kappa at origin = -2."""
        def U(X): return (X[0]*X[0]*2.0 + X[1]*X[1]) * (-1.0)
        kappa = ridge_curvature(U, [0.0, 0.0])
        assert abs(kappa - (-2.0)) < 1e-10


# ── Jacobian tests ────────────────────────────────────────────────────────────

class TestJacobian:

    def test_linear_jacobian(self):
        """f(x) = [2x0, 3x1], J = [[2,0],[0,3]]."""
        def f(X): return [X[0]*2.0, X[1]*3.0]
        J = jacobian(f, [1.0, 1.0])
        expected = np.array([[2.0, 0.0], [0.0, 3.0]])
        assert np.allclose(J, expected, atol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

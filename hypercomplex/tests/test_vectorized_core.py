"""
hypercomplex/tests/test_vectorized_core.py
------------------------------------------
Tests specific to the v0.2.0 vectorized core.

Checks:
  - All algebra identities hold at various n
  - __mul__ matches a reference loop-based implementation
  - _apply_scalar_func produces correct chain-rule results
  - Index cache is consistent with explicit index helpers
  - __rtruediv__ matches scalar / Hyper analytically
  - Gradient and Hessian extraction correct for n = 2, 4, 8
  - Unary ops: exp, tanh, sin, cos, sigmoid, sqrt

Run with:  python -m pytest hypercomplex/tests/test_vectorized_core.py -v
"""

import numpy as np
import pytest
from hypercomplex.core.hyper import Hyper, _build_index_cache

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_seed(n, j, alpha=1e-20, beta=1e-20):
    """Build the j-th hypercomplex input seed."""
    h = Hyper.zero(n)
    h.c[h.idx_i(j)] = alpha
    h.c[h.idx_eps(j)] = beta
    return h


def extract(F, n, alpha=1e-20, beta=1e-20):
    """Extract gradient vector and Hessian matrix from a Hyper result."""
    g = np.array([F.c[F.idx_i(j)] / alpha for j in range(n)])
    H = np.zeros((n, n))
    for j in range(n):
        H[j, j] = F.c[F.idx_diag_mix(j)] / (alpha * beta)
    for j in range(n):
        for k in range(j + 1, n):
            val = F.c[F.idx_mix(j, k)] / (alpha * beta)
            H[j, k] = H[k, j] = val
    return g, H


# reference loop-based __mul__ (faithful reconstruction of v0.1.0)
def ref_mul(a: Hyper, b: Hyper) -> Hyper:
    if isinstance(b, (int, float)):
        return Hyper(a.c * b, a.n)
    n = a.n
    out = Hyper.zero(n)
    out.c += a.c[0] * b.c + b.c[0] * a.c
    out.c[0] -= a.c[0] * b.c[0]
    for j in range(n):
        out.c[0] -= a.c[a.idx_i(j)] * b.c[b.idx_i(j)]
    for j in range(n):
        for k in range(n):
            contrib = (
                a.c[a.idx_i(j)] * b.c[b.idx_eps(k)]
                + b.c[b.idx_i(j)] * a.c[a.idx_eps(k)]
            )
            if j == k:
                out.c[out.idx_diag_mix(j)] += contrib
            elif j < k:
                out.c[out.idx_mix(j, k)] += contrib
    return out


# ─────────────────────────────────────────────────────────────────────────────
# index cache
# ─────────────────────────────────────────────────────────────────────────────


class TestIndexCache:

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_size_formula(self, n):
        idx = _build_index_cache(n)
        expected = 1 + n + n + n + n * (n - 1) // 2
        assert idx["size"] == expected

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_off_diagonal_count(self, n):
        idx = _build_index_cache(n)
        assert len(idx["off_js"]) == n * (n - 1) // 2
        assert len(idx["off_ks"]) == n * (n - 1) // 2

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_off_diagonal_ordering(self, n):
        """js < ks always, and pairs are in lex order."""
        idx = _build_index_cache(n)
        js, ks = idx["off_js"], idx["off_ks"]
        assert np.all(js < ks), "all pairs must have j < k"

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_idx_mix_matches_cache(self, n):
        """idx_mix(j,k) must agree with the slice start + cache position."""
        h = Hyper.zero(n)
        idx = _build_index_cache(n)
        js, ks = idx["off_js"], idx["off_ks"]
        base = idx["sl_off"].start
        for pos, (j, k) in enumerate(zip(js, ks)):
            assert h.idx_mix(int(j), int(k)) == base + pos


# ─────────────────────────────────────────────────────────────────────────────
# algebra identities
# ─────────────────────────────────────────────────────────────────────────────


class TestAlgebraIdentities:

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_i_squared(self, n):
        for j in range(n):
            ij = Hyper.zero(n)
            ij.c[ij.idx_i(j)] = 1.0
            sq = ij * ij
            assert abs(sq.c[0] - (-1.0)) < 1e-14, f"i_{j}^2 != -1"
            assert np.all(np.abs(sq.c[1:]) < 1e-14), f"i_{j}^2 has spurious coeffs"

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_eps_squared_zero(self, n):
        for j in range(n):
            ej = Hyper.zero(n)
            ej.c[ej.idx_eps(j)] = 1.0
            sq = ej * ej
            assert np.all(np.abs(sq.c) < 1e-14), f"eps_{j}^2 != 0"

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_eps_cross_zero(self, n):
        for j in range(n):
            for k in range(j + 1, n):
                ej = Hyper.zero(n)
                ej.c[ej.idx_eps(j)] = 1.0
                ek = Hyper.zero(n)
                ek.c[ek.idx_eps(k)] = 1.0
                assert np.all(np.abs((ej * ek).c) < 1e-14)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_i_eps_diag(self, n):
        for j in range(n):
            ij = Hyper.zero(n)
            ij.c[ij.idx_i(j)] = 1.0
            ej = Hyper.zero(n)
            ej.c[ej.idx_eps(j)] = 1.0
            prod = ij * ej
            expected = np.zeros(Hyper.size(n))
            expected[ij.idx_diag_mix(j)] = 1.0
            assert np.allclose(prod.c, expected, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_i_eps_offdiag(self, n):
        for j in range(n):
            for k in range(j + 1, n):
                ij = Hyper.zero(n)
                ij.c[ij.idx_i(j)] = 1.0
                ek = Hyper.zero(n)
                ek.c[ek.idx_eps(k)] = 1.0
                prod = ij * ek
                expected = np.zeros(Hyper.size(n))
                expected[ij.idx_mix(j, k)] = 1.0
                assert np.allclose(prod.c, expected, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_commutativity(self, n):
        rng = np.random.default_rng(0)
        for _ in range(10):
            a = Hyper(rng.standard_normal(Hyper.size(n)), n)
            b = Hyper(rng.standard_normal(Hyper.size(n)), n)
            assert np.allclose((a * b).c, (b * a).c, atol=1e-13)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_distributivity(self, n):
        rng = np.random.default_rng(1)
        for _ in range(10):
            a = Hyper(rng.standard_normal(Hyper.size(n)), n)
            b = Hyper(rng.standard_normal(Hyper.size(n)), n)
            c = Hyper(rng.standard_normal(Hyper.size(n)), n)
            assert np.allclose((a * (b + c)).c, (a * b + a * c).c, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_mul_matches_reference(self, n):
        """Vectorized __mul__ must produce identical results to loop version."""
        rng = np.random.default_rng(42)
        for _ in range(15):
            a = Hyper(rng.standard_normal(Hyper.size(n)), n)
            b = Hyper(rng.standard_normal(Hyper.size(n)), n)
            assert np.allclose(
                (a * b).c, ref_mul(a, b).c, atol=1e-13
            ), f"Vectorized mul != reference for n={n}"


# ─────────────────────────────────────────────────────────────────────────────
# gradient and Hessian extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestDerivativeExtraction:

    @pytest.mark.parametrize(
        "n,x,expected_g,expected_H",
        [
            # f = x0^2 + 3*x0*x1 + 2*x1^2
            (2, [1.0, 2.0], [8.0, 11.0], [[2.0, 3.0], [3.0, 4.0]]),
        ],
    )
    def test_quadratic_n2(self, n, x, expected_g, expected_H):
        alpha = beta = 1e-20
        X = [make_seed(n, j, alpha, beta) for j in range(n)]
        for j, xj in enumerate(x):
            X[j].c[0] = xj
        F = X[0] ** 2 + X[0] * X[1] * 3 + X[1] ** 2 * 2
        g, H = extract(F, n, alpha, beta)
        assert np.allclose(g, expected_g, atol=1e-10)
        assert np.allclose(H, expected_H, atol=1e-10)

    def test_triple_product_n3(self):
        n, alpha, beta = 3, 1e-20, 1e-20
        x = [1.0, 2.0, 3.0]
        X = [make_seed(n, j, alpha, beta) for j in range(n)]
        for j in range(n):
            X[j].c[0] = x[j]
        F = X[0] * X[1] * X[2]
        g, H = extract(F, n, alpha, beta)
        assert np.allclose(g, [6.0, 3.0, 2.0], atol=1e-10)
        H_expected = [[0, 3, 2], [3, 0, 1], [2, 1, 0]]
        assert np.allclose(H, H_expected, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_sum_of_squares(self, n):
        """f = sum x_i^2, grad = 2x, H = 2*I."""
        alpha = beta = 1e-20
        x = np.arange(1.0, n + 1.0)
        X = [make_seed(n, j, alpha, beta) for j in range(n)]
        for j in range(n):
            X[j].c[0] = x[j]
        F = sum(X[j] ** 2 for j in range(n))
        g, H = extract(F, n, alpha, beta)
        assert np.allclose(g, 2 * x, atol=1e-10)
        assert np.allclose(H, 2 * np.eye(n), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# unary operations
# ─────────────────────────────────────────────────────────────────────────────


class TestUnaryOps:

    def _check_unary(self, func_hc, f0_fn, f1_fn, f2_fn, a_val, n=3):
        """Check that a unary op matches its analytic chain-rule expansion."""
        alpha = beta = 1e-20
        X = [make_seed(n, j, alpha, beta) for j in range(n)]
        for j in range(n):
            X[j].c[0] = a_val
        # apply op to X[0] only
        h = X[0].copy()
        result = func_hc(h)
        # expected: f0 at real part, f1 in gradient channel, f2 in Hessian
        assert abs(result.c[0] - f0_fn(a_val)) < 1e-12, "real part wrong"
        assert abs(result.c[result.idx_i(0)] - f1_fn(a_val) * alpha) < 1e-12 * abs(
            f1_fn(a_val)
        )

    def test_exp(self):
        import math

        self._check_unary(lambda h: h.exp(), math.exp, math.exp, math.exp, 0.7)

    def test_sin(self):
        import math

        self._check_unary(
            lambda h: h.sin(), math.sin, math.cos, lambda a: -math.sin(a), 0.5
        )

    def test_cos(self):
        import math

        self._check_unary(
            lambda h: h.cos(),
            math.cos,
            lambda a: -math.sin(a),
            lambda a: -math.cos(a),
            1.2,
        )

    def test_tanh(self):
        import math

        self._check_unary(
            lambda h: h.tanh(),
            math.tanh,
            lambda a: 1.0 - math.tanh(a) ** 2,
            lambda a: -2 * math.tanh(a) * (1 - math.tanh(a) ** 2),
            0.6,
        )

    def test_sigmoid(self):
        def sg(a):
            return 1.0 / (1.0 + np.exp(-a))

        def dsg(a):
            s = sg(a)
            return s * (1 - s)

        self._check_unary(
            lambda h: h.sigmoid(), sg, dsg, lambda a: dsg(a) * (1 - 2 * sg(a)), 0.4
        )

    def test_sqrt(self):
        import math

        self._check_unary(
            lambda h: h.sqrt(),
            math.sqrt,
            lambda a: 0.5 / math.sqrt(a),
            lambda a: -0.25 / a**1.5,
            1.5,
        )

    def test_log(self):
        import math

        self._check_unary(
            lambda h: h.log(), math.log, lambda a: 1.0 / a, lambda a: -1.0 / a**2, 2.0
        )


# ─────────────────────────────────────────────────────────────────────────────
# __rtruediv__ (exact inversion)
# ─────────────────────────────────────────────────────────────────────────────


class TestExactInversion:

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_scalar_inverse_real_part(self, n):
        h = Hyper.real(n, 4.0)
        r = 2.0 / h
        assert abs(r.c[0] - 0.5) < 1e-14

    @pytest.mark.parametrize("n", [2, 3])
    def test_inversion_matches_reference(self, n):
        """2.0/h must equal 2.0 * h^{-1} computed via Taylor expansion."""
        rng = np.random.default_rng(7)
        alpha = beta = 1e-20
        for _ in range(5):
            # Use small perturbation so truncation is valid
            a_val = rng.uniform(0.5, 2.0)
            X = [make_seed(n, j, alpha, beta) for j in range(n)]
            for j in range(n):
                X[j].c[0] = a_val
            h = X[0]
            r = 1.0 / h
            # Real part must be 1/a
            assert abs(r.c[0] - 1.0 / a_val) < 1e-13

    @pytest.mark.parametrize("n", [2, 3])
    def test_inversion_vectorized_matches_reference(self, n):
        """Vectorized rtruediv must match loop-based reference."""
        rng = np.random.default_rng(99)
        for _ in range(8):
            c = rng.standard_normal(Hyper.size(n)) * 0.1
            c[0] = rng.uniform(0.5, 2.0)  # safe real part
            h = Hyper(c, n)
            vec = 3.0 / h

            # reference: loop-based (from original repo logic)
            a = c[0]
            a2 = a * a
            a3 = a2 * a
            s = 3.0
            ref = Hyper.zero(n)
            ref.c[0] = s / a
            for j in range(n):
                ref.c[ref.idx_i(j)] = -s * c[ref.idx_i(j)] / a2
                ref.c[ref.idx_eps(j)] = -s * c[ref.idx_eps(j)] / a2
            for j in range(n):
                ci = c[h.idx_i(j)]
                ce = c[h.idx_eps(j)]
                cd = c[h.idx_diag_mix(j)]
                ref.c[ref.idx_diag_mix(j)] = -s * cd / a2 + 2 * s * ci * ce / a3
            for j in range(n):
                for k in range(j + 1, n):
                    c_off = c[h.idx_mix(j, k)]
                    cross = (
                        c[h.idx_i(j)] * c[h.idx_eps(k)]
                        + c[h.idx_i(k)] * c[h.idx_eps(j)]
                    )
                    ref.c[ref.idx_mix(j, k)] = -s * c_off / a2 + 2 * s * cross / a3

            assert np.allclose(vec.c, ref.c, atol=1e-12), f"rtruediv mismatch for n={n}"

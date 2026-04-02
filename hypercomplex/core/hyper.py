"""
hypercomplex.core.hyper
-----------------------
Hypercomplex number type for one-pass exact derivative extraction.

Coefficient layout for dimension n:
  [0]              real
  [1 .. n]         i_j           (imaginary, j=0..n-1)
  [n+1 .. 2n]      eps_j         (nilpotent, j=0..n-1)
  [2n+1 .. 3n]     i_j * eps_j   (diagonal Hessian channels)
  [3n+1 .. end]    i_j * eps_k   (j < k, lex order, off-diagonal Hessian)

Vectorization notes
-------------------
The original __mul__ had two bottlenecks:
  1. idx_mix(j, k) contained a Python double-loop — O(n²) per index lookup.
  2. The i_j * eps_k accumulation used a double Python loop — O(n²) iterations.

This version precomputes all index arrays once at init (or class-level cache),
then __mul__ is pure NumPy slice/outer-product operations — no Python loops
in the hot path.

Off-diagonal block:
  The n*(n-1)/2 pairs (j<k) in lex order are precomputed as two int arrays
  _js, _ks of that length, stored on the instance.  The outer-product matrix

      P[j, k] = a_i[j] * b_eps[k] + b_i[j] * a_eps[k]

  is computed with a single np.outer call, then the upper-triangle entries
  are gathered with P[_js, _ks].  One line, one BLAS call.
"""

import numpy as np
from functools import lru_cache


# ---------------------------------------------------------------------------
# Index-layout helpers (pure functions, cached by n)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _build_index_cache(n):
    """
    Precompute all index arrays for dimension n.

    Returns a dict with:
      sl_i     : slice  into coeff vector for i_j   block  (length n)
      sl_eps   : slice  into coeff vector for eps_j  block  (length n)
      sl_diag  : slice  into coeff vector for i_j*eps_j block (length n)
      sl_off   : slice  into coeff vector for off-diag block (length n*(n-1)//2)
      off_js   : int array, row indices j for upper-triangle pairs
      off_ks   : int array, col indices k for upper-triangle pairs
      size     : total coefficient vector length
    """
    n_off = n * (n - 1) // 2
    size  = 1 + n + n + n + n_off

    sl_i    = slice(1,         1 + n)
    sl_eps  = slice(1 + n,     1 + 2*n)
    sl_diag = slice(1 + 2*n,   1 + 3*n)
    sl_off  = slice(1 + 3*n,   size)

    js, ks = np.triu_indices(n, k=1)   # upper triangle, j < k, lex order
    js = js.astype(np.intp)
    ks = ks.astype(np.intp)

    return dict(
        sl_i=sl_i, sl_eps=sl_eps, sl_diag=sl_diag, sl_off=sl_off,
        off_js=js, off_ks=ks, size=size,
    )


# ---------------------------------------------------------------------------
# Hyper class
# ---------------------------------------------------------------------------

class Hyper:
    """
    Hypercomplex number supporting exact gradient and Hessian propagation.

    Parameters
    ----------
    coeffs : array-like
        Coefficient vector of length Hyper.size(n).
    n : int
        Dimension of the input space.

    Notes
    -----
    Implements the commutative algebra:
        i_j^2   = -1
        eps_j^2 =  0
        eps_j * eps_k = 0  (j ≠ k)
    All units commute.

    __mul__ is fully vectorized — no Python loops in the hot path.
    """

    __slots__ = ('c', 'n', '_idx')

    def __init__(self, coeffs, n):
        self.c   = np.asarray(coeffs, dtype=float)
        self.n   = n
        self._idx = _build_index_cache(n)   # O(1) dict lookup after first call

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def zero(cls, n):
        idx  = _build_index_cache(n)
        h    = object.__new__(cls)
        h.c  = np.zeros(idx['size'])
        h.n  = n
        h._idx = idx
        return h

    @classmethod
    def real(cls, n, value):
        h = cls.zero(n)
        h.c[0] = value
        return h

    # ── index helpers (kept for backwards compatibility) ───────────────────────
    # These are now O(1) array lookups rather than Python loops.

    def idx_i(self, j):
        return 1 + j

    def idx_eps(self, j):
        return 1 + self.n + j

    def idx_diag_mix(self, j):
        return 1 + self.n + self.n + j

    def idx_mix(self, j, k):
        """Index of the i_j * eps_k coefficient (j < k)."""
        # O(1): position in the upper-triangle enumeration
        idx  = self._idx
        js, ks = idx['off_js'], idx['off_ks']
        # argwhere equivalent, but these arrays are tiny (n*(n-1)/2 entries)
        matches = np.where((js == j) & (ks == k))[0]
        if matches.size == 0:
            raise IndexError(f"Invalid mix index ({j}, {k}) for n={self.n}")
        return int(idx['sl_off'].start + matches[0])

    # ── size helper ────────────────────────────────────────────────────────────

    @staticmethod
    def size(n):
        return _build_index_cache(n)['size']

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __neg__(self):
        return Hyper(-self.c, self.n)

    def __add__(self, other):
        if isinstance(other, (int, float, np.floating)):
            out = self.copy()
            out.c[0] += other
            return out
        return Hyper(self.c + other.c, self.n)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, np.floating)):
            out = self.copy()
            out.c[0] -= other
            return out
        return Hyper(self.c - other.c, self.n)

    def __rsub__(self, other):
        return Hyper(other - self.c, self.n) if isinstance(other, (int, float, np.floating)) else NotImplemented

    def __mul__(self, other):
        # ── scalar fast-path ──────────────────────────────────────────────────
        if isinstance(other, (int, float, np.floating)):
            return Hyper(self.c * other, self.n)

        n   = self.n
        idx = self._idx
        a, b = self.c, other.c

        sl_i    = idx['sl_i']
        sl_eps  = idx['sl_eps']
        sl_diag = idx['sl_diag']
        sl_off  = idx['sl_off']
        js      = idx['off_js']
        ks      = idx['off_ks']

        out = np.empty_like(a)

        # ── real * anything (both ways), subtract double-counted real*real ──
        # This replicates:  out = a[0]*b + b[0]*a - a[0]*b[0]*e_0
        out[:] = a[0] * b + b[0] * a
        out[0] -= a[0] * b[0]

        # ── i_j * i_j  →  -real  (vectorized over j) ──────────────────────
        # sum of a_i[j] * b_i[j] for all j, subtracted from out[0]
        out[0] -= np.dot(a[sl_i], b[sl_i])

        # ── i_j * eps_k  →  mix channels ─────────────────────────────────
        # P[j, k] = a_i[j]*b_eps[k] + b_i[j]*a_eps[k]
        # One outer product per direction; gather with precomputed index arrays.
        a_i   = a[sl_i]    # shape (n,)
        b_i   = b[sl_i]
        a_eps = a[sl_eps]  # shape (n,)
        b_eps = b[sl_eps]

        P = np.outer(a_i, b_eps) + np.outer(b_i, a_eps)   # shape (n, n)

        # diagonal j==k → sl_diag block
        out[sl_diag] += np.diag(P)

        # upper triangle j<k → sl_off block (lex order matching triu_indices)
        out[sl_off] += P[js, ks]

        return Hyper(out, n)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Hyper(self.c / other, self.n)
        raise NotImplementedError("Hyper / Hyper not implemented")

    def __rtruediv__(self, other):
        """
        scalar / Hyper  via exact second-order inversion.

        For h = a + delta where a is the real part:
            1/h = 1/a - delta/a^2 + (cross terms)/a^3

        Vectorized: computes all n gradient channels and all Hessian
        channels without Python loops.
        """
        if not isinstance(other, (int, float, np.floating)):
            return NotImplemented

        a_val = self.c[0]
        if abs(a_val) < 1e-300:
            raise ZeroDivisionError("Hyper real part is zero")

        n   = self.n
        idx = self._idx
        sl_i    = idx['sl_i']
        sl_eps  = idx['sl_eps']
        sl_diag = idx['sl_diag']
        sl_off  = idx['sl_off']
        js      = idx['off_js']
        ks      = idx['off_ks']

        s   = float(other)
        a2  = a_val * a_val
        a3  = a2 * a_val

        c   = self.c
        out = np.zeros_like(c)

        # real part
        out[0] = s / a_val

        # gradient channels: -s * c_i / a²  and  -s * c_eps / a²
        out[sl_i]   = -s * c[sl_i]   / a2
        out[sl_eps] = -s * c[sl_eps] / a2

        # diagonal Hessian channel j:
        #   -s*c_diag[j]/a² + 2s*c_i[j]*c_eps[j]/a³
        ci   = c[sl_i]
        ceps = c[sl_eps]
        cd   = c[sl_diag]

        out[sl_diag] = -s * cd / a2 + 2.0 * s * ci * ceps / a3

        # off-diagonal Hessian channel (j, k):
        #   -s*c_off[p]/a² + 2s*(c_i[j]*c_eps[k] + c_i[k]*c_eps[j])/a³
        c_off = c[sl_off]
        cross = ci[js] * ceps[ks] + ci[ks] * ceps[js]
        out[sl_off] = -s * c_off / a2 + 2.0 * s * cross / a3

        return Hyper(out, n)

    def __pow__(self, power):
        """Integer powers via repeated squaring (exact, no loops for small p)."""
        if not isinstance(power, int) or power < 0:
            raise NotImplementedError("Only non-negative integer powers supported")
        if power == 0:
            return Hyper.real(self.n, 1.0)
        if power == 1:
            return self.copy()
        # binary exponentiation
        result = Hyper.real(self.n, 1.0)
        base   = self.copy()
        exp    = power
        while exp:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        return result

    # ── copy / repr ───────────────────────────────────────────────────────────

    def copy(self):
        return Hyper(self.c.copy(), self.n)

    def __repr__(self):
        return f"Hyper(real={self.c[0]:.6g}, n={self.n})"

    # ── mathematical functions ─────────────────────────────────────────────────
    # All follow the same pattern:
    #   f(a + δ) ≈ f(a) + f'(a)δ + ½f''(a)δ²
    # where δ is the perturbation encoded in the non-real coefficients.
    # Because eps² = 0 and i²= -1, the second-order algebra truncates exactly.
    #
    # Vectorized: compute f(a), f'(a), f''(a) once, then scale coefficient
    # blocks — no Python loops.

    def _apply_scalar_func(self, f0, f1, f2):
        """
        Return f(self) where f0=f(a), f1=f'(a), f2=f''(a) at real part a.

        Layout (all vectorized):
          real     → f0
          i_j      → f1 * c_i[j]
          eps_j    → f1 * c_eps[j]
          diag_mix → f1 * c_diag[j] + f2 * c_i[j] * c_eps[j]
          off_mix  → f1 * c_off[p]  + f2 * (c_i[j]*c_eps[k] + c_i[k]*c_eps[j])
        """
        idx     = self._idx
        sl_i    = idx['sl_i']
        sl_eps  = idx['sl_eps']
        sl_diag = idx['sl_diag']
        sl_off  = idx['sl_off']
        js      = idx['off_js']
        ks      = idx['off_ks']
        c       = self.c

        out = np.empty_like(c)
        out[0]       = f0
        out[sl_i]    = f1 * c[sl_i]
        out[sl_eps]  = f1 * c[sl_eps]

        ci   = c[sl_i]
        ceps = c[sl_eps]

        out[sl_diag] = f1 * c[sl_diag] + f2 * ci * ceps
        out[sl_off]  = (f1 * c[sl_off]
                        + f2 * (ci[js] * ceps[ks] + ci[ks] * ceps[js]))
        return Hyper(out, self.n)

    def exp(self):
        a  = self.c[0]
        ea = np.exp(a)
        return self._apply_scalar_func(ea, ea, ea)

    def log(self):
        a = self.c[0]
        return self._apply_scalar_func(np.log(a), 1.0/a, -1.0/a**2)

    def sin(self):
        a = self.c[0]
        return self._apply_scalar_func(np.sin(a), np.cos(a), -np.sin(a))

    def cos(self):
        a = self.c[0]
        return self._apply_scalar_func(np.cos(a), -np.sin(a), -np.cos(a))

    def tanh(self):
        a  = self.c[0]
        t  = np.tanh(a)
        s  = 1.0 - t*t          # sech²(a)  = f'
        s2 = -2.0 * t * s       # -2 tanh sech² = f''
        return self._apply_scalar_func(t, s, s2)

    def sigmoid(self):
        a  = self.c[0]
        sg = 1.0 / (1.0 + np.exp(-a))
        f1 = sg * (1.0 - sg)
        f2 = f1 * (1.0 - 2.0 * sg)
        return self._apply_scalar_func(sg, f1, f2)

    def sqrt(self):
        a  = self.c[0]
        sq = np.sqrt(a)
        return self._apply_scalar_func(sq, 0.5/sq, -0.25/a**1.5)

    def abs(self):
        a = self.c[0]
        if a == 0.0:
            raise ZeroDivisionError("|Hyper| undefined at real part zero")
        sgn = np.sign(a)
        return self._apply_scalar_func(abs(a), sgn, 0.0)

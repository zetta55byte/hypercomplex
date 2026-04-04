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

Backend support
---------------
Hyper stores an array-module reference (``xp``) alongside its coefficients.
By default ``xp = numpy``.  Passing ``xp = jax.numpy`` routes all coefficient
arithmetic through JAX, enabling XLA compilation of the full algebra.

The index cache (_build_index_cache) always uses NumPy because it produces
integer index arrays — these are never differentiated through and JAX would
gain nothing from tracing them.
"""

import numpy as np
from functools import lru_cache

# ---------------------------------------------------------------------------
# Index-layout helpers (pure NumPy, cached by n)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _build_index_cache(n):
    """
    Precompute all index arrays for dimension n.

    Always uses NumPy regardless of the active backend — index arrays
    are integer constants that never enter the differentiable computation.

    Returns
    -------
    dict with sl_i, sl_eps, sl_diag, sl_off, off_js, off_ks, size.
    """
    n_off = n * (n - 1) // 2
    size = 1 + n + n + n + n_off

    sl_i = slice(1, 1 + n)
    sl_eps = slice(1 + n, 1 + 2 * n)
    sl_diag = slice(1 + 2 * n, 1 + 3 * n)
    sl_off = slice(1 + 3 * n, size)

    js, ks = np.triu_indices(n, k=1)
    js = js.astype(np.intp)
    ks = ks.astype(np.intp)

    return dict(
        sl_i=sl_i,
        sl_eps=sl_eps,
        sl_diag=sl_diag,
        sl_off=sl_off,
        off_js=js,
        off_ks=ks,
        size=size,
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
        Coefficient vector of length ``Hyper.size(n)``.
    n : int
        Dimension of the input space.
    xp : module, optional
        Array module to use for coefficient arithmetic.
        Default: ``numpy``.  Pass ``jax.numpy`` for the JAX backend.

    Notes
    -----
    Algebra rules:
        i_j^2   = -1,   eps_j^2 = 0,   eps_j * eps_k = 0 (j ≠ k)
    All units commute.  ``__mul__`` has no Python loops in the hot path.
    """

    __slots__ = ("c", "n", "_idx", "_xp")

    def __init__(self, coeffs, n, xp=None):
        if xp is None:
            xp = np
        self._xp = xp
        self.n = n
        self._idx = _build_index_cache(n)
        self.c = xp.asarray(coeffs, dtype=float)

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def zero(cls, n, xp=None):
        if xp is None:
            xp = np
        idx = _build_index_cache(n)
        h = object.__new__(cls)
        h._xp = xp
        h.n = n
        h._idx = idx
        h.c = xp.zeros(idx["size"])
        return h

    @classmethod
    def real(cls, n, value, xp=None):
        h = cls.zero(n, xp=xp)
        # Use numpy indexing — JAX arrays support in-place via .at[].set
        if h._xp is np:
            h.c[0] = value
        else:
            h.c = h.c.at[0].set(value)
        return h

    # ── index helpers (O(1), backwards compatible) ────────────────────────────

    def idx_i(self, j):
        return 1 + j

    def idx_eps(self, j):
        return 1 + self.n + j

    def idx_diag_mix(self, j):
        return 1 + self.n + self.n + j

    def idx_mix(self, j, k):
        idx = self._idx
        js, ks = idx["off_js"], idx["off_ks"]
        matches = np.where((js == j) & (ks == k))[0]
        if matches.size == 0:
            raise IndexError(f"Invalid mix index ({j}, {k}) for n={self.n}")
        return int(idx["sl_off"].start + matches[0])

    @staticmethod
    def size(n):
        return _build_index_cache(n)["size"]

    # ── internal helpers ──────────────────────────────────────────────────────

    def _new(self, coeffs):
        """Construct a new Hyper with the same n and xp as self."""
        return Hyper(coeffs, self.n, xp=self._xp)

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __neg__(self):
        return self._new(-self.c)

    def __add__(self, other):
        if isinstance(other, (int, float, np.floating)):
            xp = self._xp
            if xp is np:
                out = self.copy()
                out.c[0] += other
                return out
            else:
                return self._new(self.c.at[0].add(other))
        return self._new(self.c + other.c)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, np.floating)):
            xp = self._xp
            if xp is np:
                out = self.copy()
                out.c[0] -= other
                return out
            else:
                return self._new(self.c.at[0].add(-other))
        return self._new(self.c - other.c)

    def __rsub__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return self._new(other - self.c)
        return NotImplemented

    def __mul__(self, other):
        xp = self._xp

        # ── scalar fast-path ──────────────────────────────────────────────────
        if isinstance(other, (int, float, np.floating)):
            return self._new(self.c * other)

        idx = self._idx
        a, b = self.c, other.c

        sl_i = idx["sl_i"]
        sl_eps = idx["sl_eps"]
        sl_diag = idx["sl_diag"]
        sl_off = idx["sl_off"]
        js = idx["off_js"]
        ks = idx["off_ks"]

        # real * anything (both ways), subtract double-counted real*real
        out = (
            a[0] * b + b[0] * a - a[0] * b[0] * xp.zeros_like(a).at[0].set(1.0)
            if xp is not np
            else (
                (
                    lambda o: (
                        o.__setitem__(slice(None), a[0] * b + b[0] * a),
                        o.__setitem__(0, o[0] - a[0] * b[0]),
                        o,
                    )[-1]
                )(np.empty_like(a))
            )
        )

        # ── cleaner implementation for both backends ──────────────────────────
        if xp is np:
            out = np.empty_like(a)
            out[:] = a[0] * b + b[0] * a
            out[0] -= a[0] * b[0]
            out[0] -= np.dot(a[sl_i], b[sl_i])
            a_i, b_i = a[sl_i], b[sl_i]
            a_eps, b_eps = a[sl_eps], b[sl_eps]
            P = np.outer(a_i, b_eps) + np.outer(b_i, a_eps)
            out[sl_diag] += np.diag(P)
            out[sl_off] += P[js, ks]
        else:
            # JAX: functional style using .at[].add()
            out = a[0] * b + b[0] * a
            out = out.at[0].add(-a[0] * b[0])
            out = out.at[0].add(-xp.dot(a[sl_i], b[sl_i]))
            a_i, b_i = a[sl_i], b[sl_i]
            a_eps, b_eps = a[sl_eps], b[sl_eps]
            P = xp.outer(a_i, b_eps) + xp.outer(b_i, a_eps)
            out = out.at[sl_diag].add(xp.diag(P))
            out = out.at[sl_off].add(P[js, ks])

        return self._new(out)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return self._new(self.c / other)
        raise NotImplementedError("Hyper / Hyper not implemented")

    def __rtruediv__(self, other):
        """scalar / Hyper via exact second-order inversion."""
        if not isinstance(other, (int, float, np.floating)):
            return NotImplemented

        a_val = float(self.c[0])
        if abs(a_val) < 1e-300:
            raise ZeroDivisionError("Hyper real part is zero")

        xp = self._xp
        idx = self._idx
        sl_i, sl_eps = idx["sl_i"], idx["sl_eps"]
        sl_diag, sl_off = idx["sl_diag"], idx["sl_off"]
        js, ks = idx["off_js"], idx["off_ks"]

        s, a2, a3 = float(other), a_val**2, a_val**3
        c = self.c

        if xp is np:
            out = np.zeros_like(c)
            out[0] = s / a_val
            out[sl_i] = -s * c[sl_i] / a2
            out[sl_eps] = -s * c[sl_eps] / a2
            ci, ceps = c[sl_i], c[sl_eps]
            out[sl_diag] = -s * c[sl_diag] / a2 + 2.0 * s * ci * ceps / a3
            cross = ci[js] * ceps[ks] + ci[ks] * ceps[js]
            out[sl_off] = -s * c[sl_off] / a2 + 2.0 * s * cross / a3
        else:
            out = xp.zeros_like(c)
            out = out.at[0].set(s / a_val)
            out = out.at[sl_i].set(-s * c[sl_i] / a2)
            out = out.at[sl_eps].set(-s * c[sl_eps] / a2)
            ci, ceps = c[sl_i], c[sl_eps]
            out = out.at[sl_diag].set(-s * c[sl_diag] / a2 + 2.0 * s * ci * ceps / a3)
            cross = ci[js] * ceps[ks] + ci[ks] * ceps[js]
            out = out.at[sl_off].set(-s * c[sl_off] / a2 + 2.0 * s * cross / a3)

        return self._new(out)

    def __pow__(self, power):
        """Integer powers via binary exponentiation."""
        if not isinstance(power, int) or power < 0:
            raise NotImplementedError("Only non-negative integer powers supported")
        if power == 0:
            return Hyper.real(self.n, 1.0, xp=self._xp)
        if power == 1:
            return self.copy()
        result = Hyper.real(self.n, 1.0, xp=self._xp)
        base, exp = self.copy(), power
        while exp:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        return result

    # ── copy / repr ───────────────────────────────────────────────────────────

    def copy(self):
        xp = self._xp
        c = xp.array(self.c) if xp is not np else self.c.copy()
        return Hyper(c, self.n, xp=xp)

    def __repr__(self):
        return f"Hyper(real={float(self.c[0]):.6g}, n={self.n})"

    # ── unary mathematical functions ──────────────────────────────────────────

    def _apply_scalar_func(self, f0, f1, f2):
        """
        Apply a scalar function chain-rule expansion.

        f(a + δ) = f(a) + f'(a)δ + ½f''(a)δ²  (exact in this algebra)

        Parameters
        ----------
        f0, f1, f2 : float
            f(a), f'(a), f''(a) at the real part a = self.c[0].
        """
        xp = self._xp
        idx = self._idx
        sl_i, sl_eps = idx["sl_i"], idx["sl_eps"]
        sl_diag, sl_off = idx["sl_diag"], idx["sl_off"]
        js, ks = idx["off_js"], idx["off_ks"]
        c = self.c

        if xp is np:
            out = np.empty_like(c)
            out[0] = f0
            out[sl_i] = f1 * c[sl_i]
            out[sl_eps] = f1 * c[sl_eps]
            ci, ceps = c[sl_i], c[sl_eps]
            out[sl_diag] = f1 * c[sl_diag] + f2 * ci * ceps
            out[sl_off] = f1 * c[sl_off] + f2 * (ci[js] * ceps[ks] + ci[ks] * ceps[js])
        else:
            out = xp.zeros_like(c)
            out = out.at[0].set(f0)
            out = out.at[sl_i].set(f1 * c[sl_i])
            out = out.at[sl_eps].set(f1 * c[sl_eps])
            ci, ceps = c[sl_i], c[sl_eps]
            out = out.at[sl_diag].set(f1 * c[sl_diag] + f2 * ci * ceps)
            out = out.at[sl_off].set(f1 * c[sl_off] + f2 * (ci[js] * ceps[ks] + ci[ks] * ceps[js]))

        return self._new(out)

    def exp(self):
        xp = self._xp
        a = float(self.c[0])
        ea = float(xp.exp(xp.array(a)))
        return self._apply_scalar_func(ea, ea, ea)

    def log(self):
        a = float(self.c[0])
        return self._apply_scalar_func(float(np.log(a)), 1.0 / a, -1.0 / a**2)

    def sin(self):
        a = float(self.c[0])
        return self._apply_scalar_func(float(np.sin(a)), float(np.cos(a)), -float(np.sin(a)))

    def cos(self):
        a = float(self.c[0])
        return self._apply_scalar_func(float(np.cos(a)), -float(np.sin(a)), -float(np.cos(a)))

    def tanh(self):
        a = float(self.c[0])
        t = float(np.tanh(a))
        s = 1.0 - t * t
        s2 = -2.0 * t * s
        return self._apply_scalar_func(t, s, s2)

    def sigmoid(self):
        a = float(self.c[0])
        sg = 1.0 / (1.0 + np.exp(-a))
        f1 = sg * (1.0 - sg)
        f2 = f1 * (1.0 - 2.0 * sg)
        return self._apply_scalar_func(sg, f1, f2)

    def sqrt(self):
        a = float(self.c[0])
        return self._apply_scalar_func(float(np.sqrt(a)), 0.5 / np.sqrt(a), -0.25 / a**1.5)

    def abs(self):
        a = float(self.c[0])
        if a == 0.0:
            raise ZeroDivisionError("|Hyper| undefined at real part zero")
        return self._apply_scalar_func(abs(a), float(np.sign(a)), 0.0)

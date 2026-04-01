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
"""

import numpy as np


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
    Implements the commutative algebra with i_j^2 = -1, eps_j^2 = 0,
    eps_j * eps_k = 0, and all units commuting.
    """

    def __init__(self, coeffs, n):
        self.c = np.asarray(coeffs, dtype=float)
        self.n = n

    # ── constructors ──────────────────────────────────────────────────────────

    @staticmethod
    def size(n):
        """Total number of coefficients for dimension n."""
        return 1 + n + n + n + n * (n - 1) // 2

    @staticmethod
    def zero(n):
        """Zero hypercomplex number in dimension n."""
        return Hyper(np.zeros(Hyper.size(n)), n)

    @staticmethod
    def real(n, value):
        """Hypercomplex number with only a real component."""
        h = Hyper.zero(n)
        h.c[0] = value
        return h

    # ── index helpers ─────────────────────────────────────────────────────────

    def idx_i(self, j):
        """Index of the i_j coefficient."""
        return 1 + j

    def idx_eps(self, j):
        """Index of the eps_j coefficient."""
        return 1 + self.n + j

    def idx_diag_mix(self, j):
        """Index of the i_j * eps_j coefficient."""
        return 1 + self.n + self.n + j

    def idx_mix(self, j, k):
        """Index of the i_j * eps_k coefficient (j < k)."""
        base = 1 + self.n + self.n + self.n
        offset = 0
        for jj in range(self.n):
            for kk in range(jj + 1, self.n):
                if jj == j and kk == k:
                    return base + offset
                offset += 1
        raise IndexError(f"Invalid mix index ({j}, {k}) for n={self.n}")

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, (int, float, np.floating)):
            out = self.copy()
            out.c[0] += other
            return out
        return Hyper(self.c + other.c, self.n)

    __radd__ = __add__

    def __neg__(self):
        return Hyper(-self.c, self.n)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Hyper(self.c * other, self.n)
        n = self.n
        out = Hyper.zero(n)

        # real * anything (both directions, subtract double-count)
        out.c += self.c[0] * other.c
        out.c += other.c[0] * self.c
        out.c[0] -= self.c[0] * other.c[0]

        # i_j * i_j -> -1
        for j in range(n):
            out.c[0] -= self.c[self.idx_i(j)] * other.c[self.idx_i(j)]

        # i_j * eps_k -> diagonal or off-diagonal mix channel
        for j in range(n):
            for k in range(n):
                contrib = (
                    self.c[self.idx_i(j)] * other.c[self.idx_eps(k)]
                    + other.c[self.idx_i(j)] * self.c[self.idx_eps(k)]
                )
                if j == k:
                    out.c[self.idx_diag_mix(j)] += contrib
                elif j < k:
                    out.c[self.idx_mix(j, k)] += contrib

        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Hyper(self.c / other, self.n)
        raise NotImplementedError("Hyper / Hyper not implemented")

    def __rtruediv__(self, other):
        """
        scalar / Hyper via exact second-order inversion.

        For h = a + delta where a is the real part:
        1/h = 1/a - delta/a^2 + (cross terms)/a^3
        """
        if isinstance(other, (int, float, np.floating)):
            a = self.c[0]
            if abs(a) < 1e-300:
                raise ZeroDivisionError("Hyper real part is zero")
            n = self.n
            out = Hyper.zero(n)
            a2 = a * a
            a3 = a2 * a
            out.c[0] = other / a
            for j in range(n):
                out.c[self.idx_i(j)]   = -other * self.c[self.idx_i(j)]   / a2
                out.c[self.idx_eps(j)] = -other * self.c[self.idx_eps(j)] / a2
            for j in range(n):
                ci = self.c[self.idx_i(j)]
                ce = self.c[self.idx_eps(j)]
                cd = self.c[self.idx_diag_mix(j)]
                out.c[self.idx_diag_mix(j)] = (
                    -other * cd / a2 + 2.0 * other * ci * ce / a3
                )
            for j in range(n):
                for k in range(j + 1, n):
                    ci_j = self.c[self.idx_i(j)]
                    ce_k = self.c[self.idx_eps(k)]
                    ci_k = self.c[self.idx_i(k)]
                    ce_j = self.c[self.idx_eps(j)]
                    cm   = self.c[self.idx_mix(j, k)]
                    out.c[self.idx_mix(j, k)] = (
                        -other * cm / a2
                        + other * (ci_j * ce_k + ci_k * ce_j) / a3
                    )
            return out
        raise NotImplementedError

    def __pow__(self, exp):
        if not isinstance(exp, (int, np.integer)) or exp < 0:
            raise NotImplementedError("Only non-negative integer powers supported")
        result = Hyper.real(self.n, 1.0)
        for _ in range(exp):
            result = result * self
        return result

    # ── utilities ─────────────────────────────────────────────────────────────

    def copy(self):
        return Hyper(self.c.copy(), self.n)

    @property
    def real_part(self):
        return self.c[0]

    def __repr__(self):
        return f"Hyper(real={self.c[0]:.6g}, n={self.n})"

# Sections to insert into the scaling paper
# Drop these after the Implementation section and before the Conclusion.

---

## Related Work

Second-order information is computationally expensive, and a substantial literature
addresses this cost through approximation. Becker and LeCun (1989) proposed computing
only the diagonal of the Hessian at gradient-comparable cost by propagating squared
gradient terms forward through the network. Elsayed et al. (2024) revisit and refine
this scheme as HesScale, demonstrating improvements over the original approximation
in both accuracy and applicability to reinforcement learning. These methods are
explicitly approximate by design: they trade the off-diagonal Hessian structure for
scalability to the parameter counts of deep networks (d ~ 10^6 to 10^9).

The complex-step method (Lyness and Moler 1967; Squire and Trapp 1998) achieves
machine-precision first derivatives for scalar functions by perturbing the input into
the complex plane. It avoids the cancellation error that limits finite differences but
does not extend to second-order information without additional structure.

Algorithmic differentiation (Griewank and Walther 2008) provides exact derivatives
of arbitrary order by augmenting the computational graph. Reverse-mode AD (backprop)
computes the full gradient in O(1) passes but requires O(n) reverse passes for the
full Hessian, or a forward-over-reverse strategy whose constant is implementation-
dependent. JAX's jax.hessian is the practical reference implementation; it compiles
to XLA and is highly competitive for differentiable programs expressible as JAX
primitives.

Hypercomplex perturbation occupies a complementary niche. For scalar-valued functions
of moderate dimension (n ~ 2 to 100), it embeds the entire first- and second-order
Taylor structure into a single forward evaluation by augmenting each input with a
pair of commuting infinitesimal units whose algebraic relations isolate the gradient
and Hessian in distinct coefficient channels. This gives exact full Hessians —
not diagonal approximations — at the cost of one function evaluation in an augmented
algebra. The tradeoff is the opposite of HesScale: full accuracy, bounded dimension.

For implicit models z* = f(z*, θ), the relevant second-order quantity is the Hessian
of the loss at the fixed point, computed via the implicit function theorem. Unrolled
AD differentiates through the iteration history rather than through the fixed-point
equation, producing a different (and iteration-depth-dependent) result. The
hypercomplex approach propagates perturbations through the converged fixed point
directly, recovering the IFT-correct Hessian without unrolling.


## Limitations and Scope

The method is exact for scalar-valued functions expressible in hypercomplex arithmetic.
Several practical constraints bound its applicability:

**Scalar output.** The algebra encodes the Hessian of a scalar f: R^n -> R. Vector-
valued functions require one hypercomplex pass per output dimension, recovering the
Jacobian row by row. This is equivalent in cost to forward-mode AD applied per output.

**Moderate n.** The coefficient vector has length 1 + 3n + n(n-1)/2, growing
quadratically in n. For n = 100 this is 5151 coefficients; for n = 1000 it is ~500k.
Memory and compute scale as O(n^2) per multiply, making the method uncompetitive with
diagonal approximations for the weight-space Hessians of deep networks (n ~ 10^6+).
The practical sweet spot is n ~ 2 to 100, covering low-dimensional physical systems,
small neural layers, implicit fixed-point models, and parameter-sensitivity analysis.

**Function expressibility.** The function must be implemented in hypercomplex
arithmetic. Standard NumPy ufuncs (np.sin, np.exp) do not operate on Hyper objects
directly; the user must use the provided Hyper methods (.sin(), .exp(), etc.) or
compose from supported primitives. This is analogous to the requirement in JAX that
functions be written in jnp rather than np.

**Not a replacement for diagonal approximations.** HesScale, BL89, and related methods
are designed for the deep-network regime where the full Hessian is computationally
inaccessible. hcderiv is not competitive in that regime. The methods address different
parts of the cost-accuracy tradeoff space and are best understood as complementary.

**No JIT compilation in the NumPy backend.** The current implementation uses NumPy
and incurs Python dispatch overhead per operation. A JAX backend (swapping np for jnp)
would allow XLA compilation and is a natural next step; the algebra is unchanged.


## Future Work

Several directions extend the current work naturally:

**JAX/XLA backend.** The hypercomplex algebra requires only array operations that
map directly to jnp. A jit-compiled JAX backend would eliminate Python dispatch
overhead and bring the method to parity with compiled AD for moderate n. Preliminary
experiments suggest this would recover another 10-50x over the current NumPy baseline.

**Trust-region and curvature-aware optimizers.** Exact Hessians are the natural input
to classical second-order optimizers (Newton's method, L-BFGS warm-start, trust-region
methods). A worked example connecting hcderiv to a trust-region step for a small
nonlinear system would demonstrate the practical value of exact vs approximate
curvature in an optimization context.

**Differentiable physics.** Low-dimensional physical systems (pendulum, spring-mass,
rigid body) have parameter spaces in the n = 3 to 30 range and benefit from exact
second-order sensitivity analysis. The ridge potential U(x) = -||f(x)||^2 studied in
Byte (2026b) is one instance; the method generalizes to any scalar energy or loss
defined over a physical state space.

**Implicit layers at scale.** The IFT-correct Hessian demonstrated here for n = 3 to 8
generalizes to larger implicit layers. Connecting hcderiv to deep equilibrium models
(Bai et al. 2019) or neural ODEs (Chen et al. 2018) — where the fixed-point dimension
is the hidden state size — would establish the method's relevance to the implicit
differentiation literature.

**Higher-order extensions.** The two-unit algebra (i_j, eps_j) recovers first and
second derivatives exactly. A three-unit extension would add third-order terms,
enabling exact Hessian-of-Hessian computations relevant to meta-learning and
hyperparameter sensitivity. The algebraic structure generalizes straightforwardly;
the coefficient count grows as O(n^3).

**Micro-benchmark suite.** The current benchmarks cover quadratic and Rosenbrock
functions at n = 2 to 64. A systematic suite covering n = 5 to 100 across function
classes (polynomial, transcendental, rational, implicit) with comparisons to JAX,
finite differences, and PyTorch autograd would provide a complete empirical picture
for reviewers.

---

## References to add

Becker, S., LeCun, Y. (1989). Improving the convergence of back-propagation learning
with second-order methods. Proceedings of the 1988 Connectionist Models Summer School.

Elsayed, M., Farrahi, H., Dangel, F., Mahmood, A.R. (2024). Revisiting Scalable
Hessian Diagonal Approximations for Applications in Reinforcement Learning.
arXiv:2406.03276.

Griewank, A., Walther, A. (2008). Evaluating Derivatives: Principles and Techniques
of Algorithmic Differentiation. SIAM.

Lyness, J.N., Moler, C. (1967). Numerical differentiation of analytic functions.
SIAM J. Numer. Anal. 4, 202–210.

Squire, W., Trapp, G. (1998). Using complex variables to estimate derivatives of real
functions. SIAM Rev. 40, 110–112.

Byte, Z. (2026a). One-Pass Exact Hessians via Hypercomplex Perturbation: A Vectorized
Implementation with Implicit Layer Applications. Zenodo. DOI: 10.5281/zenodo.19344150.

Byte, Z. (2026b). Exact Ridge Curvature in One Evaluation. Zenodo.
DOI: 10.5281/zenodo.19356691.

Byte, Z. (2026c). hcderiv v0.2.0 — Vectorized hypercomplex core for exact one-pass
Hessians. Zenodo. DOI: 10.5281/zenodo.19389522.

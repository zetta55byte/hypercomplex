[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19344150.svg)](https://doi.org/10.5281/zenodo.19344150)
# hypercomplex

**Exact derivatives via hypercomplex perturbation algebra.**

One-pass extraction of exact gradients, Hessians, and ridge curvature — no finite differences, no graph tracing, no step-size tuning.

## Install

```bash
pip install numpy scipy
# then drop the hypercomplex/ folder into your project, or:
pip install -e .
```

## Quick Start

```python
from hypercomplex import grad, hessian, ridge_curvature

def f(X):
    return X[0]*X[0] + X[0]*X[1]*3 + X[1]*X[1]*2

g = grad(f, [1.0, 2.0])           # [5. 11.]  — exact
H = hessian(f, [1.0, 2.0])        # [[2. 3.] [3. 4.]] — exact, one evaluation
k = ridge_curvature(f, [1.0, 2.0]) # largest eigenvalue of H
```

## How It Works

Each input coordinate is augmented with a pair of commuting infinitesimal units:

```
X_j = x_j + alpha * i_j + beta * eps_j
```

where `i_j^2 = -1` and `eps_j^2 = 0`. A single evaluation of `f` at the hypercomplex point yields coefficients that encode the exact gradient and Hessian — no approximation, no step size.

## Writing Compatible Functions

Functions must use Hyper arithmetic. Basic operations (`+`, `-`, `*`, `/`, integer `**`) work automatically:

```python
def f(X):
    # X[0], X[1] are Hyper objects
    return X[0]**2 + X[0]*X[1]*3 + X[1]**2*2 + X[0]*5

# For rational functions:
def g(X):
    denom = Hyper.real(2, 1.0) + X[0]**2
    return 1.0 / denom   # uses exact second-order inversion
```

NumPy ufuncs (`np.sin`, `np.exp`, etc.) do not work on Hyper objects directly. Use polynomial or rational approximations, or implement the function using Hyper arithmetic.

## Lambda-Switch Example

```python
from hypercomplex.examples.lambda_switch import run
results = run()
print(f"Ridge curvature at saddle: {results['kappa']:.8f}")
```

## API

| Function | Description |
|---|---|
| `grad(f, x)` | Exact gradient vector |
| `hessian(f, x)` | Exact Hessian matrix (one evaluation) |
| `grad_and_hessian(f, x)` | Both simultaneously |
| `jacobian(f, x)` | Jacobian of vector-valued f |
| `hessian_vector_product(f, x, v)` | H(x) @ v |
| `ridge_curvature(f, x)` | λ_max(H) |
| `principal_curvatures(f, x)` | Eigenvalues and eigenvectors of H |
| `curvature_map(f, xs, ys)` | Ridge curvature over 2D grid |
| `shape_operator(f, x)` | Weingarten map of level set |

## References

1. Byte, Z. (2026). *Exact Second-Order Derivatives in One Forward Pass Using Hypercomplex Perturbation Algebras*. Zenodo. DOI: [10.5281/zenodo.19344150](https://doi.org/10.5281/zenodo.19344150)

2. Byte, Z. (2026). *Exact Ridge Curvature in One Evaluation*. Zenodo. DOI: [10.5281/zenodo.19356691](https://doi.org/10.5281/zenodo.19356691)

## License

MIT

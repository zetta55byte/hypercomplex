"""
hypercomplex
============
Exact derivatives via hypercomplex perturbation algebra.

One-pass extraction of exact gradients, Hessians, and ridge curvature
for scalar and vector-valued functions.

Quick Start
-----------
    from hypercomplex import grad, hessian, ridge_curvature

    def f(X):
        return X[0]*X[0] + X[0]*X[1]*3 + X[1]*X[1]*2

    g = grad(f, [1.0, 2.0])          # exact gradient
    H = hessian(f, [1.0, 2.0])       # exact Hessian, one evaluation
    k = ridge_curvature(f, [1.0, 2.0])  # ridge curvature

References
----------
[1] Byte, Z. (2026). One-Pass Exact Hessians via Hypercomplex Perturbation:
    A Vectorized Implementation with Implicit Layer Applications. Zenodo.
    DOI: 10.5281/zenodo.19394700

[2] Byte, Z. (2026). Exact Ridge Curvature in One Evaluation.
    Zenodo. DOI: 10.5281/zenodo.19356691
"""

from .core.hyper import Hyper
from .core.utils import make_inputs, extract_gradient_hessian
from .derivatives import grad, hessian, grad_and_hessian, jacobian, hessian_vector_product
from .curvature import ridge_curvature, principal_curvatures, curvature_map, shape_operator

__version__ = "0.1.0"
__author__ = "Zetta Byte"

__all__ = [
    "Hyper",
    "make_inputs",
    "extract_gradient_hessian",
    "grad",
    "hessian",
    "grad_and_hessian",
    "jacobian",
    "hessian_vector_product",
    "ridge_curvature",
    "principal_curvatures",
    "curvature_map",
    "shape_operator",
]

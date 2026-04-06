# Contributing to hcderiv

Thank you for your interest in contributing to hcderiv.

## Getting started

```bash
git clone https://github.com/zetta55byte/hypercomplex.git
cd hypercomplex
pip install -e ".[jax]" pytest coverage ruff black pre-commit
pre-commit install
```

## Running tests

```bash
make test          # all 117 tests
make test-numpy    # NumPy backend only (no JAX required)
```

Or directly:

```bash
pytest hypercomplex/tests/ -q
pytest hypercomplex/tests/test_jax_backend.py -v       # JAX backend
pytest hypercomplex/tests/test_jax_xla_backend.py -v   # XLA backend
```

## Code style

```bash
make lint      # ruff + black check
make format    # auto-fix
```

CI enforces ruff and black on every push. All PRs must pass lint and tests.

## Project structure

```
hypercomplex/
  backends/
    __init__.py       # backend registry (numpy, jax, jax-xla)
    jax_xla.py        # JAX-XLA backend (JAXHyperArray, _mul_tensor)
  core/
    hyper.py          # Hyper class — backend-aware coefficient algebra
    utils.py          # make_inputs, extract_gradient_hessian
  derivatives/
    __init__.py       # public API: grad, hessian, grad_and_hessian, ...
  curvature/
    __init__.py       # ridge_curvature, principal_curvatures
  tests/
    test_vectorized_core.py   # 80 NumPy + JAX tests
    test_jax_backend.py       # 21 JAX backend tests
    test_jax_xla_backend.py   # 37 XLA backend tests
benchmarks/
  run_scaling.py              # NumPy vs FD vs JAX scaling
  bench_hessian_backends.py   # NumPy vs JAX vs JAX-XLA timing
examples/
  trust_region.py             # trust-region optimization demo
  pendulum_energy.py          # differentiable physics demo
```

## Adding a new backend

1. Create `hypercomplex/backends/your_backend.py`
2. Implement `make_seeds(x)`, `extract(result)`, `hessian_xla(f, x)`
3. Register in `hypercomplex/backends/__init__.py`
4. Add `backend="your-backend"` routing in `hypercomplex/derivatives/__init__.py`
5. Add tests in `hypercomplex/tests/test_your_backend.py`

See `hypercomplex/backends/jax_xla.py` as the reference implementation.

## Adding a new unary function

Add a method to both `Hyper` (in `core/hyper.py`) and `JAXHyperArray`
(in `backends/jax_xla.py`) following the chain-rule pattern:

```python
def your_func(self):
    a = float(self.coeffs[0])
    f0 = ...   # your_func(a)
    f1 = ...   # your_func'(a)
    f2 = ...   # your_func''(a)
    return self._apply_scalar_func(f0, f1, f2)
```

Then add a test in both `test_vectorized_core.py` and `test_jax_xla_backend.py`.

## Reporting bugs

Open an issue at https://github.com/zetta55byte/hypercomplex/issues with:
- Python version and OS
- hcderiv version (`python -c "import hypercomplex; print(hypercomplex.__version__)"`)
- Minimal reproducible example
- Expected vs actual output

## Questions

Open a GitHub issue with the `question` label.

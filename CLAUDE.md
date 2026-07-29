# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`sfit_minimizer` is a Python implementation of A. Gould's `sfit` gradient-type minimization
algorithm (generalization of Simpson's method to N dimensions: the function is approximated by a
tangent plane). It was designed for point-lens microlensing light curves but the core is a generic
chi2 minimizer. Scientific reference: Yee & Gould 2025 (arXiv:2502.04486).
Docs: https://jenniferyee.github.io/sfit_minimizer/

## Layout and installation

The package lives in `source/sfit_minimizer/` (`pyproject.toml` sets
`package-dir = {"" = "source"}`), so imports only work after `pip install -e .`, under
`uv run`, or with `PYTHONPATH=source`. The legacy `setup.py` is kept alongside
`pyproject.toml`; the `[project]` table is the authoritative metadata, and the version is read
dynamically from `source/sfit_minimizer/version.py`.

`MulensModel` is *not* a hard runtime dependency — the core minimizer needs only numpy and
matplotlib. It is declared in the `mulens` and `test` extras, which is why the CI job installs
`--extra test`.

`sfit_minimizer.DATA_PATH` (set in `__init__.py`) walks three directories up from the module file
to find `<repo>/data`, falling back to a package-local `data/` that does not exist in this repo.
The tests load reference data through `DATA_PATH`, so they must be run against a repo checkout,
not an installed-only copy. Importing the package also imports `matplotlib` (via `mm_funcs`);
`MulensModel` is needed only for the microlensing layer and its tests.

## Commands

```bash
# tests (pytest-style functions + unittest.TestCase classes mixed in the same files)
# This is exactly what CI runs; the whole suite takes ~7 s.
uv run --extra test pytest -v source/sfit_minimizer/test
uv run --extra test pytest source/sfit_minimizer/test/test_mm_funcs.py::test_pspl_1  # single test

# without uv
PYTHONPATH=source pytest source/sfit_minimizer/test/

# examples (runnable scripts, they pop up matplotlib windows)
PYTHONPATH=source python examples/example_00_linear_fit.py

# docs: Sphinx sources are docs/source; the built HTML is committed at the docs/ root
# for GitHub Pages (note the .nojekyll file). Build so the output lands there:
cd docs && sphinx-build -b html source .
# (docs/Makefile's `make html` writes to docs/build instead, which is NOT what is published)
```

Version lives in `source/sfit_minimizer/version.py`; past commits also bump the "Latest release"
line in `README.md`. `docs/source/conf.py` has `release = '0'` and is not kept in sync.

`build/`, `source/sfit_minimizer.egg-info/`, `uv.lock`, and the `__pycache__`/`.pytest_cache`
directories are covered by `.gitignore`. Three `.coverage` files are tracked in git from before
that existed; `.gitignore` does not untrack them.

## CI

`.github/workflows/tests.yml` runs the suite on push to `master` and on every PR, across Python
3.11 / 3.12 / 3.14 via `astral-sh/setup-uv`. It mirrors the workflow in the sibling MMEXOFAST
repo, minus the fast/slow session split (that repo needs it for ~13-minute grid searches; this
suite is ~7 s, so every interpreter gets a full run).

The job relies on `uv` installing the project **editable**, because `DATA_PATH` walks up from the
module file to find `<repo>/data`. A non-editable install would resolve `DATA_PATH` into
site-packages and every data-loading test would fail.

## Architecture

### Core algorithm (`sfit_classes.py`, `sfit_minimize.py`)

`SFitFunction` is the base class users subclass. A subclass must define:
- `calc_model()` (sets `ymod`) **or** `calc_residuals()` (sets `residuals`), and
- `calc_df()` — sets `df`, an (M, N) array of partial derivatives of the *fitting function*
  (not chi2) w.r.t. each of the M parameters at each of the N data points.

Data is always a single `np.array` of shape (N, 3) with columns (x, y, yerr).

`update_all(theta)` runs the whole computation chain in a fixed order, and each quantity is a
cached private attribute exposed through a read-only property:

```
residuals -> chi2 -> df -> dchi2 -> dvec -> bmat -> cmat -> step
```

- `dchi2[i,k] = -2 * residuals[k] * df[i,k] / yerr[k]^2`
- `dvec[i]    = -sum_k dchi2[i,k] / 2`
- `bmat[i,j]  = sum_k df[i,k] df[j,k] / yerr[k]^2`
- `cmat       = inv(bmat)`; `sigmas = sqrt(diag(cmat))`
- `step[i]    = sum_j cmat[i,j] * dvec[j]`

Setting `theta` calls `reset_all()`, nulling every cached quantity — assigning parameters directly
without calling `update_all()` leaves the object empty, not stale.

`minimize(sfit_obj, x0, tol, options, max_iter)` iterates `x_new = x_old + fac * step`, where `fac`
comes from `options['step']`: a float, `None`/absent → 0.1, or `'adaptive'` → starts at 0.01 and
switches to 0.1 once the chi2 improvement drops below 1.0. It returns an `SFitResults`
(scipy `OptimizeResult`-like: `.x`, `.sigmas`, `.fun`, `.success`, `.msg`, `.nit`). It stops with
`success=False` and rolls back to the previous parameters if chi2 gets *worse*, or if `max_iter`
is exceeded; it stops successfully when `old_chi2 - new_chi2 < tol`.

### Microlensing layer (`mm_funcs.py`)

`PointLensSFitFunction` adapts `MulensModel.Event` to `SFitFunction` and is the reason the core
API looks the way it does. Key points:

- Multiple datasets are flattened into one (N, 3) array of *good* points only
  (`_flatten_data()`); `data_len` records per-dataset lengths so `calc_df()` can slice the columns
  back apart.
- The parameter vector is `[<parameters_to_fit>..., f_s_0, f_b_0, f_s_1, f_b_1, ...]`. Fluxes
  fixed via `event.fix_source_flux` / `event.fix_blend_flux` are *removed* from the vector;
  `set_flux_indices()` builds `fs_indices`/`fb_indices` (entry `None` = fixed) mapping each
  dataset's fluxes to their column in theta/b/c/d. Change one of these and the other must follow.
- `update_all()` pushes theta back into `event.model.parameters` and the fixed fluxes, then calls
  `event.fit_fluxes(bad=False)` before delegating to the base class.
- `add_2450000` controls whether 2450000 is added to `t_0` before it goes into the model.
- Derivatives come from `fit.get_d_A_d_params_for_point_lens_model()`, scaled by source flux;
  d/d f_s is the magnification and d/d f_b is 1.

`fit_mulens_event()` is the convenience wrapper: build the function object, run `minimize` with
`options={'step': 'adaptive'}`, optionally plot.

Known failure mode: if `u_0` is fit and is too close to zero, `cmat` inversion raises
`numpy.linalg.LinAlgError: Singular matrix`.

### Tests

- `test_type_checking.py` — the setters in `SFitFunction` do explicit type/shape validation
  (TypeError vs. ValueError); these tests pin that behavior, so keep raising the specific type.
- `test_linear_fn.py` — polynomial fits against `data/PolynomialTest/`, exercises the core only.
- `test_mm_funcs.py` — regression tests against A. Gould's original Fortran `sfit`. Reference
  output lives in `data/MMTest/Matrices/<MODEL>_<step_size>/fort.NN` and is parsed by
  `FortranSFitFile` (blocks introduced by `# <attrname>` lines). `ComparisonTest` reads the step
  factor out of the directory name, compares the a/b/c/d matrices for the first 3 iterations
  element by element, and then checks the converged fit. When changing the algorithm, these
  numbers are the ground truth — do not adjust the reference files to make a change pass.

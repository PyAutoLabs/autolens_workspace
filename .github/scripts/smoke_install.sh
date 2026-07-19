#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

if [ "$PYTHON_VERSION" = "3.12" ]; then
  pip install ./PyAutoNerves ./PyAutoFit ./PyAutoArray ./PyAutoGalaxy ./PyAutoLens
  pip install "./PyAutoArray[optional]" "./PyAutoGalaxy[optional]" "./PyAutoLens[optional]"
else
  pip install ./PyAutoNerves ./PyAutoFit ./PyAutoArray ./PyAutoGalaxy ./PyAutoLens
  pip install numba nufftax
fi
# NOTE: do NOT `pip install tensorflow-probability==0.25.0` here. The stable
# release crashes at import under the resolved modern JAX
# (`jax.interpreters.xla.pytype_aval_mappings` was removed), which broke the
# JAX Matern-kernel (delaunay_mge) likelihood path. The working modified-Bessel
# dependency is `tfp-nightly`, pinned by `PyAutoArray[optional]` above.
pip install jupyter nbconvert ipynb-py-convert

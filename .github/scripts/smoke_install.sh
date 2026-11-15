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
# NOTE: tensorflow-probability is no longer a dependency at all. The JAX
# Matern-kernel path's modified-Bessel `bessel_kve` is vendored in PyAutoArray
# (autoarray/inversion/regularization/bessel.py, PyAutoArray#694), replacing
# the former `tfp-nightly` pin. Do NOT `pip install tensorflow-probability`
# here: the stable 0.25.0 release crashes at import under modern JAX
# (`jax.interpreters.xla.pytype_aval_mappings` was removed) and nothing needs it.
pip install jupyter nbconvert ipynb-py-convert

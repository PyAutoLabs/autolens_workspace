"""
Datacube Likelihood Function: Step-by-step walkthrough
=======================================================

Step-by-step exploration of the JAX-compiled likelihood for an ALMA-style
datacube modeled as a list of ``Interferometer`` channels with a shared
lens model and a per-channel pixelized source reconstruction.

The walkthrough mirrors the single-channel
``jax_profiling/interferometer/pixelization.py`` script — same lens model,
same ``RectangularAdaptDensity`` mesh + ``Constant`` regularization, same JAX
JIT pattern — but the analysis is now an ``af.FactorGraphModel`` over
N independent ``AnalysisInterferometer``s.

What's new compared to the single-channel version
-------------------------------------------------

1. **Dataset.** Instead of one ``Interferometer.from_fits`` call, we build
   a ``dataset_list`` by looping over ``channel_000/`` ... ``channel_NNN/``.
2. **Per-channel analyses.** One ``AnalysisInterferometer(dataset=..., use_jax=True)``
   per channel.
3. **Likelihood is the explicit sum.** The walkthrough builds the cube
   log-evidence as ``sum(analysis.log_likelihood_function(instance) for
   analysis in analysis_list)``, which is exactly what ``af.FactorGraphModel``
   abstracts over for the user-facing modeling script. Showing the sum
   directly keeps the JAX path easy to register as a pytree (the base
   ``Collection`` model is registered, and the same instance is reused
   across channels).
4. **All parameters are shared.** The lens mass + shear is a single set of
   priors used by every channel — that's the "global non-linear lens"
   half of Hannah's request from the prompt. The pixelization has no free
   priors; its inversion is a per-channel linear solve run inside each
   ``AnalysisInterferometer.log_likelihood_function``. That gives us the
   "per-channel linear solution" half for free, with no extra wiring.
5. **vmap is skipped.** It would have to wrap the per-channel function
   rather than the summed loop — that lives in the (deferred)
   shared-``L^T W~ L`` optimisation issue, not this prototype.

The user-facing ``modeling.py`` script wraps ``analysis_list`` in
``af.AnalysisFactor`` + ``af.FactorGraphModel`` so the non-linear search
does the same sum behind the scenes. For the likelihood walkthrough we
show the loop explicitly.

Phases
------

* **PART A — Setup.** Auto-simulate the cube if missing, load N channels,
  build the model, instantiate at prior medians, register the model as a
  JAX pytree.
* **PART B — Eager NumPy baseline.** Per-channel ``FitInterferometer``
  with ``xp=np`` to get a ground-truth log-evidence for each channel and
  the summed total.
* **PART C — JIT cube likelihood.** ``jax.jit`` over the explicit sum;
  time lower / compile / first-call / steady-state.
* **PART D — Correctness.** Eager-vs-JIT log-evidence agreement at
  ``rtol=1e-4``.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autolens as al
from autofit.jax import register_model as _register_model_pytrees


# ---------------------------------------------------------------------------
# Cube configuration
# ---------------------------------------------------------------------------

CUBE_PATH = Path("dataset") / "interferometer" / "datacube" / "sim_simple"
N_CHANNELS = 4
PIXEL_SCALE = 0.1
REAL_SPACE_SHAPE = (256, 256)
MASK_RADIUS = 3.0
MESH_SHAPE = (14, 14)
REGULARIZATION_COEFFICIENT = 1.0


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def jit_profile(func, label, *args, n_repeats=10):
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")
    return compiled, result


timer = Timer()


# ===================================================================
# PART A — Setup
# ===================================================================

print("\n=== PART A — Setup ===")

# Auto-simulate the cube on disk if needed
_script_dir = Path(__file__).resolve().parent
if not (CUBE_PATH / "channel_000" / "data.fits").exists():
    print("  Cube missing on disk, running simulator.py...")
    subprocess.run(
        [sys.executable, str(_script_dir / "simulator.py")],
        check=True,
    )

real_space_mask = al.Mask2D.circular(
    shape_native=REAL_SPACE_SHAPE,
    pixel_scales=PIXEL_SCALE,
    radius=MASK_RADIUS,
)

dataset_list = []
with timer.section("dataset_load"):
    for c in range(N_CHANNELS):
        channel_path = CUBE_PATH / f"channel_{c:03d}"
        dataset_list.append(
            al.Interferometer.from_fits(
                data_path=channel_path / "data.fits",
                noise_map_path=channel_path / "noise_map.fits",
                uv_wavelengths_path=channel_path / "uv_wavelengths.fits",
                real_space_mask=real_space_mask,
                transformer_class=al.TransformerDFT,
            )
        )

n_visibilities = dataset_list[0].uv_wavelengths.shape[0]
print(f"  channels:           {N_CHANNELS}")
print(f"  visibilities/chan:  {n_visibilities}")
print(f"  real-space grid:    {REAL_SPACE_SHAPE[0]} x {REAL_SPACE_SHAPE[1]}")

# ---------------------------------------------------------------------------
# Base model — shared lens, pixelized source
# ---------------------------------------------------------------------------

print("\n=== PART A — Model construction ===")

with timer.section("model_build"):
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    _lens_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_ell[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularAdaptDensity(shape=MESH_SHAPE),
        regularization=al.reg.Constant(coefficient=REGULARIZATION_COEFFICIENT),
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  base model free parameters: {model.total_free_parameters}")

# ---------------------------------------------------------------------------
# Per-channel analyses
# ---------------------------------------------------------------------------

print("\n=== PART A — Per-channel analyses ===")

analysis_list = [
    al.AnalysisInterferometer(dataset=dataset, use_jax=True)
    for dataset in dataset_list
]

print(f"  channels in analysis_list: {len(analysis_list)}")

# ---------------------------------------------------------------------------
# Concrete instance + pytree registration
# ---------------------------------------------------------------------------

print("\n=== PART A — Concrete instance + pytree registration ===")

with timer.section("instance_from_vector"):
    param_vector = model.physical_values_from_prior_medians
    instance = model.instance_from_vector(vector=param_vector)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

# JIT input: the instance with all parameter leaves promoted to JAX arrays
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)


# ===================================================================
# PART B — Eager NumPy baseline (per-channel + summed)
# ===================================================================

print("\n=== PART B — Eager NumPy baseline ===")

tracer = al.Tracer(galaxies=list(instance.galaxies))
print(f"  tracer planes: {tracer.total_planes}")

per_channel_fom = []
with timer.section("eager_fit_per_channel"):
    for c, dataset in enumerate(dataset_list):
        fit = al.FitInterferometer(dataset=dataset, tracer=tracer, xp=np)
        per_channel_fom.append(float(fit.figure_of_merit))
        print(f"    channel {c:03d}: figure_of_merit = {per_channel_fom[-1]:.6f}")

eager_total = sum(per_channel_fom)
print(f"  summed eager log-evidence: {eager_total:.6f}")


# ===================================================================
# PART C — JIT cube likelihood
# ===================================================================

print("\n=== PART C — JIT cube likelihood ===")


def cube_log_likelihood(params_tree):
    """Cube log-evidence as an explicit sum across channels.

    This is exactly what ``af.FactorGraphModel`` does internally for the
    user-facing modeling script: route the same ``instance`` to each
    per-channel ``AnalysisInterferometer.log_likelihood_function`` and sum
    the per-channel log-evidences.
    """
    total = jnp.zeros(())
    for analysis in analysis_list:
        total = total + analysis.log_likelihood_function(instance=params_tree)
    return total


_, jit_total = jit_profile(cube_log_likelihood, "cube_likelihood", params_tree)
jit_per_call = timer.records[-1][1] / 10

print(f"  JIT total log-evidence: {float(jit_total):.6f}")


# ===================================================================
# PART D — Correctness
# ===================================================================

print("\n=== PART D — Correctness ===")

np.testing.assert_allclose(
    float(jit_total),
    eager_total,
    rtol=1e-4,
    err_msg=(
        "datacube/likelihood_function: JIT FactorGraph log-evidence does not "
        "match the summed eager per-channel figure_of_merit"
    ),
)
print(f"  Eager-vs-JIT correctness PASSED at rtol=1e-4")
print(f"    eager total: {eager_total:.6f}")
print(f"    JIT total:   {float(jit_total):.6f}")
print(f"    abs diff:    {abs(float(jit_total) - eager_total):.3e}")
print(f"    JIT per-call (steady): {jit_per_call:.6f} s")

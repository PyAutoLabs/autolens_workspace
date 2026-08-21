"""
Start Here: Datacube
====================

A datacube is a stack of interferometer observations of the same lens taken across many spectral channels — for
example, an ALMA observation of a CO emission line in a high-redshift lensed galaxy, where each channel records
visibilities at a slightly different frequency. The lens galaxy is the same in every channel, but the source's
emission-line morphology varies across the cube: bright in channels close to the line peak, faint in channels at
the line wings, and shifting in shape if the source has internal velocity structure.

This script shows how to model such a cube in PyAutoLens by treating it as exactly that — a Python list of
`Interferometer` objects, one per channel — and tying them together with `af.FactorGraphModel`. The lens model is
shared across every channel, and each channel reconstructs its own pixelized source. The result is a per-channel
sequence of source-plane reconstructions whose combined likelihood drives a single, global lens model fit.

This is Phase 1 of datacube modeling. It deliberately runs each channel's NUFFT and source inversion
independently, because reusing existing single-channel code is the fastest path to a working end-to-end
demonstration. The faster shared-`Lᵀ W̃ L` design — which exploits the fact that `uv_wavelengths` and
`noise_map` change very little across an emission line — is a follow-up that lands once this prototype is proven.

If you've already read `interferometer/start_here.py`, the bulk of this script will look familiar. The new
ingredients are the loop that loads each channel, the `AnalysisFactor` per channel, and the `FactorGraphModel`
that sums them.

__Contents__

- **JAX:** GPU/CPU acceleration via JAX — the same backend that single-channel interferometer fits use.
- **Imports:** Standard PyAutoLens imports + `autofit` for the FactorGraph wiring.
- **Mask:** A single 2D real-space mask shared across all channels.
- **Dataset:** Where the per-channel cube lives on disk and how to point this script at your own.
- **Dataset Auto-Simulation:** Run `simulator.py` automatically if the cube isn't already on disk.
- **Dataset Loading:** Loop over channel folders to build a `dataset_list` of `Interferometer` objects.
- **Sparse Operators:** Per-channel sparse-operator pre-compute used by the pixelized source inversion.
- **Positions:** Load the cube's multiple-image positions and build a shared `PositionsLH` penalty.
- **Settings:** Disable the positive-only solver (visibility inversions can take negative pixel values).
- **Mesh Shape:** The pixelization mesh shape — fixed before modeling because JAX needs static shapes.
- **Model:** Shared lens galaxy + pixelized source. The same model is reused unchanged across every channel.
- **Per-Channel Analyses:** One `AnalysisInterferometer` per channel, all sharing the same `PositionsLH`.
- **FactorGraph:** Wrap each analysis in an `AnalysisFactor`; combine via `af.FactorGraphModel`.
- **Search:** Configure the `Nautilus` non-linear search.
- **Model Fit:** Fit the cube — the FactorGraph routes shared lens parameters into every channel's likelihood.
- **Result:** What the returned `result_list` contains and how to inspect per-channel reconstructions.
- **Wrap Up:** Pointers to `modeling.py`, `simulator.py`, and the JAX likelihood walkthrough.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU support, your
fits will run much faster (tens of minutes instead of several hours for a 4-channel cube). On CPU, JAX still
provides a meaningful speed-up via multithreading, but datacube fits are inherently more expensive than
single-channel fits because the per-channel inversion cost multiplies by the number of channels.

If you don't have a GPU locally, consider Google Colab, which provides free GPUs.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import subprocess
import sys
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Mask__

A single 2D circular mask shared across every channel. The lens galaxy and the source emission live in the same
sky region in every frequency channel, so masking once and reusing the mask is correct.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

The reference cube ships in `dataset/interferometer/datacube/sim_simple/`, with one subfolder per channel
(`channel_000/`, `channel_001/`, ...) each containing `data.fits`, `noise_map.fits`, `uv_wavelengths.fits` and
the true `tracer.json`.

To point this script at your own cube, drop your channel folders in alongside the reference cube and update
`dataset_name`. Each channel folder must contain `data.fits`, `noise_map.fits` and `uv_wavelengths.fits` in the
shape produced by `al.SimulatorInterferometer` (visibilities and noise stored as ``(n_vis, 2)`` real/imag pairs;
baselines as ``(n_vis, 2)`` u/v pairs).
"""
dataset_label = "datacube"
dataset_name = "sim_simple"
dataset_path = Path("dataset") / "interferometer" / dataset_label / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, "scripts/interferometer/features/datacube/simulator.py"],
        check=True,
    )

"""
__Dataset Loading__

Build the cube by loading each channel folder as an `Interferometer` object. The result is a Python list — no
new dataset class involved. Every downstream component (analyses, factors, fits, plotters) operates on this
list directly, which is the whole reason we picked the list-of-Interferometer design over a bespoke
`Datacube3D` class.
"""
channel_paths = sorted(
    p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("channel_")
)
print(f"Found {len(channel_paths)} channels in {dataset_path}")

dataset_list = [
    al.Interferometer.from_fits(
        data_path=channel_path / "data.fits",
        noise_map_path=channel_path / "noise_map.fits",
        uv_wavelengths_path=channel_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )
    for channel_path in channel_paths
]

aplt.subplot_interferometer_dirty_images(dataset=dataset_list[0])

"""
__Sparse Operators__

Pixelized source modeling uses sparse linear algebra to keep memory and runtime manageable. We pre-compute a
sparse-operator matrix per channel — `apply_sparse_operator()` does this work in seconds for SMA-scale data and
in minutes per channel on CPU for ALMA-scale data. For very large cubes you'll want to compute these once and
cache them; see `pixelization/many_visibilities_preparation.py` for the pattern.
"""
dataset_list = [
    dataset.apply_sparse_operator(use_jax=True, show_progress=False)
    for dataset in dataset_list
]

"""
__Positions__

Pixelized source modeling has a known failure mode: without a position-likelihood penalty, the search routinely
converges on demagnified-source local maxima where the source pixels are reconstructed in low-magnification
regions of the source plane that fit the noise rather than the lensed signal. The `PositionsLH` penalty defends
against that by reading a small set of multiple-image positions from disk and adding a likelihood penalty for
any candidate lens model whose source-plane back-projection of those positions spreads them apart.

For the cube we load `positions.json` (written by `simulator.py`) and build one `PositionsLH` that gets passed to
every per-channel analysis below. The lens model is shared across channels via the FactorGraph, so applying the
same penalty in every analysis enforces a single global constraint.

The threshold of 0.3" is generous; for a real fit you'd tighten it (typically < 0.05") once the lens model has
settled into the right region of parameter space.
"""
positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Settings__

Interferometer pixelizations disable the positive-only inversion solver. The visibility measurement process
can produce genuinely negative dirty-image pixel values, so the source-plane reconstruction must be allowed to
go negative — forcing positivity here would create unphysical bias.
"""
settings = al.Settings(use_positive_only_solver=False)

"""
__Mesh Shape__

The pixelization mesh shape is fixed before modeling because JAX needs static-shape arrays for its source-plane
linear algebra. We use a 14 x 14 `RectangularRTUAdaptDensity` mesh — small enough to make the prototype iteration
cheap, large enough to capture the emission-line source morphology produced by the simulator.
`RectangularRTUAdaptDensity` adapts the source-plane pixel density to the lensing magnification map, giving more
pixels to the highly-magnified regions of the source plane where the lensed signal is concentrated.
"""
mesh_pixels_yx = 14
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model__

The cube model has two ingredients:

 - A shared `Isothermal + ExternalShear` lens. There are 7 free parameters (mass centre, ellipticity components,
   einstein radius, two shear components). The lens does not change with frequency, so a single set of priors is
   used for every channel.
 - A pixelized source: a `RectangularRTUAdaptDensity` mesh with `Constant` regularization (1 free parameter — the
   regularization coefficient). The pixelization itself has no per-pixel priors; the source-plane fluxes are a
   linear inversion output computed by each channel's `AnalysisInterferometer` at fit time. That is what makes
   each channel an independent linear solve while sharing all of the non-linear parameters.

The total dimensionality of the non-linear parameter space is therefore 8.
"""
# Lens:
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source (pixelization, no per-pixel priors):
mesh = af.Model(al.mesh.RectangularRTUAdaptDensity, shape=mesh_shape)
regularization = af.Model(al.reg.Constant)
pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall lens model:
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Per-Channel Analyses__

One `AnalysisInterferometer` per channel — there are no per-channel parameters here, only per-channel data.
Each analysis runs its own NUFFT, builds its own visibility-space inversion, and returns its own log-evidence
when called with a candidate lens model. The shared `positions_likelihood` is passed to every analysis to apply
the same global multiple-image penalty across the cube.
"""
analysis_list = [
    al.AnalysisInterferometer(
        dataset=dataset,
        settings=settings,
        positions_likelihood_list=[positions_likelihood],
        use_jax=True,
    )
    for dataset in dataset_list
]

"""
__FactorGraph__

`af.AnalysisFactor` pairs each analysis with a deep copy of the base model, and `af.FactorGraphModel` combines
the factors into a single global model whose log-likelihood is the sum of the per-factor log-likelihoods. With
no per-factor prior overrides, every prior is *identified* across factors — the factor-graph machinery
deduplicates them — so the global model has the same dimensionality as the single-channel base model.

If you wanted per-channel free parameters (for example, a per-channel `intensity` for a parametric source),
you'd override that prior on each `model.copy()` before wrapping it in an `AnalysisFactor`. See
`autolens_workspace/scripts/multi_dataset/modeling.py` for how that works in the multi-band case.
"""
analysis_factor_list = [
    af.AnalysisFactor(prior_model=model.copy(), analysis=analysis)
    for analysis in analysis_list
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(f"  channels in factor graph:           {len(analysis_factor_list)}")
print(
    f"  global model free parameters:       {factor_graph.global_prior_model.total_free_parameters}"
)

"""
__Search__

`Nautilus` is the standard non-linear search for PyAutoLens. The lens dimensionality is unchanged from a
single-channel fit, so `n_live=100` is a reasonable starting point — but the per-likelihood cost is N times
larger because each likelihood call runs N inversions, so wall-clock time scales linearly with the number of
channels.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "datacube",
    name="start_here",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=50000,
)

"""
__Model Fit__

We pass the factor graph's `global_prior_model` as the model and the factor graph itself as the analysis — this
is the same shape used for any multi-dataset PyAutoFit fit. Internally the search proposes lens parameters from
the global prior, hands them to the FactorGraph's `log_likelihood_function`, and the FactorGraph routes them
into every channel's `AnalysisInterferometer.log_likelihood_function` and sums the per-channel log-evidences.

**Run time on CPU is dominated by the per-channel inversion.** A 4-channel SMA-scale cube finishes in a few
hours on CPU; ALMA-scale cubes with 50+ channels need GPU acceleration to complete in reasonable time.
"""
print(
    """
    The non-linear search has begun running.

    Per-channel inversions multiply the per-likelihood cost — expect this to take longer than a single-channel
    interferometer fit. On CPU plan for hours; on GPU, tens of minutes.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

`result_list` contains one entry per factor — one per channel — each carrying its own `FitInterferometer`
against the maximum-likelihood lens model. Use them to inspect:

 - Per-channel source reconstructions (each channel's pixelized source is independent).
 - Per-channel dirty images and residuals.
 - The shared maximum-likelihood lens parameters (identical across factors by construction).

Compared against `dataset/interferometer/datacube/<name>/cube_summary.json`, the per-channel reconstructed total
flux should trace the input emission-line spectrum.

__Wrap Up__

This script walks through the full datacube modeling pipeline. For deeper dives:

 - `modeling.py` — the focused FactorGraph + Nautilus example, ready to copy and adapt for your own cube.
 - `simulator.py` — how the reference cube is generated.
 - `likelihood_function.py` — a step-by-step JAX walkthrough of how the
   per-channel log-evidences are built and summed inside the FactorGraph, with an explicit eager-vs-JIT
   correctness check.

Phase 2 work — the shared-`Lᵀ W̃ L` optimisation that exploits channel-invariant `uv_wavelengths` and
`noise_map` to bring ALMA-scale cubes inside CPU runtime budgets — is a separate follow-up issue.
"""

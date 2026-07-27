"""
Modeling: Datacube
===================

This script fits a list of `Interferometer` channels — a "datacube" — with a single shared lens model and a
per-channel pixelized source reconstruction. Each channel is an independent `Interferometer` dataset; the
`af.FactorGraphModel` ties them together by feeding the same lens parameters into every channel's
`AnalysisInterferometer.log_likelihood_function` and summing the per-channel log-evidences.

A datacube modeled this way captures spatially-resolved spectral-line emission: every channel reconstructs its
own source-plane pixelization, so an emission line that brightens-and-fades across the cube produces a sequence
of source-plane reconstructions whose total flux traces the line profile while the lens mass stays fixed.

This script is the focused-modeling sibling of `start_here.py`. Read `start_here.py` first for the narrative
walkthrough; this file is the one to copy and adapt for your own cube.

__Contents__

- **Mask:** Define the 2D real-space mask applied to every channel.
- **Dataset:** Where the per-channel cube lives on disk and how to point this script at your own.
- **Dataset Auto-Simulation:** Run `simulator.py` automatically if the cube isn't already on disk.
- **Dataset Loading:** Loop over the channel folders and load each as an `Interferometer` object.
- **Sparse Operators:** Pre-compute per-channel sparse-operator matrices used by the pixelized source inversion.
- **Settings:** Disable the positive-only solver so visibility-space inversions can take negative pixel values.
- **Mesh Shape:** Pixelization mesh size — fixed before modeling because JAX needs static-shape arrays.
- **Model:** Compose the shared `Isothermal + ExternalShear` lens and pixelized source.
- **Per-Channel Analyses:** One `AnalysisInterferometer` per channel, with `use_jax=True`.
- **FactorGraph:** Wrap each analysis in an `AnalysisFactor` and combine via `af.FactorGraphModel`.
- **Search:** Configure the `Nautilus` non-linear search.
- **Model Fit:** Run the fit. Per-channel inversion cost dominates runtime — see notes inline.
- **Wrap Up:** Summary of the script and pointers to the JAX likelihood walkthrough in
  ``likelihood_function.py``.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import subprocess
import sys
from pathlib import Path

import autofit as af
import autolens as al

"""
__Mask__

Every channel uses the same `real_space_mask` — the lens galaxy and source position don't depend on frequency,
so masking once is correct. The mask radius is generous enough to contain the lensed source's full extent.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

The datacube lives under `dataset/interferometer/datacube/<dataset_name>/`, with one subfolder per channel
(`channel_000/`, `channel_001/`, ...). To point this script at your own cube, drop your channel folders in
alongside the reference cube and update `dataset_name`. Each channel folder must contain `data.fits`,
`noise_map.fits` and `uv_wavelengths.fits` in the shape produced by `al.SimulatorInterferometer`.
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
new dataset class involved. Channels are discovered by sorted directory listing, so you can add channels by
simply dropping more `channel_NNN/` folders in.
"""
channel_paths = sorted(
    p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("channel_")
)
print(f"Loading {len(channel_paths)} channels from {dataset_path}")

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

"""
__Sparse Operators__

Pixelized source modeling uses sparse linear algebra to keep memory and runtime manageable. We pre-compute a
sparse-operator matrix per channel so each `AnalysisInterferometer.log_likelihood_function` reuses it directly
during the fit. For SMA-scale data this finishes in seconds per channel; for ALMA-scale cubes it can take
minutes per channel on CPU, in which case see `pixelization/many_visibilities_preparation.py` for how to
compute and cache them once.
"""
dataset_list = [
    dataset.apply_sparse_operator(use_jax=True, show_progress=False)
    for dataset in dataset_list
]

"""
__Positions__

Load the cube's multiple-image positions (saved by `simulator.py`) and wrap them in an `al.PositionsLH` penalty.
For pixelized fits this is essentially required: without the penalty, the search routinely converges on
demagnified-source local maxima where the source pixels are reconstructed in low-magnification regions of the
source plane that fit the noise rather than the lensed signal.

The lens model is shared across every channel via the FactorGraph, so a single `PositionsLH` (built once and
passed to every per-channel analysis) applies the same global constraint everywhere.
"""
positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Settings__

Interferometer pixelizations disable the positive-only inversion solver — the visibility measurement process
can produce genuinely negative dirty-image pixel values, so the source-plane reconstruction must be allowed
to go negative.
"""
settings = al.Settings(use_positive_only_solver=False)

"""
__Mesh Shape__

The pixelization mesh shape is fixed before modeling because JAX needs static array shapes. We use a
14 x 14 ``RectangularAdaptDensity`` mesh — small enough to keep the prototype cheap, large enough to capture the
emission-line source morphology produced by the simulator. `RectangularAdaptDensity` adapts the source-plane
pixel density to the lensing magnification map, giving more pixels to the highly-magnified source-plane regions
where the lensed signal is concentrated.
"""
mesh_pixels_yx = 14
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model__

The lens galaxy is a shared `Isothermal + ExternalShear`, identical across every channel. The source galaxy is
a `Pixelization` with a `RectangularAdaptDensity` mesh and `Constant` regularization — the inversion runs
independently per channel inside each `AnalysisInterferometer`, giving each channel its own source-plane
reconstruction without adding any model parameters.

There are no per-channel free parameters: every prior in this base model is identified across factors when the
`FactorGraph` deduplicates them below.
"""
# Lens:
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source (pixelization, no free priors):
mesh = af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape)
regularization = af.Model(al.reg.Constant)
pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall lens model:
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Per-Channel Analyses__

One `AnalysisInterferometer` per channel, all with `use_jax=True` so the FactorGraph fit runs on the JAX
backend. The shared `positions_likelihood` is passed to every analysis — same penalty, every channel.

__Shared Preloads__

Because every channel shares the same lens model, a large fraction of each channel's likelihood is *identical
work*. Ray-tracing the lens model, building the source-plane mapper (its mesh and mapping matrix `L`) and the
curvature matrix `F = LᵀW̃L` are the dominant inversion-setup costs — and they are the same for every channel.
Recomputing them once per channel is pure waste.

Setting `shared_preloads=True` opts each analysis into the `FactorGraphModel` shared-state mechanism: the
channel-invariant inversion-setup quantities (the mapper and `F`) are computed **once** on the lead channel
and reused by every other channel, instead of being rebuilt `N` times. For a many-channel cube this collapses
the dominant inversion-setup cost from `N ×` to `1 ×`, a large speed-up that grows with the number of channels.

This is only correct when those quantities really are channel-invariant — i.e. when the lens model is shared
(it is here) **and** the `uv_wavelengths` and `noise_map` are the same for every channel (the curvature matrix
depends on them through `W̃`). The datacube simulated in `simulator.py` is built exactly this way (identical
`uv_wavelengths` and noise across channels — the narrow-emission-line regime). If your cube has per-channel
`uv`/noise that differ significantly, leave `shared_preloads=False` (the default) so each channel computes its
own inversion — preloading an invalid quantity would silently corrupt the likelihood. The shared and unshared
paths are asserted to give identical likelihoods in
`autolens_workspace_test/scripts/interferometer/datacube/shared_preloads.py`.
"""
analysis_list = [
    al.AnalysisInterferometer(
        dataset=dataset,
        settings=settings,
        positions_likelihood_list=[positions_likelihood],
        use_jax=True,
        shared_preloads=True,
    )
    for dataset in dataset_list
]

"""
__FactorGraph__

Each analysis is wrapped in an `af.AnalysisFactor` paired with a deep copy of the base model. With no per-factor
prior overrides, every prior is identified across factors — so the global model has the same dimensionality as
the single-channel base model. ``af.FactorGraphModel(..., use_jax=True)`` sums the per-channel log-evidences
internally, which is exactly the cube log-likelihood you'd write by hand.
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

`Nautilus` is the standard non-linear search for PyAutoLens. Datacube fits typically need fewer live points
than imaging fits because the lens dimensionality is unchanged — only the per-channel inversions multiply.
Tune `n_live` for your problem.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "datacube",
    name="modeling",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=50000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Model Fit__

Pass the factor graph's `global_prior_model` as the model and the factor graph itself as the analysis — that's
the same shape you'd use for any multi-dataset PyAutoFit fit.

**Run time on CPU is dominated by the per-channel inversion.** A 4-channel SMA-scale cube finishes in a few
hours on CPU; ALMA-scale cubes with 50+ channels need GPU acceleration to complete in reasonable time. The
``Lᵀ W̃ L`` shared-precompute optimisation (Aris's design — exploit the fact that ``uv_wavelengths`` and
``noise_map`` change very little channel-to-channel) is the natural follow-up that brings ALMA-scale cubes
back inside the budget.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell with progress once the search has completed - this could take a while
    for a datacube fit, since per-channel inversions multiply the per-likelihood cost.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Wrap Up__

The result returned by `search.fit` is a list of per-factor results — one entry per channel — each carrying its
own `FitInterferometer` against the maximum-likelihood lens model. Use them to inspect the per-channel source
reconstructions, dirty images, and residuals.

For a step-by-step look at how the per-channel likelihood is summed (and at the JAX JIT pattern that drives
this fit), see ``likelihood_function.py``.
"""

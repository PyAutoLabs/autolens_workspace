"""
Modeling: Datacube — Parametric Source
=======================================

This script fits a datacube — a list of `Interferometer` channels — with a single shared lens model and a
**parametric** source (`al.lp.Sersic`) whose morphology is shared across channels and whose `intensity` varies
channel-to-channel. It is the parametric-fit sibling of `modeling.py`, which fits the same cube with a
pixelized source instead.

Use this script when:

 - You expect the source emission to be well-described by a single Sersic-like shape across the line — for
   example, a high-signal-to-noise unresolved emission line in a kinematically simple galaxy.
 - You want a cheaper fit than the pixelization variant. Parametric inversions are 1–2 orders of magnitude
   faster than per-channel pixelizations because there is no source-plane linear inversion.
 - You want the per-channel `intensity` posterior directly, with no need to integrate a pixelized
   reconstruction back to a total flux.

Use `modeling.py` (the pixelized variant) instead when the source has complex morphology, internal velocity
structure that varies across the cube, or signal-to-noise too low for a Sersic fit to converge.

The FactorGraph wiring here is the canonical "extend the model per dataset" pattern from
`autolens_workspace/scripts/multi/modeling.py`: every prior in the base model is shared across channels by
default, and we explicitly override the source `intensity` prior per `AnalysisFactor` to make it per-channel.

__Contents__

- **Mask:** Define the 2D real-space mask applied to every channel.
- **Dataset:** Where the per-channel cube lives on disk and how to point this script at your own.
- **Dataset Auto-Simulation:** Run `simulator.py` automatically if the cube isn't already on disk.
- **Dataset Loading:** Loop over the channel folders and load each as an `Interferometer` object.
- **Positions:** Load multiple-image positions and build a shared `PositionsLH` penalty.
- **Settings:** Default `al.Settings()` — no positive-only-solver tweak needed (no inversion).
- **Model:** Compose the shared `Isothermal + ExternalShear` lens and `Sersic` source.
- **Per-Channel Analyses:** One `AnalysisInterferometer` per channel, with `use_jax=True` and the shared `PositionsLH`.
- **FactorGraph:** Per-factor `model.copy()` with the source `intensity` prior overridden per channel.
- **Search:** Configure the `Nautilus` non-linear search.
- **Model Fit:** Run the fit. Per-channel cost is much cheaper than the pixelization variant.
- **Wrap Up:** Pointers to `modeling.py`, `start_here.py`, and the JAX likelihood walkthrough.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import subprocess
import sys
from pathlib import Path

import autofit as af
import autolens as al

"""
__Mask__
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

The datacube lives under `dataset/interferometer/datacube/<dataset_name>/`, with one subfolder per channel.
"""
dataset_label = "datacube"
dataset_name = "sim_simple"
dataset_path = Path("dataset") / "interferometer" / dataset_label / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    subprocess.run(
        [sys.executable, "scripts/interferometer/features/datacube/simulator.py"],
        check=True,
    )

"""
__Dataset Loading__
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
__Positions__

Load the cube's multiple-image positions and wrap them in an `al.PositionsLH` penalty. Pixelized fits really
need this; parametric Sersic fits less so, but the penalty still helps the search avoid local maxima where the
mass model places multiple images far apart in the source plane.
"""
positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Settings__

Parametric sources don't run a source-plane inversion, so we don't need to disable a positive-only solver
here. Default settings are fine.
"""
settings = al.Settings()

"""
__Model__

The cube model has two ingredients:

 - A shared `Isothermal + ExternalShear` lens (7 free parameters).
 - A parametric `al.lp.Sersic` source. Its morphology — `centre`, `ell_comps`, `effective_radius`,
   `sersic_index` — is shared across channels (4 free parameters). Its `intensity` will be overridden
   per-factor below to give each channel its own free `intensity` parameter, capturing the emission-line
   spectrum.

Total free parameters:
   7 (lens)  +  4 (source morphology, shared)  +  N_channels * 1 (per-channel intensity)
   = 11 + N_channels

For the 4-channel reference cube that's 15 free parameters total — tractable for `Nautilus`.
"""
# Lens:
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source (parametric Sersic — base prior on intensity gets overridden per factor below):
bulge = af.Model(al.lp.Sersic)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall lens model:
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Per-Channel Analyses__

The shared `positions_likelihood` is passed to every per-channel analysis so the multiple-image penalty applies
globally to the lens model.
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

Each analysis is wrapped in an `af.AnalysisFactor` paired with a deep copy of the base model. We then override
the source `intensity` prior per factor — overwriting the prior with a fresh `LogUniformPrior` makes that
parameter per-channel rather than identified across factors. Every other source parameter (centre, ell_comps,
effective_radius, sersic_index) is left untouched, so the FactorGraph identifies them across channels.

This is the canonical "extend the model per dataset" pattern from `autolens_workspace/scripts/multi/modeling.py`.
"""
analysis_factor_list = []

for analysis in analysis_list:
    model_analysis = model.copy()

    # Per-channel intensity prior — overrides the shared default with a fresh prior object,
    # which the FactorGraph treats as a distinct (per-factor) parameter.
    model_analysis.galaxies.source.bulge.intensity = af.LogUniformPrior(
        lower_limit=1e-3, upper_limit=10.0
    )

    analysis_factor_list.append(
        af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)
    )

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(f"  channels in factor graph:           {len(analysis_factor_list)}")
print(
    f"  global model free parameters:       {factor_graph.global_prior_model.total_free_parameters}"
)

"""
__Search__

Parametric datacube fits have a higher non-linear dimensionality than the pixelization variant (per-channel
intensity adds one parameter per channel) but a much cheaper per-likelihood cost. Tuning depends on your cube;
`n_live=150` is a reasonable starting point.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "datacube",
    name="modeling_parametric",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=20,
    iterations_per_quick_update=50000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Model Fit__

Run the fit. Per-channel cost is much cheaper than `modeling.py` because there is no source-plane inversion,
just a forward Sersic evaluation per channel followed by the Fourier transform.
"""
print(
    """
    The non-linear search has begun running.

    Parametric datacube fits are typically faster than pixelized ones — the per-likelihood cost is dominated
    by the per-channel NUFFT, with no per-channel inversion on top.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Wrap Up__

The result returned by `search.fit` is a list of per-factor results — one entry per channel — each carrying
its own `FitInterferometer` against the maximum-likelihood lens + source-morphology model and the per-channel
intensity. The recovered per-channel intensities should trace the input emission-line spectrum stored in
`dataset/interferometer/datacube/<name>/cube_summary.json`.

For the pixelized variant of this fit (free-form per-channel source reconstruction), see `modeling.py`. For
the narrative walkthrough, see `start_here.py`. For a step-by-step JAX likelihood walkthrough of how the
per-channel log-evidences are summed inside the FactorGraph, see
`autolens_workspace_developer/datacube/likelihood_function.py`.
"""

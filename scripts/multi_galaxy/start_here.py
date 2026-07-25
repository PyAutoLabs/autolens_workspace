"""
Start Here: Multi Galaxy
========================

Multi-galaxy strong lenses have **two or more galaxies of comparable mass which both contribute significantly to
the lensing** of a single background source. Neither galaxy is a minor perturber — they are co-dominant
deflectors, and every one of them gets its own free light and mass model.

This script shows you how to model a multi-galaxy lens system using **PyAutoLens** with as little setup as
possible. In about 15 minutes you'll be able to point the code at your own FITS files and fit your first
multi-galaxy lens.

__Which Regime Is My Lens?__

PyAutoLens organises lenses above the single-galaxy scale into a ladder of three regimes. Every group and cluster
is a multi-galaxy system, but not vice versa — what changes as you climb is first the mass model, then the entire
analysis strategy:

 - **Multi galaxy** (this package): 2+ co-dominant galaxies (individual halos ~10^11-10^13 M_sun), NO shared
   dark-matter halo. One free mass model per deflector + external shear. The source is a single extended galaxy,
   reconstructed at pixel level from CCD imaging — the **standard extended-source workflow**, unchanged
   from `imaging/`.

 - **Group** (`group/start_here.ipynb`): a dominant group-scale halo (~10^13-10^14 M_sun) enters as an *explicit
   modelling choice*, and the galaxies split into tiers (main / extra / scaling galaxies, the latter tidally
   truncated and tied to a luminosity scaling relation). Still one extended source, still the same
   pixel-level `AnalysisImaging` workflow — the sophistication moves into the mass model.

 - **Cluster** (`cluster/start_here.ipynb`): the mass framework is the same as a group's (host halo(s) + many
   truncated members on scaling relations), but the **analysis itself changes**: dozens of sources at different
   redshifts are fitted as point-source multiple-image positions (`AnalysisPoint` + a factor graph, multi-plane
   ray tracing), and the lens galaxies' light is not modeled at all.

If your system has one dominant lens galaxy, start with `imaging/start_here.ipynb` instead.

__The Example System__

We fit a simulated merging pair of lens galaxies modeled on **SDSS J1011+0143** (Shu et al. 2016, ApJ 820, 43,
arXiv:1602.02927) — one of the cleanest co-dominant pairs known: two early-type galaxies separated by ~4.2 kpc
(~0.9") at z=0.331, lensing a z=2.701 Lyman-alpha emitter into a wide Einstein cross of radius ~1.8". The
published model of the real system is exactly the model this script fits: two isothermal mass profiles plus
external shear, with an extended source. Its headline science — kiloparsec-scale offsets between each galaxy's
mass and light, a potential probe of dark-matter self-interaction — is a measurement only a multi-deflector
model can make.

__Contents__

- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **Google Colab Setup:** Run this example in a web browser without local installation.
- **Imports:** Import the required Python libraries.
- **Dataset:** Load (auto-simulating if absent) and plot the strong lens dataset.
- **Main Lens Galaxies:** Every deflector in a multi-galaxy lens is a main lens galaxy.
- **Masking:** Mask the region of the image the model is fitted to.
- **Model:** Compose the lens model — one free light + mass model per deflector.
- **Model Fit:** Perform the model-fit using the search and analysis.
- **Result:** Overview of the results, including the mass/light offset measurement.
- **Model Your Own Lens:** Adapting this script to your own imaging data.
- **Wrap Up:** Summary and the ladder up to groups and clusters.

__JAX__

PyAutoLens runs multi-galaxy model-fits on JAX by default — `al.AnalysisImaging` auto-enables `use_jax=True` if
you installed `autolens[jax]`. The multi-galaxy deflection sum vectorises cleanly, so fits benefit substantially
from GPU acceleration. Expect ~10-20 minutes on a GPU for this example.

For the broader JAX principles, see the top-level `autolens_workspace/start_here.py` `__JAX__` section.

__Google Colab Setup__

The `start_here` examples are runnable on Google Colab without local PyAutoLens installation. The block below
installs the dependencies and downloads the example files if you're on Colab; running it locally is a no-op.
"""

try:
    import google.colab
except ImportError:
    from autolens import setup_colab as _setup_colab
else:
    import importlib
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autonerves", "--no-deps"]
    )
    _setup_colab = importlib.import_module("autonerves.setup_colab")

_setup_colab.for_autolens(
    raise_error_if_not_gpu=False  # Switch to True to require GPU on Colab.
)

"""
__Imports__
"""
from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the multi-galaxy dataset `simple`: HST-resolution (0.05"/pixel) CCD imaging of the simulated merging-pair
lens. If the dataset is not found on disk it is simulated automatically by `multi_galaxy/simulator.py`, so this
script runs with no manual setup.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Galaxies__

For a multi-galaxy lens, two or more galaxies' light and mass all contribute significantly to the lensing of the
source. We call these the "main lens galaxies" and model each one individually — and unlike the group and cluster
regimes, there are no other tiers: no extra galaxies, no scaling relations, no host halo. Every deflector is a
main lens galaxy.

We load the centres of the main lens galaxies from a `.json` file in the dataset folder. These centres are used
to initialize the model for each lens galaxy. For your own data, the centre-input GUI shown in
`group/start_here.ipynb` writes this file from mouse clicks on the image.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Masking__

Lens modeling does not need to fit the entire image, only the region containing the lens and source light. We
define a circular mask around the system — for a multi-galaxy lens make sure it encloses the *combined* Einstein
ring (the lensed arcs wrap around the pair as a whole), not just one galaxy's light.

We also oversample the central pixels of each galaxy, which improves modeling accuracy without adding
unnecessary cost far from the lens.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

To perform lens modeling we define a lens model describing the light and mass of every lens galaxy and the light
of the source galaxy.

The model mirrors the published model of the real SDSS J1011+0143 pair, in the parameterization that works
brilliantly for the vast majority of lenses:

 - Each main lens galaxy's light is a Multi Gaussian Expansion (MGE) — flexible enough to capture the blended
   light of a close pair without adding many free parameters.
 - Each main lens galaxy's mass is a Singular Isothermal Ellipsoid (SIE). These are **untruncated** profiles:
   truncation encodes tidal stripping by a host halo, and a multi-galaxy lens by definition has none. (Truncated
   dPIE members appear when you climb to the group and cluster regimes.)
 - The first lens galaxy carries the system's single `ExternalShear`.
 - The source galaxy's light is an MGE.

__List-Based Model Composition__

Each main lens galaxy is created in a loop over the main lens galaxy centres and stored as `lens_0`, `lens_1`,
etc. This list-based API scales to any number of co-dominant deflectors, and is the same API the group package
uses — so when you climb the ladder there is nothing to re-learn.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

"""
Print the model to see its free parameters — note `lens_0` and `lens_1` each carry their own free mass model,
the signature of the multi-galaxy regime.
"""
print(model.info)

"""
__Model Fit__

We fit the data using the nested sampling algorithm Nautilus and an `AnalysisImaging` object, which defines the
`log_likelihood_function` fitted to the imaging data.

This is the point worth pausing on: `AnalysisImaging` is the **extended-source, pixel-level analysis** — the same
object used for single-galaxy (`imaging/`) and group-scale fits. The multi-galaxy regime changes the mass model,
not the analysis. Only at cluster scale does the analysis itself switch, to `AnalysisPoint` fits of
multiple-image positions.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy"),  # The path where results are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=200,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously.
    iterations_per_full_update=100000,  # Every N iterations results are written to hard-disk.
    live_visual_update=False,  # Set True for a live matplotlib window (script) or refreshing cell (notebook).
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,  # JAX uses GPUs for acceleration if available, else multithreaded CPUs.
)

print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The `output` folder contains many results of the fit in human-readable formats. Here we print the result info
and plot the maximum likelihood tracer and fit.
"""
print(result.info)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Mass/Light Offsets__

Because every deflector has its own free light and mass model, the fit measures each galaxy's mass centre
independently of its light centre. For the real SDSS J1011+0143 pair this measurement revealed offsets of up to
~1.7 kpc (Shu et al. 2016) — interpreted as a signature of the ongoing interaction and a potential test of
self-interacting dark matter. A single-galaxy model cannot make this measurement at all; at group/cluster scale
the scaling-tier members have their mass pinned to their light by construction. It is *the* multi-galaxy science
case.
"""
tracer = result.max_log_likelihood_tracer

for i in range(len(main_lens_centres)):
    galaxy = tracer.galaxies[i]
    print(
        f"lens_{i}:  light centre = {galaxy.bulge.centre}   mass centre = {galaxy.mass.centre}"
    )

"""
__Model Your Own Lens__

If you have your own multi-galaxy lens imaging data, adapt the code above by inputting the paths to your own
.fits files into `Imaging.from_fits()`:

- Supply your own CCD image, PSF, and RMS noise-map.
- Double-check `pixel_scales` for your telescope/detector.
- Adjust the mask radius to enclose the combined Einstein ring and all lens galaxies.
- Provide the centres of all main lens galaxies in a `main_lens_centres.json` file (the GUI in
  `group/start_here.ipynb` writes this from mouse clicks).
- Start with the default model — one MGE + SIE per deflector works very well for pretty much all
  multi-galaxy lenses!

__Wrap Up__

This script has shown how to model a multi-galaxy strong lens: the standard extended-source imaging workflow,
with one free light and mass model per co-dominant deflector.

Where to go next:

- `autolens_workspace/*/multi_galaxy/modeling`: the full modeling API and how to customize the fit.
- `autolens_workspace/*/multi_galaxy/simulator`: how the example dataset was simulated.
- `autolens_workspace/*/multi_galaxy/features`: extensions — extra galaxies, scaling galaxies (untruncated),
  pixelized source reconstructions.
- `autolens_workspace/*/group`: the next rung of the ladder — tiered galaxy populations and the group halo as an
  explicit modelling choice.
- `autolens_workspace/*/cluster`: the top rung — point-source constraints, many sources, multi-plane ray tracing.
- `autolens_workspace/guides/results`: loading and analyzing the results of your fits.
"""

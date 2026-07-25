"""
Multi Galaxy: Modeling
======================

This script models an example strong lens on the 'multi galaxy' scale, where two (or more) galaxies of comparable
mass both contribute significantly to the lensing of a single background source.

The multi-galaxy regime keeps the **standard extended-source analysis workflow**: the data is CCD imaging, the
source is reconstructed at pixel level (parametric Sersic / MGE, or a pixelized mesh) via `AnalysisImaging`, and
the lens galaxies' light is modeled. What changes relative to `imaging/` is only the mass model: **one free light
and mass model per deflector**, because every deflector is co-dominant. This is a deliberate contrast with the
regimes above it on the ladder:

 - `group/`: still an extended-source `AnalysisImaging` fit, but the galaxies split into tiers (main / extra /
   scaling galaxies) and a group-scale dark-matter halo enters as an explicit modelling choice.
 - `cluster/`: the analysis itself changes — many sources at many redshifts are fitted as **point-source multiple
   image positions** via `AnalysisPoint` and a factor graph, with no lens light in the model at all.

All groups and clusters are multi-galaxy systems, but not vice versa: this package is the base rung, where the
only new concept is "more than one main lens galaxy".

This example uses a list-based model composition API, where main lens galaxies are built in a loop over centres
loaded from a JSON file and stored in the model as `lens_0`, `lens_1`, etc. Only the first (`lens_0`) carries an
`ExternalShear`, as the system has one overall external shear. This API scales naturally to any number of
co-dominant deflectors.

__Contents__

- **Example:** The dataset fitted in this example and the model fitted to it.
- **Simulation:** Overview of how the simulated dataset was generated.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Main Lens Galaxies:** In the multi-galaxy regime every deflector is a main lens galaxy.
- **Redshifts:** All deflectors are at the same redshift, so ray tracing is single-plane.
- **Model:** Compose the lens model fitted to the data.
- **Search + Analysis:** Configure the non-linear search and the `AnalysisImaging` object.
- **Run Times:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Mass/Light Offsets:** The science a multi-deflector model uniquely enables.
- **Features:** Extensions of the multi-galaxy model (extra galaxies, scaling galaxies, pixelized sources).

__Example__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - The lens is a pair of co-dominant galaxies, each with its own MGE light model and `Isothermal` mass model.
 - The first lens galaxy also carries an `ExternalShear`.
 - The source galaxy's light is an MGE.

The pair configuration is modeled on **SDSS J1011+0143** (Shu et al. 2016, ApJ 820, 43, arXiv:1602.02927), a
merging pair of early-type galaxies at z=0.331 (projected separation ~4.2 kpc) lensing a z=2.701 Lyman-alpha
emitter into a wide Einstein cross / arc. The published model of that system is exactly the model composed here:
two isothermal mass profiles plus shear, with an extended source.

__Simulation__

This script fits a simulated `Imaging` dataset produced by `autolens_workspace/*/multi_galaxy/simulator.py`. If
the dataset is not found on disk it is simulated automatically before the fit.

__Data Preparation__

The `Imaging` dataset fitted in this example conforms to a number of standards that make it suitable to be fitted
in **PyAutoLens**. If you are intending to fit your own strong lens data, ensure it conforms to these standards,
which are described in `autolens_workspace/*/imaging/data_preparation/start_here.ipynb`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the multi-galaxy strong lens dataset `simple`, which is the dataset we will use to perform lens modeling.

This is loaded via .fits files, which is a data format used by astronomers to store images.

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are
using is 0.05" / pixel (Hubble Space Telescope ACS resolution, matching the data the real SDSS J1011+0143 pair
was modeled with).
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
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

"""
Use an `aplt.subplot_imaging_dataset` to plot the data, noise-map and PSF of the dataset.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Galaxies__

In the group and cluster packages, galaxies split into tiers — main galaxies modeled freely, extra galaxies with
restricted freedom, scaling galaxies tied to a luminosity relation — because there are too many deflectors to free
them all. The multi-galaxy regime needs no tiers: there are only a handful of deflectors and each contributes
comparably to the lensing, so **every galaxy is a main lens galaxy** with its own free light and mass model.

We load the centres of the main lens galaxies from a `.json` file in the dataset folder. These centres initialize
the centre priors of each galaxy's light and mass profiles. For your own data, the `group/start_here.ipynb`
example provides a GUI for clicking the galaxy centres on the image.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

Define a circular mask enclosing the lensed source emission and every lens galaxy. The multi-galaxy Einstein
radius is that of the *combined* mass distribution (~1.8" here), so the mask must comfortably enclose the full
image ring, not just one galaxy's light.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

Apply adaptive over sampling centred on every main lens galaxy, so each galaxy's central light is evaluated
accurately without paying the cost across the whole image.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Redshifts__

Both deflectors are at the same redshift (as in a physically associated pair like SDSS J1011+0143), so ray
tracing is single-plane and the two galaxies' deflection fields simply add.

If your two deflectors are at *different* redshifts, the system is a compound, multi-plane lens (e.g. the
"Einstein zigzag" J1721+8842) — PyAutoLens supports this natively by simply assigning each galaxy its redshift,
and the cluster package documents multi-plane ray tracing in detail.

__Model__

We compose the lens model, with one entry per co-dominant deflector:

 - Each main lens galaxy's light is an MGE (Multi Gaussian Expansion), which captures complex morphologies with
   few free parameters — important here, where the two galaxies' light blends together.
 - Each main lens galaxy's mass is an `Isothermal` (SIE), the standard galaxy-scale profile. Note these are
   **untruncated** profiles: truncation of a galaxy's mass encodes tidal stripping by a host halo's potential,
   and the multi-galaxy regime by definition has no host halo. Truncated (dPIE) profiles enter with the
   group regime's Lenstool-style workflow (`group/features/group_halo`) and are the default at cluster
   scale.
 - The first lens galaxy (`lens_0`) carries the system's single `ExternalShear`.
 - The source galaxy's light is an MGE.

An upgrade path used by published multi-deflector analyses is to swap each `Isothermal` for a `PowerLaw` (EPL),
freeing each galaxy's density slope — a one-line change per galaxy (`al.mp.PowerLaw`), best made after this
simpler model has converged.

__List-Based Model Composition__

Each main lens galaxy is created in a loop over the main lens galaxy centres and stored as `lens_0`, `lens_1`,
etc. — the same list-based API the group package uses, so moving up the ladder later requires no re-learning.
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
Print the model to show the parameters it is composed of. Note how each lens galaxy is listed as `lens_0`,
`lens_1`, etc., each with its own free mass model — the signature of the multi-galaxy regime.
"""
print(model.info)

"""
__Search + Analysis__

We fit the data with the lens model using the nested sampling algorithm Nautilus, via an `AnalysisImaging`
object — the same extended-source analysis used at galaxy and group scale. (The analysis object is where the
regime ladder forks: cluster-scale fits instead use `AnalysisPoint` on multiple-image positions.)

__JAX__

`AnalysisImaging` defaults to `use_jax=True`; the search driver wraps the likelihood in `jax.vmap(jax.jit(...))`.
The multi-galaxy deflection sum vectorises cleanly, so fits benefit substantially from a GPU. Force NumPy with
`use_jax=False` (or `PYAUTO_DISABLE_JAX=1`) when debugging.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy"),
    name="modeling",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_full_update=100000,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

"""
__Run Times__

With two MGE light models, two free mass profiles, a shear and an MGE source, this model has more free parameters
than a single-galaxy fit, and the run time is correspondingly longer: expect ~10-20 minutes on a GPU and an hour
or more on CPU. Each additional co-dominant deflector adds its own light and mass parameters, which is why the
tiered group/cluster APIs exist for systems with many galaxies.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

Print the results and plot the maximum likelihood tracer and fit.
"""
print(result.info)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Mass/Light Offsets__

With two free mass profiles and two free light models, the fit measures each galaxy's mass centre and light
centre *independently*. Comparing them probes the dark-matter physics of the interaction: for the real
SDSS J1011+0143 pair, Shu et al. (2016) measured mass/light offsets of up to ~1.7 kpc, discussed as a potential
test of self-interacting dark matter. This measurement is unique to the multi-galaxy regime — a single-galaxy
model cannot produce it, and at group/cluster scale the member masses are tied to their light by construction.

The code below compares the fitted centres:
"""
tracer = result.max_log_likelihood_tracer

for i in range(len(main_lens_centres)):
    galaxy = tracer.galaxies[i]
    light_centre = galaxy.bulge.centre if hasattr(galaxy.bulge, "centre") else None
    mass_centre = galaxy.mass.centre
    print(f"lens_{i}:  light centre = {light_centre}  mass centre = {mass_centre}")

"""
__Features__

The examples in `autolens_workspace/*/multi_galaxy/features` extend this model:

 - **Extra galaxies**: more distant perturbers added with restricted freedom (fixed centres) — the tier below
   co-dominance.
 - **Scaling galaxies**: a population of faint galaxies far from the lens tied to a luminosity relation, using
   **untruncated** isothermal profiles (no host halo means no tidal truncation; the truncated dPIE
   variant belongs to the group/cluster Lenstool-style workflows).
 - **Pixelized sources**: swap the MGE source for a Delaunay / adaptive mesh reconstruction, exactly as at
   galaxy scale.

__Where To Go Next__

- `autolens_workspace/*/group`: the next rung of the ladder — tiered galaxies and the optional group halo.
- `autolens_workspace/*/cluster`: the top rung — point-source constraints, scaling relations, multi-plane.
- `autolens_workspace/guides/results`: analyzing the results of your fits.
"""

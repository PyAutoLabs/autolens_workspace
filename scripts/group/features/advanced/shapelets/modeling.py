"""
Modeling Features: Shapelets (Group)
====================================

A shapelet is a basis function that is appropriate for capturing the exponential / disk-like features of a galaxy.
It has been employed in many strong lensing studies to model the light of lensed source galaxies, because it can
represent features of disky star forming galaxies that a single Sersic function cannot.

Shapelets are described in full in the following paper:

 https://arxiv.org/abs/astro-ph/0105178

This script performs a group-scale model-fit using shapelets to decompose the source galaxy's light into ~20
shapelet basis functions. The ``intensity`` of every shapelet is solved for via linear algebra.

For group-scale lenses, the main lens galaxies and extra galaxies are modeled with MGE light profiles (which are
more efficient than shapelets for smooth elliptical galaxies), while the source galaxy benefits from the
flexibility of shapelets to capture complex morphology.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Centres:** The centres of the main lens galaxies and extra galaxies are loaded from JSON files.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Model:** Compose the lens model fitted to the data.
**Search:** Configure the non-linear search used to fit the model.
**Analysis:** Create the Analysis object that defines how the model is fitted to the data.
**Result:** Overview of the results of the model-fit.

__Model__

This script fits an ``Imaging`` dataset of a 'group-scale' strong lens where:

 - Each main lens galaxy's light is an MGE bulge.
 - The first main lens galaxy's total mass distribution is an ``Isothermal`` and ``ExternalShear``.
 - There are two extra lens galaxies with MGE light and ``IsothermalSph`` total mass distributions.
 - The source galaxy's light is a superposition of ~20 linear ``ShapeletPolar`` profiles.

__Start Here Notebook__

If any code in this script is unclear, refer to the ``group/modeling`` and
``imaging/features/advanced/shapelets/modeling`` notebooks.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset ``simple``.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "group", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We use a 7.5 arcsecond circular mask for group-scale lenses.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres__

Load the centres of the main lens galaxies and extra galaxies from JSON files.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__

Over sampling at each galaxy centre (both main lens galaxies and extra galaxies).
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

We compose a group lens model where:

 - Main lens galaxies use MGE light profiles (efficient for smooth elliptical galaxies).
 - Extra galaxies use MGE light profiles with fixed centres.
 - The source galaxy uses a shapelet basis decomposition, which captures complex morphology using fewer
   non-linear parameters than a Sersic profile.

The shapelets are composed as a ``Basis`` of ~20 ``ShapeletPolar`` profiles with linked centres, elliptical
components and beta parameters. Only the centre, ellipticity and beta size parameter are non-linear; the
intensity of each shapelet is solved via linear algebra.

__Positive Negative Solver__

Shapelets require the ability to use negative intensities in the linear algebra solution (unlike MGE which
uses positive-only). We therefore use ``use_positive_only_solver=False`` in the analysis settings.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_dict[f"lens_{i}"] = lens

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    # Extra Galaxy Light

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
    )

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source (Shapelet Basis):

total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(al.lp_linear.ShapeletPolar) for _ in range(total_n + total_m + 1)
)

n_count = 1
m_count = -1

for i, shapelet in enumerate(shapelets_bulge_list):
    if i == 0:
        shapelet.n = 0
        shapelet.m = 0

    else:
        shapelet.n = n_count
        shapelet.m = m_count

        m_count += 2

        if m_count > n_count:
            n_count += 1
            m_count = -n_count

    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.ell_comps = shapelets_bulge_list[0].ell_comps
    shapelet.beta = shapelets_bulge_list[0].beta

bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=shapelets_bulge_list,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    extra_galaxies=extra_galaxies,
)

"""
The ``info`` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="shapelets",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
)

"""
__Analysis__

Create the ``AnalysisImaging`` object. We set ``use_positive_only_solver=False`` because shapelets require
negative intensities in the linear solution.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    settings=al.Settings(use_positive_only_solver=False),
    use_jax=True,
)

"""
__Run Time__

The likelihood evaluation time for shapelets is significantly slower than standard light profiles because
every shapelet image must be computed and convolved with the PSF. However, gains are made from the reduced
number of non-linear parameters (the source has only ~3 free parameters: centre, ellipticity, beta).

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result contains entries for each main lens galaxy, the source galaxy (with shapelet basis) and the
extra galaxies.
"""
print(result.info)

print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script shows how to fit a group-scale lens model where the source galaxy light is decomposed into a
shapelet basis. Shapelets capture complex morphology and are useful for irregular star-forming source galaxies,
while the group lens galaxies are efficiently modeled with MGE profiles.
"""

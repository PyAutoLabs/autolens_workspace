"""
Fit Features: Shapelets (Group)
===============================

A shapelet is a basis function that is appropriate for capturing the exponential / disk-like features of a galaxy.
This script demonstrates how to create a fit using shapelet light profiles for a group-scale strong lens, without
performing a non-linear search.

This is useful for understanding the shapelet API and visualizing how shapelets decompose the source galaxy's light
in a group lens context. The lens mass model and galaxy parameters are specified manually using the true values
from the simulation.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Centres:** Load galaxy centres from JSON files.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Basis:** Build a ``Basis`` of multiple linear shapelet light profiles.
**Fit:** Fit the lens model to the dataset using the shapelet basis.
**Intensities:** Extract the solved-for intensity values from the fit.

__Model__

This script fits an ``Imaging`` dataset of a 'group-scale' strong lens where:

 - The main lens galaxy and extra galaxies use the true simulation parameters.
 - The source galaxy's light is a superposition of ~20 linear ``ShapeletPolar`` profiles.

__Start Here Notebook__

If any code in this script is unclear, refer to the ``group/fit`` and
``imaging/features/advanced/shapelets/fit`` notebooks.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt
from autogalaxy.profiles.plot.basis_plots import subplot_image as subplot_basis_image

"""
__Dataset__

Load the strong lens group dataset ``simple``.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if not dataset_path.exists():
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
__Basis__

We build a ``Basis`` of ~20 linear polar shapelet light profiles for the source galaxy. These shapelets:

 - All share the same centre and elliptical components.
 - The size of the shapelet basis is controlled by a ``beta`` parameter.
 - The ``intensity`` of each shapelet is solved via linear algebra.

We use the true source centre (0.0, 0.1) and a reasonable guess for ``beta``.
"""
total_n = 5
total_m = sum(range(2, total_n + 1)) + 1

n_count = 1
m_count = -1

shapelets_bulge_list = []

shapelet_0 = al.lp_linear.ShapeletPolar(
    n=0,
    m=0,
    centre=(0.0, 0.1),
    ell_comps=(0.0, 0.0),
    beta=1.0,
)

shapelets_bulge_list.append(shapelet_0)

for i in range(total_n + total_m):
    shapelet = al.lp_linear.ShapeletPolar(
        n=n_count,
        m=m_count,
        centre=(0.0, 0.1),
        ell_comps=(0.0, 0.0),
        beta=1.0,
    )

    shapelets_bulge_list.append(shapelet)

    m_count += 2

    if m_count > n_count:
        n_count += 1
        m_count = -n_count

bulge = al.lp_basis.Basis(profile_list=shapelets_bulge_list)

"""
__Fit__

We now create a fit using the true group lens galaxy parameters and the shapelet source model.

The main lens galaxy and extra galaxies use the true simulation parameters, while the source galaxy uses
the shapelet basis whose intensities will be solved via linear algebra.

We set ``use_positive_only_solver=False`` because shapelets require negative intensities.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=bulge,
)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    settings=al.Settings(use_positive_only_solver=False),
)

"""
By plotting the fit, we see that the shapelets do a reasonable job at capturing the appearance of the source galaxy
in the group lens context, with faint residuals where the lensed source is located.
"""
aplt.subplot_fit_imaging(fit=fit)

print(fit.log_likelihood)

"""
__Intensities__

The fit contains the solved-for intensity values for each shapelet. These can be extracted from the
``linear_light_profile_intensity_dict``.
"""
source_bulge = fit.tracer.galaxies[-1].bulge

print(
    f"\nIntensity of source galaxy's first shapelet = "
    f"{fit.linear_light_profile_intensity_dict[source_bulge.profile_list[0]]}"
)

"""
A ``Tracer`` where all linear light profiles are replaced with standard light profiles using the solved-for
intensities is accessible from the fit. This can be used for visualization.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

subplot_basis_image(basis=tracer.galaxies[-1].bulge, grid=grid)

"""
__Wrap Up__

This script shows how to fit a group-scale lens using shapelets for the source galaxy light, demonstrating
the ``Basis`` API and how shapelet intensities are solved via linear algebra.
"""

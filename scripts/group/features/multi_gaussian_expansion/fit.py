"""
Fit Features: Multi Gaussian Expansion (Group)
===============================================

This guide shows how to fit data using the `FitImaging` object for group-scale strong lenses, including visualizing
and interpreting its results.

A Multi Gaussian Expansion (MGE) decomposes each galaxy's light into ~10-30+ Gaussians whose intensities are
solved via linear algebra. For group-scale lenses, this means that adding extra galaxies does not increase the
number of non-linear parameters, making MGE the recommended approach for group modeling.

In this example, we use simple `SersicSph` light profiles to create concrete galaxy instances for the fit
demonstration, because specifying concrete MGE instances requires providing intensity and sigma values for
every Gaussian. In practice, MGE light profiles would be used via lens modeling (see the `modeling.py` example),
where the intensities are determined automatically via linear algebra.

__Contents__

**Loading Data:** Load the group-scale strong lens dataset.
**Mask:** Define the 2D mask applied to the dataset for the model-fit.
**Galaxy Centres:** Load centres of main lens galaxies and extra galaxies from JSON files.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Fitting:** Fit the lens model to the dataset and inspect the results.
**Bad Fit:** A bad lens model will show features in the residual-map and chi-squared map.
**Fit Quantities:** The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.
**Figures of Merit:** There are single valued floats which quantify the goodness of fit.
**MGE In Practice:** How MGE would be used in a real fit via modeling.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Loading Data__

We begin by loading the group-scale strong lens dataset `simple` from .fits files.
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

We use a 7.5" circular mask, which is larger than a typical galaxy-scale lens mask because the group-scale
lens has emission spread over a wider area due to the multiple lens galaxies.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.plot_array(array=dataset.data, title="Image Data With Mask Applied")

"""
__Galaxy Centres__

For group-scale lenses we load the centres of the main lens galaxies and extra galaxies from JSON files.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

print(f"Main lens centres: {main_lens_centres}")
print(f"Extra galaxies centres: {extra_galaxies_centres}")

"""
__Over Sampling__

Over sampling at each galaxy centre ensures that the light profiles of every galaxy in the group are
accurately evaluated.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Fitting__

We create a tracer from a collection of light profiles, mass profiles and galaxies.

For this fit demonstration, we use simple `SersicSph` light profiles for the lens galaxies. In a real
analysis, these would be replaced with MGE light profiles determined via lens modeling, which would capture
more complex morphological features.

The combination of light and mass profiles below is the same as those used to generate the simulated dataset.
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
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

"""
We now use a `FitImaging` object to fit this tracer to the dataset.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.plot_array(array=fit.model_data, title="Model Image")

"""
A subplot can be plotted which contains all fit quantities.
"""
aplt.subplot_fit_imaging(fit=fit)

print(fit.log_likelihood)

"""
__Bad Fit__

A bad lens model will show features in the residual-map and chi-squared map. We demonstrate this by
offsetting the main lens galaxy's mass centre.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.2, 0.2), einstein_radius=4.0),
)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(fit.log_likelihood)

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.
"""
print(fit.model_data.slim)
print(fit.residual_map.slim)
print(fit.normalized_residual_map.slim)
print(fit.chi_squared_map.slim)

"""
__Figures of Merit__

There are single valued floats which quantify the goodness of fit.
"""
print(fit.chi_squared)
print(fit.noise_normalization)
print(fit.log_likelihood)

"""
__Plane Quantities__

The `FitImaging` object has specific quantities which break down each image of each plane. For group-scale
lenses, all lens galaxies (main and extra) are at the same redshift and therefore in the same plane.
"""
print(fit.model_images_of_planes_list[0].slim)
print(fit.model_images_of_planes_list[1].slim)

print(fit.subtracted_images_of_planes_list[0].slim)
print(fit.subtracted_images_of_planes_list[1].slim)

"""
__Unmasked Quantities__

The `FitImaging` can also compute the unmasked blurred image of each plane.
"""
print(fit.unmasked_blurred_image.native)
print(fit.unmasked_blurred_image_of_planes_list[0].native)
print(fit.unmasked_blurred_image_of_planes_list[1].native)

"""
__MGE In Practice__

In a real analysis, the light profiles of each galaxy would be MGE models constructed via
`al.model_util.mge_model_from`, whose intensities are solved via linear algebra during model fitting.

The `FitImaging` object works identically with MGE light profiles -- the fit subplot, residual maps,
chi-squared maps, and all other quantities are computed in the same way. The only difference is that the
light profile images are the sum of many Gaussian components rather than a single analytic profile.

After lens modeling with MGE, you would extract the `max_log_likelihood_fit` from the result object:

    fit = result.max_log_likelihood_fit
    aplt.subplot_fit_imaging(fit=fit)

This fit object contains the same quantities demonstrated above, but with MGE light profiles providing
a more accurate representation of each galaxy's light.

See the `modeling.py` example in this folder for the full MGE modeling workflow.
"""

"""
Fin.
"""

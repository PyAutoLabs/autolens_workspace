"""
Fit Features: Linear Light Profiles (Group)
===========================================

This script shows how to fit data using the ``FitImaging`` object for group-scale strong lenses when using
**linear light profiles**, where the ``intensity`` of every light profile is solved via linear algebra rather
than being specified as a parameter.

A group-scale lens differs from a galaxy-scale lens in that there are multiple lens galaxies contributing to the
lensing. In this example, there is a single main lens galaxy and two extra galaxies nearby whose mass contributes
significantly to the ray-tracing.

The key difference from the standard group fit script is that all light profiles use the ``lp_linear`` module
and therefore do not have an ``intensity`` parameter -- it is solved automatically when the ``FitImaging``
object is created.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Galaxy Centres:** Load the centres of the main lens galaxies and extra galaxies from JSON.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Fitting:** Fit the lens model to the dataset using linear light profiles.
**Intensities:** Extract the solved-for intensity values.
**Visualization:** Convert linear light profiles to ordinary profiles for visualization.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset ``simple`` from .fits files.
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

We use a 7.5 arcsecond circular mask.
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

Load the centres of the main lens galaxies and extra galaxies from JSON files.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

print(f"Main lens centres: {main_lens_centres}")
print(f"Extra galaxies centres: {extra_galaxies_centres}")

"""
__Over Sampling__

Over sampling at each galaxy centre ensures that light profiles are accurately evaluated.
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

We now create galaxies using linear light profiles (via the ``lp_linear`` module). Note that no ``intensity``
parameter is specified for any light profile -- it will be solved via linear algebra when the ``FitImaging``
object is created.

All other parameters (centre, effective_radius, sersic_index, etc.) are specified with known values.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.SersicSph(
        centre=(0.0, 0.0), effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.SersicSph(
        centre=(3.5, 2.5), effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.SersicSph(
        centre=(-4.4, -5.0), effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

"""
We now use a ``FitImaging`` object to fit this tracer to the dataset. The fit automatically detects that
the tracer contains linear light profiles and solves for their ``intensity`` values via a linear inversion.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(fit.log_likelihood)

"""
__Intensities__

The fit contains the solved-for intensity values. These are accessible via the
``linear_light_profile_intensity_dict``, which maps each linear light profile to its solved ``intensity``.
"""
lens_bulge = tracer.galaxies[0].bulge
extra_0_bulge = tracer.galaxies[1].bulge
extra_1_bulge = tracer.galaxies[2].bulge
source_bulge = tracer.galaxies[3].bulge

print(fit.linear_light_profile_intensity_dict)

print(
    f"\n Intensity of main lens bulge = {fit.linear_light_profile_intensity_dict[lens_bulge]}"
)
print(
    f"\n Intensity of extra galaxy 0 bulge = {fit.linear_light_profile_intensity_dict[extra_0_bulge]}"
)
print(
    f"\n Intensity of extra galaxy 1 bulge = {fit.linear_light_profile_intensity_dict[extra_1_bulge]}"
)
print(
    f"\n Intensity of source bulge = {fit.linear_light_profile_intensity_dict[source_bulge]}"
)

"""
__Visualization__

Linear light profiles cannot be plotted directly because they do not have an ``intensity`` value until the
inversion is performed. The ``model_obj_linear_light_profiles_to_light_profiles`` property returns a ``Tracer``
where all linear light profiles are replaced with ordinary light profiles using the solved-for ``intensity``
values. This can be used for visualization.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

print(f"Main lens bulge intensity: {tracer.galaxies[0].bulge.intensity}")
print(f"Extra galaxy 0 bulge intensity: {tracer.galaxies[1].bulge.intensity}")
print(f"Extra galaxy 1 bulge intensity: {tracer.galaxies[2].bulge.intensity}")
print(f"Source bulge intensity: {tracer.galaxies[3].bulge.intensity}")

aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Tracer Image")

"""
Fin.
"""

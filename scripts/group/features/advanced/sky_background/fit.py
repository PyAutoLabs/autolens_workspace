"""
Fit Features: Sky Background (Group)
=====================================

The background of an image is the light that is not associated with the strong lens we are interested in. This
script demonstrates how to include the sky background in a fit for a group-scale strong lens, without performing
a non-linear search.

This illustrates the ``DatasetModel`` API for sky background subtraction using standard objects like a ``Galaxy``,
``Tracer`` and ``FitImaging``.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Centres:** Load galaxy centres from JSON files.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Fit:** Demonstrate fitting with a ``DatasetModel`` that includes sky background.

__Model__

This script fits an ``Imaging`` dataset of a 'group-scale' strong lens where:

 - The main lens galaxy and extra galaxies use the true simulation parameters.
 - The sky background is modeled using a ``DatasetModel`` with the true sky level.

__Start Here Notebook__

If any code in this script is unclear, refer to the ``group/fit`` and
``imaging/features/advanced/sky_background/fit`` notebooks.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset ``sky_background``, which has not had the sky background subtracted.
"""
dataset_name = "sky_background"
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
        [
            sys.executable,
            "scripts/group/features/advanced/sky_background/simulator.py",
        ],
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
__Fit__

We create a fit using the true group lens galaxy parameters and a ``DatasetModel`` that includes the true
sky background level of 5.0 electrons per second.

For the galaxies, we use the true parameters from the simulation. The key addition is the ``DatasetModel``
with ``background_sky_level=5.0``, which subtracts the sky from the data during the fit.
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

dataset_model = al.DatasetModel(background_sky_level=5.0)

fit = al.FitImaging(dataset=dataset, tracer=tracer, dataset_model=dataset_model)

"""
By plotting the fit, we see that the sky is subtracted from the data such that the outskirts are zero.
The group galaxies and lensed source are well fitted.
"""
aplt.subplot_fit_imaging(fit=fit)

print(fit.log_likelihood)

"""
__Wrap Up__

This script shows how to include the sky background in a group-scale lens fit using a ``DatasetModel`` object.
"""

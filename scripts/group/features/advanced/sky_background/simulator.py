"""
Simulator: Sky Background (Group)
=================================

This script simulates a group-scale strong lens dataset where the sky background is NOT subtracted from the image.
The sky background is therefore present in the data and must be accounted for during model-fitting.

This is used to demonstrate sky background modeling in the
``group/features/advanced/sky_background/modeling.py`` example.

__Contents__

**Dataset Paths:** The ``dataset_type`` describes the type of data being simulated.
**Grid:** Define the 2d grid of (y,x) coordinates for the simulation.
**Galaxy Centres:** Define the centres of the main lens galaxies and extra galaxies.
**Over Sampling:** Set up the adaptive over-sampling grid.
**Main Lens Galaxies:** The main lens galaxy at the origin.
**Extra Galaxies:** Two companion galaxies near the lens system.
**Source Galaxy:** The source galaxy whose lensed images we simulate.
**Ray Tracing:** Use all galaxies to set up a tracer.
**Dataset:** Simulate and output the dataset.
**Centre JSON Files:** Save the centres as JSON files.

__Model__

This script simulates ``Imaging`` of a 'group-scale' strong lens where:

 - The group consists of one main lens galaxy and two extra galaxies with ``SersicSph`` light and
   ``IsothermalSph`` mass profiles.
 - A single source galaxy with ``SersicCore`` light.
 - The sky background level is 5.0 electrons per second and is NOT subtracted.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is output to ``dataset/group/sky_background``.
"""
dataset_type = "group"
dataset_name = "sky_background"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2d grid of (y,x) coordinates for the simulation.
"""
grid = al.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

Define the centres of the main lens galaxies and extra galaxies.
"""
main_lens_centres = [(0.0, 0.0)]
extra_galaxies_centres = [(3.5, 2.5), (-4.4, -5.0)]

"""
__Over Sampling__

Adaptive oversampling at all galaxy centres.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres + extra_galaxies_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data.

The ``background_sky_level`` is set to 5.0 electrons per second, much higher than the standard 0.1. The
``subtract_background_sky=False`` flag ensures the sky is NOT subtracted, so it remains in the output data.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=5.0,
    add_poisson_noise_to_data=True,
    subtract_background_sky=False,
)

"""
__Main Lens Galaxies__

The main lens galaxy at the origin with spherical Sersic light and isothermal mass.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

main_lens_galaxies = [lens_0]

"""
__Extra Galaxies__

Two companion galaxies near the lens system.
"""
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

extra_galaxies = [extra_galaxy_0, extra_galaxy_1]

"""
__Source Galaxy__

The source galaxy whose lensed images we simulate.
"""
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

"""
__Ray Tracing__

Use all galaxies to set up a tracer.
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + extra_galaxies + [source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Dataset__

Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset.
"""
aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

"""
__Tracer json__

Save the ``Tracer`` in the dataset folder as a .json file.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the centres of the main lens galaxies and extra galaxies as JSON files.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

al.output_to_json(
    obj=al.Grid2DIrregular(extra_galaxies_centres),
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

"""
The dataset can be viewed in the folder ``autolens_workspace/dataset/group/sky_background``.
"""

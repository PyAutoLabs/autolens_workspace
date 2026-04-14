"""
Simulator: No Lens Light (Group)
================================

This script simulates `Imaging` of a 'group-scale' strong lens where none of the lens galaxies have visible
light emission — only their mass profiles contribute to the ray-tracing. The source galaxy still has light.

This is the group-scale analogue of `imaging/features/no_lens_light/simulator.py`. In a group context, "no lens
light" means that **all** main lens galaxies **and** all extra galaxies are modeled with mass profiles only.

This script simulates `Imaging` of a 'group-scale' strong lens where:

 - The group consists of one main lens galaxy and two extra galaxies, all with `IsothermalSph` mass profiles
   and no light profiles.
 - A single source galaxy is observed whose `LightProfile` is a `SersicCore`.

__Contents__

**Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a.
**Grid:** Define the 2d grid of (y,x) coordinates that the lens and source galaxy images are evaluated and.
**Galaxy Centres:** Define the centres of the main lens galaxies and extra galaxies.
**Over Sampling:** Set up over-sampling at the source centre for accurate light profile evaluation.
**Main Lens Galaxies:** The main lens galaxy at the origin, mass only, no light.
**Extra Galaxies:** The two extra galaxies, mass only, no light.
**Source Galaxy:** The source galaxy whose lensed images we simulate.
**Ray Tracing:** Use all galaxies to setup a tracer, which will generate the image for the simulated `Imaging`.
**Dataset:** Load and plot the strong lens dataset.
**Visualize:** Output a subplot of the simulated dataset.
**Tracer json:** Save the `Tracer` in the dataset folder as a .json file.
**Centre JSON Files:** Save the centres of the main lens galaxies and extra galaxies as JSON files.
**Positions:** Solve for the lensed positions of the source galaxy.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They
define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/group/simple__no_lens_light/data.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/group/simple__no_lens_light/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/group/simple__no_lens_light/psf.fits`.
"""
dataset_type = "group"
dataset_name = "simple__no_lens_light"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2d grid of (y,x) coordinates that the lens and source galaxy images are evaluated and therefore simulated
on, via the inputs:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.
"""
grid = al.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

Define the centres of the main lens galaxies and extra galaxies. These are used for over-sampling and are also
output to JSON files so that the modeling scripts can load them.
"""
main_lens_centres = [(0.0, 0.0)]
extra_galaxies_centres = [(3.5, 2.5), (-4.4, -5.0)]

"""
__Over Sampling__

Because no galaxy has a light profile, there is no need for adaptive over-sampling at the galaxy centres. However,
the lensed source light still requires accurate evaluation, so we apply over-sampling at the source centre
(approximately the primary lens centre where the lensed arcs appear).

An adaptive oversampling grid cannot be defined for the lensed source because its light appears in different regions
of the image plane for each dataset. For this reason, we use a cored light profile for the source galaxy (`SersicCore`)
which changes gradually in its central regions, allowing accurate evaluation without requiring heavy oversampling.

We still apply a mild over-sampling at the centre of the image where the lensed arcs are brightest.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxies__

The main lens galaxy is at the origin (0.0, 0.0). It has an isothermal mass profile but **no light profile**.

In the list-based API used by the group modeling scripts, main lens galaxies are stored in a list called
`main_lens_galaxies`, where each galaxy is referred to as `lens_0`, `lens_1`, etc.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

main_lens_galaxies = [lens_0]

"""
__Extra Galaxies__

The two extra galaxies are companion galaxies near the lens system. They have isothermal mass profiles
but **no light profiles**, with centres offset from the origin.

In the list-based API, extra galaxies are stored in a list called `extra_galaxies`.
"""
extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

extra_galaxies = [extra_galaxy_0, extra_galaxy_1]

"""
__Source Galaxy__

The source galaxy whose lensed images we simulate. It uses a cored Sersic profile so that adaptive over-sampling
is not required for the source.
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

Use all galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.

The tracer combines main lens galaxies, extra galaxies and the source galaxy. Because the lens galaxies have
no light profiles, the simulated image contains only the lensed source emission.
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + extra_galaxies + [source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Dataset__

Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Lets plot the simulated `Imaging` dataset before we output it to fits.
"""

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

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""

aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the centres of the main lens galaxies and extra galaxies as JSON files. These are loaded by the group
modeling scripts to set up the lens model (e.g. fixing centres of extra galaxies, defining scaling relations).
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
__Positions__

Solve for the lensed positions of the source galaxy, which are used as input for the group
modeling scripts (e.g. SLaM pipeline) to help the non-linear search converge.
"""
import os

small_datasets = os.environ.pop("PYAUTO_SMALL_DATASETS", None)

solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.1),
    pixel_scale_precision=0.001,
    magnification_threshold=0.01,
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

if small_datasets is not None:
    os.environ["PYAUTO_SMALL_DATASETS"] = small_datasets

al.output_to_json(
    obj=positions,
    file_path=dataset_path / "positions.json",
)

"""
Finished.
"""

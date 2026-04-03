"""
Simulator: Group
================

This script simulates an example strong lens on the 'group' scale, where there is a single primary lens galaxy
and two smaller extra galaxies nearby, whose mass contributes significantly to the ray-tracing and is therefore
included in the strong lens model.

This script simulates `Imaging` of a 'group-scale' strong lens where:

 - The group consists of one main lens galaxy and two extra galaxies whose light distributions are `SersicSph`
 profiles and total mass distributions are `IsothermalSph` profiles.
 - A single source galaxy is observed whose `LightProfile` is a `SersicCore`.

__Main Lens Galaxies vs Extra Galaxies__

For group-scale lens modeling, galaxies are organized into two categories:

 - `main_lens_galaxies`: The primary lens galaxies that dominate the light and mass of the system. These are
   modeled individually with unique parametric light and mass profiles.

 - `extra_galaxies`: Companion galaxies near the lens system that contribute to lensing but are modeled with
   more restrictive assumptions (e.g. fixed centres, scaling relations).

Centres for each category are saved to separate JSON files (`main_lens_centres.json` and
`extra_galaxies_centres.json`) so that the modeling scripts can load them directly.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "group"
dataset_name = "simple"

"""
The path where the dataset will be output.

In this example, this is: `/autolens_workspace/dataset/group/simple`
"""
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

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

For lensing calculations, the high magnification regions of a lensed source galaxy require especially high levels of
over sampling to ensure the lensed images are evaluated accurately.

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

An adaptive oversampling scheme is implemented, evaluating the central regions at (0.0", 0.0") of the light profile at a
resolution of 32x32, transitioning to 8x8 in intermediate areas, and 2x2 in the outskirts. This ensures precise and
accurate image simulation while focusing computational resources on the bright regions that demand higher oversampling.

This adaptive over sampling is also applied at the centre of every other galaxy in the group.

An adaptive oversampling grid cannot be defined for the lensed source because its light appears in different regions of
the image plane for each dataset. For this reason, most workspace examples utilize cored light profiles for the
source galaxy. Cored light profiles change gradually in their central regions, allowing accurate evaluation without
requiring oversampling.

Once you are more experienced, you should read up on over-sampling in more detail via
the `autolens_workspace/*/guides/over_sampling.ipynb` notebook.
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

The main lens galaxy is at the origin (0.0, 0.0). It has a spherical Sersic light profile and an isothermal
mass profile.

In the list-based API used by the group modeling scripts, main lens galaxies are stored in a list called
`main_lens_galaxies`, where each galaxy is referred to as `lens_0`, `lens_1`, etc.
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

The two extra galaxies are companion galaxies near the lens system. They have spherical Sersic light profiles
and isothermal mass profiles, with centres offset from the origin.

In the list-based API, extra galaxies are stored in a list called `extra_galaxies`.
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

The tracer combines main lens galaxies, extra galaxies and the source galaxy.
"""
tracer = al.Tracer(
    galaxies=main_lens_galaxies + extra_galaxies + [source_galaxy]
)

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
Finished.
"""

"""
Simulator: Group
================

This script simulates `Imaging` and a `PointDataset` of a 'group-scale' strong lens where:

 - The group consists of three lens galaxies whose ligth distributions are `SersicSph` profiles and
 total mass distributions are `IsothermalSph` profiles.
 - A single source galaxy is observed whose `LightProfile` is an `Sersic`.

The brightest pixels of the source in the image-plane are used to create a point-source dataset.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "group"
dataset_name = "lens_x3__source_x1"

"""
The path where the dataset will be output, which in this case is:
`/autolens_workspace/output/group`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

For simulating an image of a strong lens, we recommend using a Grid2DIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid2D, but when the light-profile`s image is evaluated below (using the Tracer) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the source galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
grid = al.Grid2DIterate.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 8, 16, 24],
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Ray Tracing__

Setup the mass models of the three lens galaxies using the `IsothermalSph` model and the source galaxy light using 
an elliptical Sersic for this simulated lens.
"""
lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

lens_galaxy_2 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
    point_0=al.ps.Point(centre=(0.0, 0.1)),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy_0, lens_galaxy_1, lens_galaxy_2, source_galaxy]
)

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
__Point Source__

It is common for group-scale strong lens datasets to be modeled assuming that the source is a point-source. Even if 
it isn't, this can be necessary due to computational run-time making it unfeasible to fit the imaging dataset outright.

We will use a `PositionSolver` to locate the multiple images, using computationally slow but robust settings to ensure w
e accurately locate the image-plane positions.
"""
solver = al.PointSolver(
    grid=grid,
    use_upscaling=True,
    pixel_scale_precision=0.001,
    upscale_factor=2,
    magnification_threshold=1.0,
)

"""
We now pass the `Tracer` to the solver. This will then find the image-plane coordinates that map directly to the
source-plane coordinate (0.0", 0.0").
"""
positions = solver.solve(
    lensing_obj=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

positions = al.Grid2DIrregular(
    values=[
        positions.in_list[0],
        positions.in_list[2],
        positions.in_list[3],
        positions.in_list[-1],
    ]
)

"""
Use the positions to compute the magnification of the `Tracer` at every position.
"""
magnifications = tracer.magnification_2d_via_hessian_from(grid=positions)

"""
We can now compute the observed fluxes of the `Point`, give we know how much each is magnified.
"""
flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]
fluxes = al.ArrayIrregular(values=fluxes)

"""
Create a point-source dictionary data object and output this to a `.json` file, which is the format used to load and
analyse the dataset.
"""
point_dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=positions.values_via_value_from(value=grid.pixel_scale),
    fluxes=fluxes,
    fluxes_noise_map=al.ArrayIrregular(values=[1.0, 1.0, 1.0, 1.0]),
)

point_dict = al.PointDict(point_dataset_list=[point_dataset])

point_dict.output_to_json(
    file_path=path.join(dataset_path, "point_dict.json"), overwrite=True
)

"""
__Dataset__

Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Lets plot the simulated `Imaging` dataset before we output it to fits, including the (y,x) coordinates of the multiple
images in the image-plane.
"""
visuals = aplt.Visuals2D(multiple_images=positions)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D()
)
dataset_plotter.subplot_dataset()


"""
Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_plane_images()

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `Tracer.from_json`.
"""
tracer.output_to_json(file_path=path.join(dataset_path, "tracer.json"))

"""
Finished.
"""

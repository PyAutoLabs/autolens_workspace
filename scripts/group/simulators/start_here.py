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

When simulating the amount of emission in each image pixel from the lens and source galaxies, a two dimensional
line integral of all of the emission within the area of that pixel should be performed. However, for complex lens
models this can be difficult to analytically compute and can lead to slow run times.

Instead, an iterative ray-tracing algorithm is used to approximate the line integral. Grids of increasing resolution
are used to evaluate the flux in each pixel from the lens and source galaxies. Grids of higher resolution are used
until the fractional accuracy of the flux in each pixel meets a certain threshold, which we set below to 99.99%

This uses the `OverSamplingIterate` object, which is input into to the `Grid2D` object you may have seen in other 
example scripts, however it make sit perform the iterative ray-tracing described above.
"""
grid = al.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingIterate(
        fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    ),
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
tracer = al.Tracer(
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

A `PointSolver` determines the multiple-images of the mass model for a point source at location (y,x) in the source 
plane. It does this by iteratively ray-tracing light rays from the image-plane to the source-plane, until it finds 
the image-plane coordinate that rays converge at for a given  source-plane (y,x).

For the lens mass model defined above, it computes the multiple images of the source galaxy
at the (y,x) source-plane coordinates (0.0", 0.0") of the point source.

The `PointSolver` requires a starting grid of (y,x) coordinates in the image-plane, which are iteratively traced 
and refined to locate the image-plane coordinates that map directly to the source-plane coordinate.

The `pixel_scale_precision` is the resolution up to which the multiple images are computed. The lower the value, the
longer the calculation, with a value of 0.001 being efficient but more than sufficient for most point-source datasets.

Strong lens mass models have a multiple image called the "central image", which is located at the centre of the lens.
However, the image is nearly always demagnified due to the mass model, and is therefore not observed and not
something we want to be included in the simulated dataset. The `maginification_threshold` removes this image, by
discarding any image with a magnification below the threshold.
"""
grid = al.Grid2D.uniform(
    shape_native=(300, 300),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the `Tracer` to the solver. This will then find the image-plane coordinates that map directly to the
source-plane coordinate (0.0", 0.0").
"""
positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

print(positions)


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
__Point Datasets__

All of the quantities computed above are input into a `PointDataset` object, which collates all information
about the multiple images of a point-source strong lens system.

In this example, it contains the image-plane coordinates of the multiple images, the fluxes of the multiple images,
and their associated noise-map values.

It also contains the name `point_0`, which is a label given to the dataset to indicate that it is a dataset of a single
point-source. This label is important, it is used for lens modeling in order to associate the dataset with the correct
point-source in the model.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=grid.pixel_scale,
    fluxes=fluxes,
    fluxes_noise_map=al.ArrayIrregular(
        values=[np.sqrt(flux) for _ in range(len(fluxes))]
    ),
)

""""
We now output the point dataset to the dataset path as a .json file, which is loaded in the point source modeling
examples.

In this example, there is just one point source dataset. However, for group and cluster strong lenses there
can be many point source datasets in a single dataset, and separate .json files are output for each.
"""
al.output_to_json(
    obj=dataset,
    file_path=path.join(dataset_path, "point_dataset.json"),
)

"""
__Visualize__

Output a subplot of the simulated point source dataset as a .png file.
"""
mat_plot_1d = aplt.MatPlot1D(output=aplt.Output(path=dataset_path, format="png"))
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

point_dataset_plotter = aplt.PointDatasetPlotter(
    dataset=dataset, mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d
)
point_dataset_plotter.subplot_dataset()

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

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=path.join(dataset_path, "tracer.json"),
)

"""
Finished.
"""

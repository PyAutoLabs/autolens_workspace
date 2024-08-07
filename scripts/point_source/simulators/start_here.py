"""
Simulator: Start Here
=====================

This script is the starting point for simulating point source strong lens datasets, for example a lensed quasar
or supernova, and it provides an overview of the lens simulation API.

After reading this script, the `examples` folder provide examples for simulating more complex lenses in different ways.

__Model__

This script simulates `PointDataset` data of a strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a `Point`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `PointDataset` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/positions.json`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.json`.
"""
dataset_type = "point_source"
dataset_name = "simple"

"""
The path where the dataset will be output, which in this case is:
`/autolens_workspace/dataset/positions/simple`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE) and source galaxy (a point source) for this simulated lens. 

We include a faint extended light profile for the source galaxy for visualization purposes, in order to show where 
the multiple images of the lensed source appear in the image-plane.

For lens modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoLens** `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialCore(
        centre=(0.0, 0.0), intensity=0.1, effective_radius=0.02, radius_break=0.025
    ),
    point_0=al.ps.Point(centre=(0.0, 0.0)),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Point Solver__

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
    shape_native=(100, 100),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the `Tracer` to the solver, which calculates the image-plane multiple image coordinates that map directly 
to the source-plane coordinate (0.0", 0.0").
"""
positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

"""
__Fluxes__

The flux of the multiple images are also simulated.

Given a mass model and (y,x) image-plane coordinates, the magnification at that point on the image-plane can be
calculated. 

This is performed below for every multiple image image-plane coordinate, which will be used to simulate the fluxes
of the multiple images.
"""
magnifications = tracer.magnification_2d_via_hessian_from(grid=positions)

"""
To simulate the fluxes, we assume the source galaxy point-source has a total flux of 1.0.

Each observed image has a flux that is the source's flux multiplied by the magnification at that image-plane coordinate.
"""
flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]
fluxes = al.ArrayIrregular(values=fluxes)

"""
__Output__

We now output the image of this strong lens to `.fits` which can be used for visualize when performing point-source 
modeling and to `.png` for general inspection.
"""
visuals = aplt.Visuals2D(multiple_images=positions)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, visuals_2d=visuals)
tracer_plotter.figures_2d(image=True)

mat_plot = aplt.MatPlot2D(
    output=aplt.Output(path=dataset_path, filename="data", format="fits")
)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.figures_2d(image=True)

mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

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

Output a subplot of the simulated point source dataset and the tracer's quantities to the dataset path as .png files.
"""
mat_plot_1d = aplt.MatPlot1D(output=aplt.Output(path=dataset_path, format="png"))
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

point_dataset_plotter = aplt.PointDatasetPlotter(
    dataset=dataset, mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d
)
point_dataset_plotter.subplot_dataset()

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

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

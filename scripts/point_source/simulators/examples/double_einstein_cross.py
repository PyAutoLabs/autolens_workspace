"""
Simulator: Point Source
=======================

This script simulates `PointDataset` data of a strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source `Galaxy` is a `Point`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
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
gives it a descriptive name. 

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/positions.json`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.json`.
"""
dataset_type = "point_source"
dataset_name = "double_einstein_cross"

"""
The path where the dataset will be output, which in this case is:
`/autolens_workspace/dataset/positions/simple`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE+Shear) and source galaxy `Point` for this simulated lens. We include a 
faint dist in the source for purely visualization purposes to show where the multiple images appear.

For lens modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.Isothermal(
        centre=(0.02, 0.03),
        einstein_radius=0.2,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
    ),
    light=al.lp.ExponentialCore(
        centre=(0.02, 0.03), intensity=0.1, effective_radius=0.02
    ),
    point_0=al.ps.Point(centre=(0.02, 0.03)),
)


source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    light=al.lp.ExponentialCore(
        centre=(0.0, 0.0), intensity=0.1, effective_radius=0.02
    ),
    point_1=al.ps.Point(centre=(0.0, 0.0)),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

"""
__Point Solver__

We use a `PointSolver` to locate the multiple images. 
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the `Tracer` to the solver. 

This finds the image-plane coordinates that map directly to the source-plane centres (0.02", 0.03") and (0.0", 0.0").

A double Einstein ring is a multi-plane lensing system, therefore for each source we also input their redshifts into
the solver so that it finds the multiple images properly accounting for the multi-plane lensing.
"""
positions_0 = solver.solve(
    tracer=tracer,
    source_plane_coordinate=source_galaxy_0.point_0.centre,
    source_plane_redshift=source_galaxy_0.redshift,
)

positions_1 = solver.solve(
    tracer=tracer,
    source_plane_coordinate=source_galaxy_1.point_1.centre,
    source_plane_redshift=source_galaxy_1.redshift,
)

"""
Use the positions to compute the magnification of the `Tracer` at every position.
"""
magnifications_0 = tracer.magnification_2d_via_hessian_from(grid=positions_0)
magnifications_1 = tracer.magnification_2d_via_hessian_from(grid=positions_1)

"""
We can now compute the observed fluxes of the `Point`, give we know how much each is magnified.
"""
flux = 1.0
fluxes_0 = [flux * np.abs(magnification) for magnification in magnifications_0]
fluxes_0 = al.ArrayIrregular(values=fluxes_0)
fluxes_1 = [flux * np.abs(magnification) for magnification in magnifications_1]
fluxes_1 = al.ArrayIrregular(values=fluxes_1)

"""
We now output the image of this strong lens to `.fits` which can be used for visualize when performing point-source 
modeling and to `.png` for general inspection.
"""
visuals = aplt.Visuals2D(multiple_images=[positions_0, positions_1])

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

Create a point-source data object and output this to a `.json` file, which is the format used to load and
analyse the dataset.
"""
dataset_0 = al.PointDataset(
    name="point_0",
    positions=positions_0,
    positions_noise_map=al.ArrayIrregular(values=len(positions_0) * [grid.pixel_scale]),
    fluxes=fluxes_0,
    fluxes_noise_map=al.ArrayIrregular(values=[1.0, 1.0, 1.0, 1.0]),
)
dataset_1 = al.PointDataset(
    name="point_1",
    positions=positions_1,
    positions_noise_map=al.ArrayIrregular(values=len(positions_1) * [grid.pixel_scale]),
    fluxes=fluxes_1,
    fluxes_noise_map=al.ArrayIrregular(values=[1.0, 1.0, 1.0, 1.0]),
)


""""
We now output the point datasets to the dataset path as a .json file, which is loaded in the point source modeling
examples.
"""
al.output_to_json(
    obj=dataset_0,
    file_path=path.join(dataset_path, "point_dataset_0.json"),
)

al.output_to_json(
    obj=dataset_1,
    file_path=path.join(dataset_path, "point_dataset_1.json"),
)

"""
__Visualize__

Output a subplot of the simulated point source dictionary and the tracer's quantities to the dataset path as .png files.
"""
mat_plot_1d = aplt.MatPlot1D(output=aplt.Output(path=dataset_path, format="png"))
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

point_dataset_plotter = aplt.PointDatasetPlotter(
    dataset=dataset_0, mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d
)
point_dataset_plotter.subplot_dataset()

point_dataset_plotter = aplt.PointDatasetPlotter(
    dataset=dataset_1, mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d
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

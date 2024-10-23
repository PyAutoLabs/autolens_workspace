"""
Simulator: Debelending
======================

This script simulations a `Point` dataset of a galaxy-scale strong lens which is identical to the dataset
simulated in the `start_here.ipynb` example, but where an image of the multiply imaged lensed point source (e.g.
the quasar) and its lens galaxy are included.

It is used in `autolens_workspace/notebooks/point_source/modeling/features/deblending.ipynb` to illustrate how to
perform deblending of a point source dataset, in order to measure the image-plane multiple image positions, fluxes
and lens galaxy light.

The simulation procedure in this script simulates the lens in two steps:

1) Simulate the point-source dataset, in an identical fashion to the `start_here.ipynb` example.
2) Use this result to simulate the imaging dataset of the lensed point source and lens galaxy.

__Model__

This script simulates `Imaging` and `PointDataset` data of a strong lens where:

 - The lens galaxy's light profile is a `Sersic`.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a `Point`.
 - The multiple images of each lensed point source are `Gaussian` which already represent the PSF convolved images.

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

The path where the dataset will be output.
"""
dataset_type = "point_source"
dataset_name = "deblending"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Ray Tracing (Point Source)__

Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.
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
    point_0=al.ps.Point(centre=(0.0, 0.0)),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

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
We now pass the `Tracer` to the solver. This will then find the image-plane coordinates that map directly to the
source-plane coordinate (0.0", 0.0").
"""
positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)


"""
__Fluxes__

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
__Point Datasets (Point Source)__

Create the `PointDataset`  and `PointDataset` objects using identical code to the `start_here.ipynb` example.
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

al.output_to_json(
    obj=dataset,
    file_path=path.join(dataset_path, "point_dataset.json"),
)


"""
__Visualize (Point Source)__

Visualize the `PointDataset` using identical code to the `start_here.ipynb` example.
"""
mat_plot_1d = aplt.MatPlot1D(output=aplt.Output(path=dataset_path, format="png"))
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

point_dataset_plotter = aplt.PointDatasetPlotter(
    dataset=dataset, mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d
)
point_dataset_plotter.subplot_dataset()

"""
Output subplots of the tracer's images, including the positions of the multiple images on the image.
"""
visuals = aplt.Visuals2D(multiple_images=positions)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d, visuals_2d=visuals
)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

"""
__Tracer json (Point Source)__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=path.join(dataset_path, "tracer_point.json"),
)

"""
__Simulate (Imaging)__

Simulate the image using a `Grid2D` with the `OverSamplingIterate` object.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingIterate(
        fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    ),
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf_sigma = 0.1

psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=psf_sigma, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Lensed Source Image (Imaging)__

The `positions` and `fluxes` above represent the location and brightnesses of the multiple images in the image-plane.

To include these multiple images in the imaging simulation, we add each multiple image individually in the image-plane. 
These multiple images are assumed to have already been convolved with the PSF, which is why they use the `lp_operated` 
profile (see `autolens_workspace/*/notebooks/modeling/imaging/features/advanced/operated_light_profiles.py`).

The `Imaging` simulation procedure therefore does not place a point-source in the source-plane, and use ray-tracing
to determine its image-plane multiple images. It is effectively doing this, because it uses the `positions` and
`fluxes` above to add the multiple images in the image-plane, but the `Tracer` below does not explicitly perform
this ray-tracing calculation.

The reason we choose this approach is because it is closer to how we model the multiple images of actual lensed point 
sources, where each multiple image is modeled in the image-plane as a separate light 
profile (see `point_source/modeling/features/debeleing.ipynb` for a description of why).
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    point_image_0=al.lp_operated.Gaussian(
        centre=positions[0], intensity=fluxes[0], sigma=psf_sigma
    ),
    point_image_1=al.lp_operated.Gaussian(
        centre=positions[1], intensity=fluxes[1], sigma=psf_sigma
    ),
    point_image_2=al.lp_operated.Gaussian(
        centre=positions[2], intensity=fluxes[2], sigma=psf_sigma
    ),
    point_image_3=al.lp_operated.Gaussian(
        centre=positions[3], intensity=fluxes[3], sigma=psf_sigma
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

"""
The source galaxy now long uses a `Point` component as the multiple images are included in the image-plane instead.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
The `Imaging` simulation now uses the normal API for simulating images.
"""

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
__Output__

We now output the image of this strong lens to `.fits` which can be used for visualize when performing point-source 
modeling and to `.png` for general inspection.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.set_filename(filename="subplot_imaging")
dataset_plotter.subplot_dataset()  #
dataset_plotter.set_filename(filename="data")
dataset_plotter.figures_2d(data=True)

visuals = aplt.Visuals2D(multiple_images=positions)

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
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=path.join(dataset_path, "tracer_imaging.json"),
)

"""
Finished.
"""

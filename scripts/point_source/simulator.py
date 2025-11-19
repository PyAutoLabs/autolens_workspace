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

__Pre-requisites__

It is strongly recommended you read the `autolens_workspace/scripts/point_source/start_here` notebook before
running this script, as it gives a full overview of the point source modeling API and how lensing calculations
are performed.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
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
The path where the dataset will be output. 

In this example, this is: `/autolens_workspace/dataset/positions/simple`
"""
dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE) and source galaxy (a point source) for this simulated lens. 

We include a faint extended light profile for the source galaxy for visualization purposes, in order to show where 
the multiple images of the lensed source appear in the image-plane.

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

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialCore(
        centre=(0.07, 0.07), intensity=0.1, effective_radius=0.02, radius_break=0.025
    ),
    point_0=al.ps.Point(centre=(0.07, 0.07)),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Point Solver__

For a point source, our goal is to find the (y, x) coordinates in the image plane that map directly to the center of 
the point source in the source plane—these are its "multiple images." This is achieved using a `PointSolver`, which 
determines the multiple images of the mass model for a point source located at a given (y, x) position in the 
source plane.

The solver works by ray tracing triangles from the image plane back to the source plane and checking whether the 
source-plane (y, x) center lies inside each triangle. It iteratively refines this process by ray tracing progressively 
smaller triangles, allowing the multiple image positions to be determined with sub-pixel precision.

The `PointSolver` requires an initial grid of (y, x) coordinates in the image plane, which defines the first set of 
triangles to ray trace. It also needs a `pixel_scale_precision` parameter, specifying the resolution at which the 
multiple images are computed. Smaller values increase precision but require longer computation times. The value 
of 0.001 used here balances efficiency and accuracy.

Strong lens mass models often predict a "central image," a multiple image that is usually heavily demagnified and thus 
not observed. Since the `PointSolver` finds all valid multiple images, it will locate this central image regardless of 
its visibility. To avoid including this unobservable image, we set a `magnification_threshold=0.1`, which discards any 
images with magnifications below this value.

If your dataset does include a detectable central image, you should lower this threshold accordingly to include it in 
your analysis.

We now compute the multiple image positions by creating a `PointSolver` object and passing it the tracer of our 
strong lens system.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the tracer to the solver, to determine the image-plane multiple images for the source centre.

The solver will find the image-plane coordinates that map directly to the source-plane coordinate (0.07", 0.07").
"""
positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

"""
__Point Datasets__

All the quantities computed above are stored in a `PointDataset` object, which organizes information about the multiple 
images of a point-source strong lens system.

This dataset is labeled with the `name` `point_0`, identifying it as corresponding to a single point source called 
`point_0`. The name is essential for associating the dataset with the correct point source in the lens model during 
fitting.

The dataset contains the image-plane coordinates of the multiple images and their corresponding noise-map values. 
Typically, the noise value for each position is set to the pixel scale of the CCD image, representing the area the 
point source occupies. Although sub-pixel accuracy can be achieved with more detailed analysis, this example does not 
cover those techniques.

Note also that this dataset does not contain fluxes or time delays, which are often included in point source datasets
and are included in a separate simulation below.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=grid.pixel_scale,
)

""""
We now output the point dataset to the dataset path as a .json file, which is loaded in the point source modeling
examples.

In this example, there is just one point source dataset. However, for group and cluster strong lenses there
can be many point source datasets in a single dataset, and separate .json files are output for each.
"""
al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_positions_only.json",
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
Output subplots of the tracer's images, including the positions of the multiple images on the image.
"""
visuals = aplt.Visuals2D(multiple_images=positions)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d, visuals_2d=visuals
)
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
    file_path=dataset_path / "tracer.json",
)

"""
__Imaging__

Point-source data typically comes with imaging data of the strong lens, for example showing the 4 multiply
imaged point-sources (e.g. the quasar images).

Whilst this data may not be used for point-source modeling, it is often used to measure the locations of the point
source multiple images in the first place, and is also useful for visually confirming the images we are using are in 
right place. It may also contain emission from the lens galaxy's light, which can be used to perform point-source 
modeling.

We therefore simulate imaging dataset of this point source and output it to the dataset folder in an `imaging` folder
as .fits and .png files. 

If you are not familiar with the imaging simulator API, checkout the `imaging/simulator.py` example 
in the `autolens_workspace`.
"""
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

imaging_path = dataset_path / "imaging"

mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=imaging_path, format="png"))

imaging_plotter = aplt.ImagingPlotter(
    dataset=imaging, mat_plot_2d=mat_plot_2d, visuals_2d=visuals
)
imaging_plotter.subplot_dataset()

imaging.output_to_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

dataset_plotter = aplt.ImagingPlotter(dataset=imaging, mat_plot_2d=mat_plot_2d)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

"""
__Fluxes__

Another measurable quantity of a point source is its flux—the total amount of light received from each multiple image of 
the point source (e.g., the quasar images).

In practice, fluxes are often measured but not used directly when analyzing lensed point sources such as quasars or 
supernovae. This is because fluxes can be significantly affected by microlensing, which many lens models do not 
accurately capture. However, in this simulation, microlensing is not included, so the fluxes can be simulated and fitted reliably.

We now simulate the fluxes of the multiple images of this point source.

Given a mass model and the (y, x) image-plane coordinates of each image, the magnification at each point can be 
calculated.

Below, we compute the magnification for every multiple image coordinate, which will then be used to simulate their 
fluxes.
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
The noise values of the fluxes are set to the square root of the flux, which is a common given that Poisson noise
is expected to dominate the noise of the fluxes.
"""
fluxes_noise_map = al.ArrayIrregular(values=[np.sqrt(flux) for _ in range(len(fluxes))])

"""
__Point Dataset__

The fluxes are not input a `PointDataset` object, alongside the image-plane coordinates of the multiple images
and their associated noise-map values. 

We again give the dataset the name `point_0`, which is a label given to the dataset to indicate that it is a dataset 
of a single point-source.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=grid.pixel_scale,
    fluxes=fluxes,
    fluxes_noise_map=fluxes_noise_map,
)

""""
We now output the point dataset to the dataset path as a .json file, which is loaded in the point source modeling
examples.

In this example, there is just one point source dataset. However, for group and cluster strong lenses there
can be many point source datasets in a single dataset, and separate .json files are output for each.
"""
al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_with_fluxes.json",
)

"""
__Time Delays__

Another measurable quantity of a point source is its time delay—the time it takes for light to travel from the
source to the observer for each multiple image of the point source (e.g., the quasar images). This is often expressed
as the relative time delay between each image and the image with the shortest time delay, which is often referred to as
the "reference image."

Time delays are commonly used in strong lensing analyses, for example to measure the Hubble constant, since
they are less affected by microlensing and can provide robust cosmological constraints.

We now simulate the same point source dataset, but this time including the time delays of the multiple images.

Given a mass model and (y, x) image-plane coordinates, the time delay at each image-plane position can be
calculated from the mass model. It includes the contribution of both the geometric time delay (the time it takes
different light rays to travel from the source to the observer) and the Shapiro time delay (the time it takes
light to travel through the gravitational potential of the lens galaxy).
"""
time_delays = tracer.time_delays_from(grid=positions)

"""
In real observations, times delays are measured by taking photometric measurements of the multiple images over time,
aligning the light curves, and measuring the time delays between the images.

This processes estimates with it uncertainties, which are often represented as noise-map values in the dataset.
For simplicity, in this simulation we assume the time delays have a noise value which is a quarter of their
measurement value, however it is not typical that the noise value is directly proportional to the time delay.
"""
time_delays_noise_map = al.ArrayIrregular(values=time_delays * 0.25)

"""
__Point Dataset__

The time delays are input into a `PointDataset` object, alongside the image-plane coordinates of the multiple images
and their associated noise-map values. 

We again give the dataset the name `point_0`, which is a label given to the dataset to indicate that it is a dataset 
of a single point-source.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=grid.pixel_scale,
    time_delays=time_delays,
    time_delays_noise_map=time_delays_noise_map,
)

"""
We now output the point dataset to the dataset path as a .json file, which can be loaded in point source modeling
examples.

While this example contains one point source dataset, group and cluster lenses can contain multiple datasets,
with separate .json files saved for each.
"""
al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_with_time_delays.json",
)

"""
We output a final point source dataset containing the positions, fluxes and time delays, which could be used
to perform lens modeling of all measurements simultaneously.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=grid.pixel_scale,
    fluxes=fluxes,
    fluxes_noise_map=fluxes_noise_map,
    time_delays=time_delays,
    time_delays_noise_map=time_delays_noise_map,
)

al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_with_fluxes_and_time_delays.json",
)

"""
Finished.
"""

"""
Guide: Point Sources
--------------------

The examples covered so far have focused on strongly lensed galaxies, where extended surface brightness is warped into
stunning giant arcs and Einstein rings visible in high-quality telescope images. For these extended sources, light
profile objects—such as analytic Sersic profiles—are used to represent their surface brightness.

However, some observed sources are extremely small, spanning just light weeks or days across. In these cases, only the
source’s central point of light is detected in each multiple image. Such sources are called **point sources**,
which typically include quasars, supernovae, or stars.

Strictly speaking, a point source does have a finite size—on the order of light weeks or days—but it is considered a
point source because its size is orders of magnitude smaller than the resolution of the telescope. As a result, it
appears as a single point of light, with all the flux of each multiple image effectively contained within a single pixel.

Point sources affect lensing calculations differently than extended sources, requiring dedicated methods and
functionality. This functionality is described here and used throughout the `point_source` simulation and modeling
examples.

If you are new to analyzing strong lenses with point sources, this guide is the ideal place to start!
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Lensed Point Source__

To begin, we create a strong lens image using an isothermal mass model and a source with a compact exponential light profile.

Although our goal is to demonstrate solving for the multiple image positions of a point source, simulating the data 
with a compact extended source makes the visualization of the point solver’s solutions clearer.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

isothermal_mass_profile = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

exponential_light_profile = al.lp.ExponentialCore(
    centre=(0.07, 0.07), intensity=0.1, effective_radius=0.1
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=isothermal_mass_profile,
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=exponential_light_profile,
)

tracer_extended = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
We plot the image of our strongly lensed source galaxy.

Clearly visible are four multiple images arranged in a cross configuration. The brightest pixels correspond to the 
four (y, x) multiple image positions that our point source solver aims to identify.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer_extended, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
__Point Source__

The image above visually illustrates where the source’s light is traced on the image plane.

We now treat this source as a point source by defining a source galaxy using the `Point` class.

This point source shares the same center as the compact source above, ensuring that the multiple image positions 
coincide with those previously shown in the image plane.
"""
point_source = al.ps.Point(centre=(0.07, 0.07))

source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

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
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the tracer to the solver, to determine the image-plane multiple images for the source centre.

The solver will find the image-plane coordinates that map directly to the source-plane coordinate (0.07", 0.07"), 
which we plot below.

The plot shows the four solved multiple image positions (with the central image excluded) as a scatter plot. To make 
the positions clearer, we increase the marker size and use asterisks—PyAutoLens’s standard symbol for denoting 
multiple images of strong lenses.
"""
positions = solver.solve(tracer=tracer, source_plane_coordinate=(0.07, 0.07))

grid_plotter = aplt.Grid2DPlotter(
    grid=positions,
    mat_plot_2d=aplt.MatPlot2D(grid_scatter=aplt.GridScatter(s=100, marker="*")),
)
grid_plotter.figure_2d()

"""
The plot above makes it difficult to directly compare the multiple image positions with the image of the strong lens itself.

To improve clarity, we overplot the multiple image positions on the strong lens image. This clearly shows that the 
multiple images coincide with the centers of the brightest pixels of the lensed source galaxy.
"""
visuals = aplt.Visuals2D(multiple_images=positions)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_extended, grid=grid, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
__Number of Solutions__

The number of solutions (e.g. the number of image-plane multiple images that map to the source centre) depends
on the mass model of the lens: 

 - For spherical mass profiles, there are three unique solutions, including a demagnified central image.

 - For elliptical mass profiles, there are five unique solutions, again including a demagnified central image.

 - For lenses with multiple mass profiles (e.g. two galaxies) and more exotic mass distributions, the number of 
   solutions can be even higher. 

__Solving the Lens Equation__

In the literature, the process of finding the multiple images of a source in the image-plane is often referred to as
'solving the lens equation'.

There lens equation is a fundamental equation in lensing, which describes how light rays are deflected from the
image-plane to the source-plane. It is given by:

$\beta = \theta - \hat{\alpha}(\theta)$

Where:

$\beta$ is the source-plane (y,x) coordinate.
$\theta$ is the image-plane (y,x) coordinate.
$\hat{\alpha}(\theta)$ is the deflection angle at image-plane (y,x) coordinate $\theta$.

The lens equation is non-linear, as the deflection angle $\hat{\alpha}$ depends on the mass model of the lens galaxy.

It is therefore called solving the lens equation because we are trying to find the image-plane (y,x) coordinates $\theta$
that satisfies the equation above for a given source-plane (y,x) coordinate $\beta$.

__Triangle Tracing__

Computing the multiple image positions of a point source is a non-linear problem. Given a source-plane (y,x) coordinate,
there are multiple image-plane (y,x) coordinates that trace to that source-plane coordinate, and there is no simple
analytic solution to determine these image-plane coordinates.

The solver therefore uses a triangulation approach to find the multiple image positions. It first overlays a grid of
triangles over the image-plane, and uses the mass model to trace these triangles to the source-plane. If a triangle
contains the source-plane (y,x) coordinate, it is retained and its image-plane coordinates are assigned as a multiple
image of the source.

We require the grid of triangles to be fine enough such that the source-plane (y,x) coordinate is contained within
one of the triangles to a sufficient precision for our science case. This is controlled by the `pixel_scale_precision`
input, which sets the target pixel scale of the grid. 

Triangles of iteratively finer resolution are created until this precision is met, therefore a lower value of
`pixel_scale_precision` will lead to a more precise estimate of the multiple image positions at the expense of
increased computational overhead.

Here is a visualization of the triangulation approach:

[CODE]

__Dataset__

We first create a `PointDataset` object, which is similar to an `Imaging` or `Interferometer` object but contains the
positions of the multiple images of the point source and their noise-map values. The noise values are the pixel-scale
of the data, as this is the uncertainty of where we measure the multiple images in the image.

We manually specify the positions of the multiple images below, which correspond to the multiple images of the
isothermal mass model used above.

The demagnified central image is not included in the dataset, as it is not observed in the image-plane. This is
standard practice in point-source modeling.

It also contains the name `point_0`, which is an important label, as explained in more detail below.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=al.Grid2DIrregular(
        [
            [-1.03884121e00, -1.03906250e00],
            [4.41972024e-01, 1.60859375e00],
            [1.17899573e00, 1.17890625e00],
            [1.60930210e00, 4.41406250e-01],
        ],
    ),
    positions_noise_map=al.ArrayIrregular([0.05, 0.05, 0.05, 0.05]),
)

"""
We can print this dictionary to see the dataset's `name`, `positions` and `fluxes` and noise-map values.
"""
print("Point Dataset Info:")
print(dataset.info)

"""
The positions can be plotted over the observed image, to make sure they overlap with the multiple images we expect.
"""
visuals = aplt.Visuals2D(positions=dataset.positions)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_extended, grid=grid, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
__Name Pairing__

The names of the point-source datasets have an even more important role, the names are used to pair each dataset to the
point sources in the lens model used to fit it.

For example, when creating the tracer at the beginning of this script, we named the point source `point_0`:

point_source = al.ps.Point(centre=(0.07, 0.07))
source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

When we fit the point source dataset using this tracer, the name is again used in order to pair the dataset to the
this point source. This means that point source with a centre of (0.07", 0.07") is used to fit the dataset with the
name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` an error will be raised.

In this example, where there is just one source, name pairing is redundant. However, point-source datasets may
have many source galaxies in them, and name pairing allows us to extend the point-source modeling to systems with
many point sources.

__Fitting__

Just like we used a `Tracer` to fit imaging and interferometer data, we can use it to fit point-source data via the
`FitPoint` object.

The name pairing described above is used internally into the `FitPointDataset` object to ensure that the correct point
source is fitted to each dataset. 

The fit is returned as a dictionary which mirrors the `PointDataset`, where its keys are again the names of the datasets.
"""
fit = al.FitPointDataset(
    dataset=dataset,
    tracer=tracer,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,  # This input is describe below
)

print(fit.positions.residual_map)
print(fit.positions.normalized_residual_map)
print(fit.positions.chi_squared_map)
print(fit.positions.log_likelihood)

"""
__Chi Squared__

For point-source modeling, there are many different ways to define the likelihood function, broadly referred to a
an `image-plane chi-squared` or `source-plane chi-squared`. This determines whether the multiple images of the point
source are used to compute the likelihood in the source-plane or image-plane.

The default settings used above use the image-plane chi-squared, which uses the `PointSolver` to determine the 
multiple images of the point source in the image-plane for the given mass model and compares the positions of these 
model images to the observed images to compute the chi-squared and likelihood.

There are still many different ways the image-plane chi-squared can be computed, for example do we allow for 
repeat image-pairs (i.e. the same multiple image being observed multiple times)? Do we pair all possible combinations
of multiple images to observed images? This default settings use the simplest approach, which pair each multiple image
with the observed image that is closest to it, allowing for repeat image pairs. 

For example, we can repeat the fit above whilst not allowing for repeat image pairs as follows:
"""
fit = al.FitPointDataset(
    dataset=dataset,
    tracer=tracer,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePair,  # Different input to the one used above
)

print(
    "Minimum Distance Between Observed Multiple Images and Model Multiple Images Without Repeats:"
)
print(fit.positions.residual_map)

print("Log Likelihood Without Repeats:")
print(fit.positions.log_likelihood)

"""
We can allow for repeat image pairs by using the `FitPositionsImagePairRepeat` class, which is the default input.
"""
fit = al.FitPointDataset(
    dataset=dataset,
    tracer=tracer,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,  # Different input to the one used above
)

print(
    "Minimum Distance Between Observed Multiple Images and Model Multiple Images With Repeats:"
)
print(fit.positions.residual_map)

print("Log Likelihood With Repeats:")
print(fit.positions.log_likelihood)

"""
For a "source-plane chi-squared", the likelihood is computed in the source-plane. The analysis is simpler, it ray-traces
the multiple images back to the source-plane and defines a chi-squared metric. For example, the default implementation 
sums the Euclidean distance between the image positions and the point source centre in the source-plane.

The source-plane chi-squared is significantly faster to compute than the image-plane chi-squared, however it is 
less robust than the image-plane chi-squared and can lead to biased lens model results. 

Here is an example of how to use the source-plane chi-squared:
"""
fit = al.FitPointDataset(
    dataset=dataset,
    tracer=tracer,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,  # Different input to the one used above
)

print(
    "Minimum Distance Between Source Plane Centre and Model Source Plane Images After Ray Tracing:"
)
print(fit.positions.residual_map)

print("Log Likelihood in the Source Plane:")
print(fit.positions.log_likelihood)

"""
__Fluxes__

Another measurable quantity of a point source is its flux—the total amount of light received from each multiple image 
of the point source (e.g., the quasar images).

In practice, fluxes are often measured but not used directly when analyzing lensed point sources such as quasars or 
supernovae. This is because fluxes can be significantly affected by microlensing, which many lens models do not 
accurately capture. However, in this simulation, microlensing is not included, so the fluxes can be simulated and 
fitted reliably.

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
__Flux Point Dataset__

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

"""
__Flux Fitting__

Above, we used a `FitPointDataset` to fit the positions of the point source in the image-plane.

We can also use it to fit the fluxes of the point source, which is done by passing the new dataset also containing
the `fluxes` and `fluxes_noise_map` to the fit.

To fit fluxes, our model point source also needs a flux parameter, which is done by using the `PointFlux`
component instead of the `Point` component. 
"""
point = al.ps.PointFlux(centre=(0.07, 0.07), flux=1.0)

source_galaxy = al.Galaxy(redshift=1.0, point_0=point)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitPointDataset(
    dataset=dataset,
    tracer=tracer,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,  # This input is describe below
)

"""
The fit now contains both a `positions` and `fluxes` attribute, which contain the fit of the positions and fluxes
of the point source.
"""
print(fit.positions.residual_map)
print(fit.positions.normalized_residual_map)
print(fit.positions.chi_squared_map)
print(fit.positions.log_likelihood)

print(fit.flux.residual_map)
print(fit.flux.normalized_residual_map)
print(fit.flux.chi_squared_map)
print(fit.flux.log_likelihood)

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
__Time Delay Fitting__

We can also use the `FitPointDataset` to fit the time delays of the point source, which is done by passing the new
dataset also containing the `time_delays` and `time_delays_noise_map` to the fit.

To fit time delays, the model point source does not need any special parameters (like it did for flux fitting),
so we can revert back to the normal `Point` component.
"""
point = al.ps.Point(centre=(0.07, 0.07))

source_galaxy = al.Galaxy(redshift=1.0, point_0=point)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitPointDataset(
    dataset=dataset,
    tracer=tracer,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,  # This input is describe below
)

"""
The fit now contains both a `positions` and `time_delays` attribute, which contain the fit of the positions and fluxes
of the point source.
"""
print(fit.positions.residual_map)
print(fit.positions.normalized_residual_map)
print(fit.positions.chi_squared_map)
print(fit.positions.log_likelihood)

print(fit.time_delays.residual_map)
print(fit.time_delays.normalized_residual_map)
print(fit.time_delays.chi_squared_map)
print(fit.time_delays.log_likelihood)

"""
Checkout the guide `autolens_workspace/*/guides/point_source.py` for more details and a full illustration of the
different ways the chi-squared can be computed.

__New User Wrap Up__

The `point_source` package of the `autolens_workspace` contains numerous example scripts for performing point source
modeling. These focus on "galaxy scale" lenses, which are lenses that have a single lens galaxy, as opposed to
"group scale" or "cluster scale" lenses which have multiple lens galaxies.

Point source modeling is at the heart of group and cluster scale lens modeling, and is the topic of the
next overview script.

__Shape Solver__

All calculations above assumed the source was a point source with no size. 

This was built into the point-solver, for example when we solved for the multiple images of the point source in the 
image-plane, we ray-traced triangles to the source-plane and asked whether the source-plane (y,x) centre was within 
the triangle.

There is functionality to include the size and shape of the source in the calculation, which uses the `ShapeSolver`
class. This still traces triangles, but each iteration of the solver now computes the area of each image-plane triangle 
that is within the source-plane shape. This means we can determine the area in the image-plane that maps within an 
extended region of the source-plane shape.

For example, by inputting the shape `Circle` with a radius of 0.001", the shape solver will determine the area of the 
multiple images pixel which fall within this circle, which is different information to the point solver which told
us the exact (y,x) coordinates of the multiple images.

The ratio of the total image pixel area to the area within the source-plane 
circle is the magnification factor of the source. This magnification factor then changes the observed flux of each 
multiple image.

Observations we might think are fully in the point source regime therefore may have an observable signature of the size
of the source in the flux ratios and magnifications of the multiple images. Therefore, sometimes the source size 
is large enough that it is important we account for it.
"""
solver = al.ShapeSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

solver.find_magnification(tracer=tracer, shape=al.Circle(x=0.0, y=0.0, radius=0.001))

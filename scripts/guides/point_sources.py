"""
Guide: Point Sources
--------------------

The examples covered so far have shown strongly lensed galaxies, whose extended surface brightness is lensed into
the awe-inspiring giant arcs and Einstein rings we see in high quality telescope imaging. For extended sources,
light profile objects are used to represent their surface brightness as analytic functions (e.g. Sersic profiles).

However, the observed source could be extremely small, for example just light weeks or days across. In this case,
we effectively only observe the source's central point of light in each multiple image. In this case, we refer to the
source as a 'point source', which will typically be a quasar, supernova or star.

Stricly speaking, a point source does have a size and is not a point, it has a diameter some light weeks or light days.
We consider it a point source because its size is orders of magnitude smaller than the resolution of the telescope that
it is observed using, such that it appears as a point of light (e.g. all of the flux of each multiple image is
contained within a single pixel).

Point sources radially change the lensing calculations, meaning they require their own dedicated functionality.
This functionality is described here, and is then used in the `point_source` simulation and modeling examples.

If you are a new user who will be analysing point sources strong lenses, this guide is the right starting point!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Lensed Point Source__

To begin, we create an image of strong lens using a isothermal mass model and source with a compact exponential light 
profile. 

Although our aim is to illustrate solving for the multiple image positions of a point source, by simulating the data 
with a compact extended source visualization of the point solver's solutions will be clearer.
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

We can clearly see there are four multiple images located in a cross configuration. Their brightest pixels are the 
four (y,x) multiple image coordinates our point source multiple image position solver should find.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer_extended, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
__Point Source__

The image above visually illustrates where the source's light traces in the image-plane. 

Lets now treat this source as a point source, by setting up a source galaxy using the `Point` class. 

It has the same centre as the compact source above, to ensure the multiple image positions are located at the same
locations in the image-plane.
"""
point_source = al.ps.Point(centre=(0.07, 0.07))

source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Multiple Image Solving__

For a point source, our goal is to find the (y,x) coordinates in the image-plane that directly map to the centre
of the point source in the source plane, its "multiple images". This uses a `PointSolver`, which determines the 
multiple-images of the mass model for a point source at location (y,x) in the source plane. 

It does this by ray tracing triangles from the image-plane to the source-plane and calculating if the 
source-plane (y,x) centre is inside the triangle. The method gradually ray-traces smaller and smaller triangles so 
that the multiple images can be determine with sub-pixel precision.

The `PointSolver` requires a starting grid of (y,x) coordinates in the image-plane which defines the first set
of triangles that are ray-traced to the source-plane. It also requires that a `pixel_scale_precision` is input, 
which is the resolution up to which the multiple images are computed. The lower the `pixel_scale_precision`, the
longer the calculation, with the value of 0.001 below balancing efficiency with precision.

Strong lens mass models have a multiple image called the "central image". However, the image is nearly always 
significantly demagnified, meaning that it is not observed and cannot constrain the lens model. As this image is a
valid multiple image, the `PointSolver` will locate it irrespective of whether its so demagnified it is not observed.
To ensure this does not occur, we set a `magnification_threshold=0.1`, which discards this image because its
magnification will be well below this threshold.

If your dataset contains a central image that is observed you should reduce to include it in
the analysis.

which we compute below by creating a
`PointSolver` object and passing it the tracer of our strong lens system.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now pass the tracer to the solver. 

This will then find the image-plane coordinates that map directly to the source-plane coordinate (0.07", 0.07"), 
which we plot below.

The plot is the 4 solved for multiple image positions (with the central image removed) on a scatter plot. To make 
the positions clearer, we increase the size of the markers to ensure they are visible and plot them as asterisks, 
which is the standard symbol used to denote multiple images of strong lenses in PyAutoLens.
"""
positions = solver.solve(tracer=tracer, source_plane_coordinate=(0.07, 0.07))

grid_plotter = aplt.Grid2DPlotter(
    grid=positions,
    mat_plot_2d=aplt.MatPlot2D(grid_scatter=aplt.GridScatter(s=100, marker="*")),
)
grid_plotter.figure_2d()

"""
The plot above makes it difficult to compare the multiple image positions to the image of the strong lens itself.

We can therefore overplot the multiple image positions on the image of the strong lens, which clearly shows that the
multiple images trace the centre of the brightest pixels of the lensed source galaxy.
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
Checkout the guide `autolens_workspace/*/guides/point_source.py` for more details and a full illustration of the
different ways the chi-squared can be computed.

__New User Wrap Up__

The `point_source` package of the `autolens_workspace` contains numerous example scripts for performing point source
modeling. These focus on "galaxy scale" lenses, which are lenses that have a single lens galaxy, as opposed to
"group scale" or "cluster scale" lenses which have multiple lens galaxies.

Point source modeling is at the heart of group and cluster scale lens modeling, and is the topic of the
next overview script.

__Fluxes and Time Delays__

The point-source dataset can also include the fluxes and time-delays of each multiple image. 

This information can be computed for a lens model via the `PointSolver`, and used in modeling to constrain the 
lens model.

A full description of how to include this information in the model-fit is given in 
the `autolens_workspace/*/guides/point_source.py` and 
the `autolens_workspace/*/point_sources/modeling/features/fluxes_and_time_delays.py` example script.

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

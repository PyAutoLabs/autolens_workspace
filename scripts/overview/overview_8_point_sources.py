"""
Overview: Point Sources
-----------------------

So far, overview examples have shown strongly lensed galaxies, whose extended surface brightness is lensed into
the awe-inspiring giant arcs and Einstein rings we see in high quality telescope imaging. There are many lenses where
the background source is not extended but is instead a point-source, for example strongly lensed quasars and supernovae.

For these objects, it is invalid to model the source using light profiles, which implicitly assume an extended
surface brightness distribution. Instead, we assume that the source is a point source with a centre (y,x).

Our ray-tracing calculations no longer trace extended light rays from the source plane to the image-plane, but
instead find the locations the point-source's multiple images appear in the image-plane.

Finding the multiple images of a mass model given a (y,x) coordinate in the source plane is an iterative problem
performed differently to ray-tracing a light profile. This example introduces the `MultipleImageSolver` object, which
finds the image-plane multiple images of a point source and makes the analysis of strong lensed quasars, supernovae and
other point-like source's possible.

We also show how these tools can compute the fluxes and time delays of the point-source.
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
__Position Solving__

For a point source, our goal is to find the (y,x) coordinates in the image-plane that directly map to the centre
of the point source in the source plane. 

The number of solutions (e.g. the number of image-plane positions that map directly to the source-plane centre) depends
on the mass model of the lens: 

 - For spherical mass profiles, there are three unique solutions, where one is a demagnified central image that is 
 typically so demagnified it is not observed in imaging data. 
 
 - For elliptical mass profiles, there are typically 5 solutions, again including a demagnified central image.
 
 - For lenses with multiple mass profiles (e.g. two galaxies) and more exotic mass distributions, the number of 
   solutions can be even higher. 

This example uses an elliptical mass profile, so we should expect 5 solutions, which we compute below by creating a
`MultipleImageSolver` object and passing it the tracer of our strong lens system.
"""
solver = al.MultipleImageSolver(
    lensing_obj=tracer,
    grid=grid,
    pixel_scale_precision=0.001,
)

"""
We now pass the tracer to the solver. 

This will then find the image-plane coordinates that map directly to the source-plane coordinate (0.07", 0.07"), 
which we plot below.

The plot is simply all 5 solved for positions on a scatter plot. To make the positions clearer, we increase
the size of the markers to ensure they are visible and plot them as asterisks, which is the standard symbol used
to denote multiple images of strong lenses in PyAutoLens.
"""
positions = solver.solve(source_plane_coordinate=(0.07, 0.07))

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

__How Does The Multiple Image Solver Work?__

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

A full description of the triangulation approach alongside more customizations options can be found in the
guide `autolens_workspace/notebooks/guides/multiple_image_solver.ipynb`.

__Extended Source__

This explains why extended light profiles are not used to model point sources. 

When we simulate a lensed source using a light profile, its multiple images are visible as the brightest pixels in
the image. 

However, we cannot make a statement about the exact location of each multiple image to a greater precision
than the pixel scale of the data we observe each multiple image on. For the data we simulated above, this pixel scale
is 0.05".

The multiple image solver can determine the positions of the multiple images to much higher precision than this, for
example the `pixel_scale_precision` of 0.001" we used above. 

This is why point sources are treated separately, using their own dedicated `MultipleImageSolver` object.

__Modeling__

We can perform lens modeling using point sources using an analogous API to that used for imaging and interferometer
datasets. 

This modeling is appropriate for strongly lensed quasars or supernovae.

We first create a `PointDataset` object, which is similar to an `Imaging` or `Interferometer` object but contains the
positions of the multiple images of the point source and their noise-map values. The noise values are the pixel-scale
of the data, as this is the uncertainty of where we measure the multiple images in the image.

We manually specify the positions of the multiple images below, which correspond to the multiple images of the
isothermal mass model used above.

The demagnified central image is not included in the dataset, as it is not observed in the image-plane. This is
standard practice in point-source modeling.

It also contains the name `point_0`, which is an important label, as explained in more detail below.
"""
point_dataset = al.PointDataset(
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
The positions can be plotted over the observed image, to make sure they overlap with the multiple images we expect.
"""
visuals = aplt.Visuals2D(positions=point_dataset.positions)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_extended, grid=grid, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
__Point Source Dictionary__

We input this `PointDataset` into a `PointDict`, which is a dictionary containing the dataset. This is the object 
used in the `modeling` scripts to perform lens modeling.

In this example only one `PointDataset` is input into the `PointDict`, therefore the point dictionary seems somewhat
redundant. 

However, for datasets where the multiple images of multiple different point sources are observed in the strong lens
system, each will have their own unique `PointDataset`, which are all stored in the `PointDict`.

This occurs in group and cluster scale strong lenses, and very rare and exotic galaxy scale strong lenses.
"""
point_dict = al.PointDict(point_dataset_list=[point_dataset])

"""
We can print the `positions` of this dictionary and dataset, as well as their noise-map values.

The key of the point dictionary we print corresponds to the name of the dataset, which was specified above as `point_0`.
If there are multiple datasets in the `PointDict`, they would have different names, with their results accessible via
using different keys in the dictionary corresponding to these names.
"""
print("Point Source Dataset Name:")
print(point_dict["point_0"].name)
print("Point Source Multiple Image (y,x) Arc-second Coordinates:")
print(point_dict["point_0"].positions.in_list)
print("Point Source Multiple Image Noise-map Values:")
print(point_dict["point_0"].positions_noise_map.in_list)

"""
__Name Pairing__

The names of the point-source datasets have an even more important role, the names are used to pair each dataset to the
point sources in the lens model used to fit it.

For example, when creating the tracer at the beginning of this script, we named the point source `point_0`:

point_source = al.ps.PointSourceChi(centre=(0.07, 0.07))
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

The name pairing described above is used internally into the `FitPointDict` object to ensure that the correct point
source is fitted to each dataset. 

The fit is returned as a dictionary which mirrors the `PointDict`, where its keys are again the names of the datasets.
"""
fit = al.FitPointDict(point_dict=point_dict, tracer=tracer, solver=solver)

print(fit["point_0"].positions.residual_map)
print(fit["point_0"].positions.normalized_residual_map)
print(fit["point_0"].positions.chi_squared_map)
print(fit["point_0"].positions.log_likelihood)

"""
__Model__

It is straight forward to fit a lens model to a point source dataset, using the same API that we saw for dataset and
interferometer datasets.

This uses an `AnalysisPoint` object which fits the lens model in the correct way for a point source dataset.
This includes mapping the `name`'s of each dataset in the `PointDict` to the names of the point sources in
the lens model.
"""
# Lens:

bulge = af.Model(al.lp.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

point_0 = af.Model(al.ps.Point)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

# Overall Lens Model:

galaxies = af.Collection(lens=lens, source=source)
model = af.Collection(galaxies=galaxies)

# Search + Analysis + Model-Fit

search = af.Nautilus(path_prefix="overview", name="point_source")

analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

result = search.fit(model=model, analysis=analysis)

"""
__Source Plane Chi Squared__

This example above performs point-source fitting and modeling using what is often referred to as an "image-plane 
chi-squared". This means that the multiple image positions predicted by the model are computed in the image-plane
and compared to the observed multiple image positions in the image-plane.

An alternative approach is to perform point-source modeling using a "source-plane chi-squared". This means that the
multiple image positions predicted by the model are computed and compared in the source-plane.

This is often regarded as a less robust way to perform point-source modeling, which has been shown to produce biased
and incorrect lens models. This is because a mass model may predict multiple images that are not observed in the 
image-plane, which the source-plane chi-squared calculation fails to properly account for. 

However, a source-plane chi-squared is much faster to compute than an image-plane chi-squared, as it does not require
triangles to be iteratively traced to the source-plane to find the multiple images. In certain science cases, it is
therefore useful, and therefore supported by **PyAutoLens**. 

The advanced search chaining feature of PyAutoLens allows one initially fit a model quickly using a 
source-plane chi-squared and then switch to an image-plane chi-squared for a robust final lens model.

Using a source-plane chi-squared requires us to simply change the point source model input into the source
galaxy to a `PointSourceChi` object
"""
point_source = al.ps.PointSourceChi(centre=(0.07, 0.07))

source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

tracer_extended = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitPointDict(point_dict=point_dict, tracer=tracer, solver=solver)

print(fit["point_0"].positions.residual_map)
print(fit["point_0"].positions.normalized_residual_map)
print(fit["point_0"].positions.chi_squared_map)
print(fit["point_0"].positions.log_likelihood)

"""
__Result__

The **PyAutoLens** visualization library and `FitPoint` object includes specific methods for plotting the results.

__Wrap Up__

The `point_source` package of the `autolens_workspace` contains numerous example scripts for performing point source
modeling to datasets where there are only a couple of lenses and lensed sources, which fall under the category of
'galaxy scale' objects.

This also includes examples of how to add and fit other information that are observed by a point-source source,
for example the flux of each image.
"""

"""
Fits
====

This tutorial inspects an inferred model using the `Tracer` object inferred by the non-linear search.
This allows us to visualize and interpret its results.

The first half of this tutorial repeats the over example `overview/overview_1_lensing.py` and contains the
following:

This tutorial focuses on explaining how to use the inferred tracer to compute results as numpy arrays and only
briefly discusses visualization.

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `guides/units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoLens** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autolens_workspace/*/guides/data_structures.ipynb` guide.

__Other Models__

This tutorial does not use a pixelized source reconstruction or linear light profiles, which have their own dediciated
functionality that interfacts with the `FitImaging` object.

These are described in the dedicated example scripts `results/examples/linear.py` and `results/examples/pixelizaiton.py`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
__Grids__

To describe the deflection of light, **PyAutoLens** uses `Grid2D` data structures, which are two-dimensional
Cartesian grids of (y,x) coordinates. 

Below, we make and plot a uniform Cartesian grid in units of arcseconds. 

All quantities which are distance units (e.g. coordinate centre's radii) are in units of arc-seconds, as this is the
most convenient unit to represent lensing quantities.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.set_title(label="Cartesian (y,x) Grid (arcsec)")
grid_plotter.figure_2d()

"""
__Light Profiles__

We will ray-trace this `Grid2D`'s coordinates to calculate how the lens galaxy's mass deflects the source 
galaxy's light. We therefore need analytic functions representing a galaxy's light and mass distributions. 

This requires analytic functions representing the light and mass distributions of galaxies, for example the 
elliptical `Sersic` `LightProfile`:
"""
sersic_light_profile = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.2, 0.1),
    intensity=0.005,
    effective_radius=2.0,
    sersic_index=2.5,
)

"""
By passing this profile a `Grid2D`, we can evaluate the light at every (y,x) coordinate on the `Grid2D` and create an 
image of the Sersic.

All images in **PyAutoLens** are in units of electrons per second.
"""
image = sersic_light_profile.image_2d_from(grid=grid)

"""
The **PyAutoLens** plot module provides methods for plotting objects and their properties, like light profile's image.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile, grid=grid
)
light_profile_plotter.set_title(label="Image of Sersic Light Profile")
light_profile_plotter.figures_2d(image=True)

"""
__Mass Profiles__

**PyAutoLens** uses `MassProfile` objects to represent a galaxy's mass distribution and perform ray-tracing
calculations. 

Below we create an `Isothermal` mass profile and compute its deflection angles on our Cartesian grid, which describe
how the source galaxy's light rays are deflected as they pass this mass distribution.
"""
isothermal_mass_profile = al.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.1, 0.0), einstein_radius=1.6
)
deflections = isothermal_mass_profile.deflections_yx_2d_from(grid=grid)

"""
Lets plot the isothermal mass profile's deflection angle map.

The black curve on the figure is the tangential critical curve of the mass profile, if you do not know what this is
don't worry about it for now!
"""
mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=isothermal_mass_profile, grid=grid
)
mass_profile_plotter.set_title(label="Isothermal Deflection Angles (y)")
mass_profile_plotter.figures_2d(
    deflections_y=True,
)
mass_profile_plotter.set_title(label="Isothermal Deflection Angles (x)")
mass_profile_plotter.figures_2d(
    deflections_x=True,
)

"""
There are many other lensing quantities which can be plotted, for example the convergence and gravitational
potential.

If you are not familiar with gravitational lensing and therefore are unclear on what the convergence and potential 
are, don't worry for now!
"""
mass_profile_plotter.set_title(label="Isothermal Mass Convergence")
mass_profile_plotter.figures_2d(
    convergence=True,
)
mass_profile_plotter.set_title(label="Isothermal Mass Potential")
mass_profile_plotter.figures_2d(
    potential=True,
)

"""
__Galaxies__

A `Galaxy` object is a collection of `LightProfile` and `MassProfile` objects at a given redshift. 

The code below creates two galaxies representing the lens and source galaxies shown in the strong lensing diagram above.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5, bulge=sersic_light_profile, mass=isothermal_mass_profile
)

source_light_profile = al.lp.ExponentialCore(
    centre=(0.3, 0.2), ell_comps=(0.1, 0.0), intensity=0.1, effective_radius=0.5
)

source_galaxy = al.Galaxy(redshift=1.0, bulge=source_light_profile)

"""
The geometry of the strong lens system depends on the cosmological distances between the Earth, the lens galaxy and 
the source galaxy. It there depends on the redshifts of the `Galaxy` objects. 

By passing these `Galaxy` objects to a `Tracer` with a `Cosmology` object, **PyAutoLens** uses these galaxy redshifts 
and a cosmological model to create the appropriate strong lens system.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.Planck15())

"""
__Ray Tracing__

We can now create the image of the strong lens system! 

When calculating this image, the `Tracer` performs all ray-tracing for the strong lens system. This includes using the 
lens galaxy's total mass distribution to deflect the light-rays that are traced to the source galaxy. As a result, 
the source's appears as a multiply imaged and strongly lensed Einstein ring.
"""
image = tracer.image_2d_from(grid=grid)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.set_title(label="Image of Strong Lens System")
tracer_plotter.figures_2d(image=True)

"""
__Log10__

The light and masss distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `MatPlot2D` object has an input `use_log10`, which will do this automatically when we call the `figures_2d` method.
Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer,
    grid=grid.mask.derive_grid.all_false,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
tracer_plotter.figures_2d(image=True)

"""
The `TracerPlotter` includes the mass quantities we plotted previously, which can be plotted as a subplot 
that plots all these quantities simultaneously.

The black and white lines in the source-plane image are the tangential and radial caustics of the mass, which again
you do not need to worry about for now if you don't know what that is!
"""
tracer_plotter.set_title(label=None)
tracer_plotter.subplot_tracer()

"""
The tracer is composed of planes. The system above has two planes, an image-plane (at redshift=0.5) and a 
source-plane (at redshift=1.0). 

When creating an image via a Tracer, the mass profiles are used to ray-trace the image-plane grid (plotted above) 
to a source-plane grid, via the mass profile's deflection angles.

We can use the tracer`s `traced_grid_2d_list_from` method to calculate and plot the image-plane and source-plane grids.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

grid_plotter = aplt.Grid2DPlotter(grid=traced_grid_list[0])
grid_plotter.set_title(label="Image Plane Grid")
grid_plotter.figure_2d()

grid_plotter = aplt.Grid2DPlotter(grid=traced_grid_list[1])
grid_plotter.set_title(label="Source Plane Grid")
grid_plotter.figure_2d()  # Source-plane grid.

"""
__Extending Objects__

The **PyAutoLens** API has been designed such that all of the objects introduced above are extensible. `Galaxy` 
objects can take many `LightProfile`'s and `MassProfile`'s. `Tracer`' objects can take many `Galaxy`'s. 

If the galaxies are at different redshifts a strong lensing system with multiple lens planes will be created, 
performing complex multi-plane ray-tracing calculations.

To finish, lets create a `Tracer` with 3 galaxies at 3 different redshifts, forming a system with two distinct Einstein
rings! The mass distribution of the first galaxy also has separate components for its stellar mass and dark matter.
"""
lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05),
        intensity=0.5,
        effective_radius=0.3,
        sersic_index=3.5,
        mass_to_light_ratio=0.6,
    ),
    disk=al.lmp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.1),
        intensity=1.0,
        effective_radius=2.0,
        mass_to_light_ratio=0.2,
    ),
    dark=al.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.08, scale_radius=30.0),
)

lens_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Exponential(
        centre=(0.00, 0.00),
        ell_comps=(0.05, 0.05),
        intensity=1.2,
        effective_radius=0.1,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.05, 0.05), einstein_radius=0.3
    ),
)

source_galaxy = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.4,
        effective_radius=0.1,
        sersic_index=1.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])

"""
This is what the lens looks like. 

Note how crazy the critical curves are!
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.set_title(label="Image of Complex Strong Lens")
tracer_plotter.figures_2d(image=True)


"""
__Attributes__

Printing individual attributes of the max log likelihood tracer gives us access to the inferred parameters of the
lens and source galaxies.

The tracer contains the galaxies as both a list and an instance of the model used to fit it. This means we can
access the same values in two ways, either indexing the galaxies list index or by the name used in model composition.

It can be difficult to track which galaxy is which index in the list, so it is recommended to use the model
composition to access the galaxies.
"""
print(f"Einstein Radius via list index = {tracer.galaxies[1].mass.einstein_radius}")

"""
__Lensing Quantities__

The maximum log likelihood tracer contains a lot of information about the inferred model.

For example, by passing it a 2D grid of (y,x) coordinates we can return a numpy array containing its 2D image. This
includes the lens light and lensed source images.

Below, we use the grid of the `imaging` to computed the image on, which is the grid used to fit to the data.
"""
image = tracer.image_2d_from(grid=grid)

"""
__Data Structures Slim / Native__

The image above is returned as a 1D numpy array. 

**PyAutoLens** includes dedicated functionality for manipulating this array, for example mapping it to 2D or
performing the calculation on a high resolution sub-grid which is then binned up. 

This uses the data structure API, which is described in the `results/examples/data_structures.py` example. This 
tutorial will avoid using this API, but if you need to manipulate results in more detail you should check it out.
"""
print(image.slim)

"""
__Grid Choices__

We can input a different grid, which is not masked, to evaluate the image everywhere of interest. We can also change
the grid's resolution from that used in the model-fit.

The examples uses a grid with `shape_native=(3,3)`. This is much lower resolution than one would typically use to 
perform ray tracing, but is chosen here so that the `print()` statements display in a concise and readable format.
"""
grid = al.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.1)

image = tracer.image_2d_from(grid=grid)

print(image.slim)

"""
__Sub Gridding__

A grid can also have a sub-grid, defined via its `sub_size`, which defines how each pixel on the 2D grid is split 
into sub-pixels of size (`sub_size` x `sub_size`). 

The calculation below shows how to use a sub-grid and return an image which has already been binned up. 

Full details of the API for this calculation are given in the `guides/over_sampling.py` example.
"""
grid = al.Grid2D.uniform(
    shape_native=grid.shape_native,
    pixel_scales=grid.pixel_scales,
    over_sampling=al.OverSamplingUniform(sub_size=2),
)

grid_sub = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1)

image = tracer.image_2d_from(grid=grid_sub)

print(image)

"""
__Positions Grid__

We may want the image at specific (y,x) coordinates.

We can use an irregular 2D (y,x) grid of coordinates for this. The grid below evaluates the image at:

- y = 1.0, x = 1.0.
- y = 1.0, x = 2.0.
- y = 2.0, x = 2.0.
"""
grid_irregular = al.Grid2DIrregular(values=[[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]])

image = tracer.image_2d_from(grid=grid_irregular)

print(image)

"""
__Scalar Lensing Quantities__

The tracer has many scalar lensing quantities, which are all returned using an `Array2D` and therefore use the same 
interface as images, described above.

For example, we can compute the `Tracer`'s convergence using all of the grids above.
"""
convergence_2d = tracer.convergence_2d_from(grid=grid)
print(convergence_2d)

convergence_2d = tracer.convergence_2d_from(grid=grid_sub)
print(convergence_2d)

convergence_2d = tracer.convergence_2d_from(grid=grid_irregular)
print(convergence_2d)

"""
This is the convergence of every galaxy in the tracer summed together. It may not be appropriate if your lens model 
performs multi-plane ray-tracing (e.g. there are more than 2 redshifts containing galaxies). Later results tutorials
provide tools that are more appropriate for multi-plane tracers.

There are other scalar quantities accessible via the tracer (those not familiar with strong lensing mathematical 
formalism may not recognise what these quantities are -- don't worry about it for now!):
"""
potential_2d = tracer.potential_2d_from(grid=grid)

tangential_eigen_value = tracer.tangential_eigen_value_from(grid=grid)
radial_eigen_value = tracer.radial_eigen_value_from(grid=grid)


"""
A 2D magnification map is available, which using only the ray-tracing and therefore mass model quantities how much
light rays are focus at a given point in the image-plane.

If you are studying a strongly lensed source galaxy and want to know how much the galaxy itself is magnified, the
magnification below is not of too much use too you. In the result tutorial `galaxies.py` we explain how the 
magnification of the source can be quantified.
"""
magnification_2d = tracer.magnification_2d_from(grid=grid)

"""
__Vector Quantities__

Many lensing quantities are vectors. That is, they are (y,x) coordinates that have 2 values representing their
magnitudes in both the y and x directions.

These quantities also have a dedicated data structure which is described fully in 
the `results/examples/data_structure.py` example.

The most obvious of these is the deflection angles, which are used throughout lens modeling to ray-trace grids
from the image-plane to the source-plane via a lens galaxy mass model.

To indicate that a quantities is a vector, **PyAutoLens** uses the label `_yx`
"""
deflections_yx_2d = tracer.deflections_yx_2d_from(grid=grid)

"""
For vector quantities the has shape `2`, corresponding to the y and x vectors respectively.
"""
print(deflections_yx_2d[0, :])

"""
The `VectorYX2D` object has a built in method to return the magnitude of each vector, which is a scalar quantity
and therefore returned using a 1D Numpy array.
"""
deflection_magnitudes_2d = deflections_yx_2d.magnitudes
print(deflection_magnitudes_2d)

"""
__Other Vector Lensing Quantities__

The tracer has other vector lensing quantities, which use the same interface described above.
"""
shear_yx_2d = tracer.shear_yx_2d_via_hessian_from(grid=grid)

"""
__Other Quantities__

Many more quantities are shown below.

A full description of each can be found in the docstring of the source code of each function:

   https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/operate/deflections.py
"""
tangential_critical_curve = tracer.tangential_critical_curve_list_from(grid=grid)

radial_critical_curve = tracer.radial_critical_curve_list_from(grid=grid)

tangential_caustic = tracer.tangential_caustic_list_from(grid=grid)

radial_caustic = tracer.radial_caustic_list_from(grid=grid)

### You should be able to comment this out and it work fine ###

# area_within_tangential_critical_curve = (
#     tracer.tangential_critical_curve_area_list_from(grid=grid)
# )
#
# einstein_radius = tracer.einstein_radius_from(grid=grid)
#
# einstein_mass_angular = tracer.einstein_mass_angular_from(grid=grid)

"""
Fin.
"""

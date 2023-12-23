"""
Plots: GalaxyPlotter
====================

This example illustrates how to plot a `Galaxy` using a `GalaxyPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
__Galaxy__

First, lets create a `Galaxy` with multiple `LightProfile`'s and a `MassProfile`.
"""
bulge = al.lp.Sersic(
    centre=(0.0, -0.05),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = al.lp.Exponential(
    centre=(0.0, 0.05),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=2.0,
    effective_radius=1.6,
)

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=0.8,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
)

galaxy = al.Galaxy(redshift=0.5, bulge=bulge, disk=disk, mass=mass)

"""
__Grid__

We also need the 2D grid the `Galaxy`'s `Profile`'s are evaluated on.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Figures__

We now pass the galaxy and grid to a `GalaxyPlotter` and call various `figure_*` methods to plot different attributes.

Below, we create 2D figures showing the image, convergence and other properties of the galaxy. Later in the script
we show how to make 1D plots as a function of radius of these quantities.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)
galaxy_plotter.figures_2d(
    image=True,
    convergence=True,
    potential=False,
    deflections_y=True,
    deflections_x=True,
    magnification=True,
)

"""
__Subplots__

The `GalaxyPlotter` also has subplot method that plot each individual `Profile` in 2D as well as a 1D plot showing all
`Profiles` together.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)
galaxy_plotter.subplot_of_mass_profiles(
    convergence=True, potential=True, deflections_y=True, deflections_x=True
)

"""
__Include__

A `Galaxy` and its `Grid2D` contains the following attributes which can be plotted automatically via 
the `Include2D` object.

(By default, a `Grid2D` does not contain a `Mask2D`, we therefore manually created a `Grid2D` with a mask to illustrate
plotting its mask and border below).
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
)

mask = al.Mask2D.circular(
    shape_native=grid.shape_native, pixel_scales=grid.pixel_scales, radius=2.0
)
masked_grid = al.Grid2D.from_mask(mask=mask)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=masked_grid, include_2d=include)
galaxy_plotter.figures_2d(image=True)


"""
__Figures 1D__

We can plot 1D profiles, which display a properties of the galaxy in 1D as a function of radius.

For the 1D plot of each profile, the 1D grid of (x,) coordinates is centred on the profile and aligned with the 
major-axis. 

Because the `GalaxyPlotter` above has an input `Grid2D` object, the 1D grid of radial coordinates used to plot
these quantities is derived from this 2D grid. The 1D grid corresponds to the longest radial distance from the centre
of the galaxy's light or mass profiles to the edge of the 2D grid.
"""
galaxy_plotter.figures_1d(image=True, convergence=True, potential=True)

"""
If we want a specific 1D grid of a certain length over a certain range of coordinates, we can manually input a `Grid1D`
object.

Below, we create a `Grid1D` starting from 0 which plots the image and convergence over the radial range 0.0" -> 10.0".
"""
grid_1d = al.Grid1D.uniform_from_zero(shape_native=(1000,), pixel_scales=0.01)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)

galaxy_plotter.figures_1d(image=True, convergence=True)

"""
Using a `Grid1D` which does not start from 0.0" plots the 1D quantity with both negative and positive radial 
coordinates.

This plot isn't particularly useful, but it shows how 1D plots work.
"""
grid_1d = al.Grid1D.uniform(shape_native=(1000,), pixel_scales=0.01)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)

galaxy_plotter.figures_1d(image=True, convergence=True)

"""
We can also plot decomposed 1D profiles, which display the 1D quantity of every individual light and / or mass profiles. 

For the 1D plot of each profile, the 1D grid of (x) coordinates is centred on the profile and aligned with the 
major-axis. This means that if the galaxy consists of multiple profiles with different centres or angles the 1D plots 
are defined in a common way and appear aligned on the figure.

We'll plot this using our masked grid above, which converts the 2D grid to a 1D radial grid used to plot every
profile individually.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=masked_grid)

galaxy_plotter.figures_1d_decomposed(image=True, convergence=True, potential=True)

"""
__Errors__

Using a `GalaxyPDFPlotter`, we can make 1D plots that show the errors of the light and mass models estimated via a 
model-fit. 

Here, the `galaxy_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of a model-fit (the 
database tutorials show how these can be easily computed after a model fit). 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light or mass profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` and `einstein_radius1 with errors, which are plotted as vertical lines.

Below, we manually input two `Galaxy` objects with ligth and mass profiles that clearly show these errors on the figure.
"""
bulge_0 = al.lp.Sersic(intensity=4.0, effective_radius=0.4, sersic_index=3.0)

disk_0 = al.lp.Exponential(intensity=2.0, effective_radius=1.4)

mass_0 = al.mp.Isothermal(einstein_radius=0.7)

mass_clump_0 = al.mp.Isothermal(einstein_radius=0.1)

galaxy_0 = al.Galaxy(
    redshift=0.5, bulge=bulge_0, disk=disk_0, mass=mass_0, mass_clump=mass_clump_0
)

bulge_1 = al.lp.Sersic(intensity=4.0, effective_radius=0.8, sersic_index=3.0)

disk_1 = al.lp.Exponential(intensity=2.0, effective_radius=1.8)

mass_1 = al.mp.Isothermal(einstein_radius=0.9)

mass_clump_1 = al.mp.Isothermal(einstein_radius=0.2)

galaxy_1 = al.Galaxy(
    redshift=0.5, bulge=bulge_1, disk=disk_1, mass=mass_1, mass_clump=mass_clump_1
)

galaxy_pdf_plotter = aplt.GalaxyPDFPlotter(
    galaxy_pdf_list=[galaxy_0, galaxy_1], grid=grid, sigma=3.0
)
galaxy_pdf_plotter.figures_1d(image=True, convergence=True, potential=True)

"""
A decomposed plot of the individual light profiles of the galaxy, with errors, can also be created.
"""
galaxy_pdf_plotter.figures_1d_decomposed(image=True, convergence=True, potential=True)

"""
Finish.
"""

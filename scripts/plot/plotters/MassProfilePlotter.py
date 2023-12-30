"""
Plots: MassProfilePlotter
=========================

This example illustrates how to plot a `MassProfile` using a `MassProfilePlotter`.

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
__Mass Profile__

First, lets create a simple `MassProfile` which we'll plot.
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
)

"""
__Grid__

We also need the 2D grid the `MassProfile` is evaluated on.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Figures__

We now pass the mass profile and grid to a `MassProfilePlotter` and call various `figure_*` methods to 
plot different attributes in 1D and 2D.
"""
mass_profile_plotter = aplt.MassProfilePlotter(mass_profile=mass, grid=grid)
mass_profile_plotter.figures_2d(
    convergence=True,
    potential=True,
    deflections_y=True,
    deflections_x=True,
    magnification=True,
)
mass_profile_plotter.figures_1d(convergence=True, potential=True)

"""
__Include__

A `MassProfile` and its `Grid2D` contains the following attributes which can be plotted automatically via 
the `Include2D` object.

(By default, a `Grid2D` does not contain a `Mask2D`, we therefore manually created a `Grid2D` with a mask to illustrate
plotting its mask and border below).
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
)

mask = al.Mask2D.circular_annular(
    shape_native=grid.shape_native,
    pixel_scales=grid.pixel_scales,
    inner_radius=0.3,
    outer_radius=2.0,
)
masked_grid = al.Grid2D.from_mask(mask=mask)

mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=mass, grid=masked_grid, include_2d=include
)
mass_profile_plotter.figures_2d(convergence=True)

"""
__Log10__

Mass profiles are often clearer in log10 space, which inputting `use_log10=True` into the `MatPlot2D` object
will do.

The same image can be set up manually via the `CMap`, `Contour` and `Colorbar` objects, but given this is a common
use-case, the `use_log10` input is provided for convenience.
"""
mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=mass, grid=masked_grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
mass_profile_plotter.figures_2d(convergence=True, potential=True)

"""
__Errors__

Using a `MassProfilePDFPlotter`, we can make 1D plots that show the errors of a mass model estimated via a model-fit. 

Here, the `mass_profile_pdf_list` is a list of `MassProfile` objects that are drawn randomly from the PDF of a 
model-fit (the database tutorials show how these can be easily computed after a model fit). 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D mass profile, which is plotted as a shaded region on the figure. 
 - The median `einstein_radius` with errors, which are plotted as vertical lines.

Below, we manually input two `MassProfiles` that clearly show these errors on the figure.
"""
mass_0 = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.5,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
)

mass_1 = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.7,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
)

mass_profile_pdf_plotter = aplt.MassProfilePDFPlotter(
    mass_profile_pdf_list=[mass_0, mass_1], grid=grid, sigma=3.0
)
mass_profile_pdf_plotter.figures_1d(convergence=True, potential=True)

"""
Finish.
"""

"""
Plots: LightProfilePlotter
==========================

This example illustrates how to plot a `LightProfile` using a `LightProfilePlotter`.

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
__Light Profile__

First, lets create a simple `LightProfile` which we'll plot.
"""
bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=1.0,
    effective_radius=0.8,
    sersic_index=4.0,
)

"""
__Grid__

We also need the 2D grid the `LightProfile` is evaluated on.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Figures__

We now pass the light profile and grid to a `LightProfilePlotter` and call various `figure_*` methods to 
plot different attributes in 1D and 2D.
"""
light_profile_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=grid)
light_profile_plotter.figures_1d(image=True)
light_profile_plotter.figures_2d(image=True)

"""
__Include__

A `LightProfile` and its `Grid2D` contains the following attributes which can be plotted automatically via 
the `Include2D` object.

(By default, a `Grid2D` does not contain a `Mask2D`, we therefore manually created a `Grid2D` with a mask to illustrate
plotting its mask and border below).
"""
include = aplt.Include2D(
    origin=True, mask=True, border=True, light_profile_centres=True
)

mask = al.Mask2D.circular(
    shape_native=grid.shape_native, pixel_scales=grid.pixel_scales, radius=2.0
)
masked_grid = al.Grid2D.from_mask(mask=mask)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=bulge, grid=masked_grid, include_2d=include
)
light_profile_plotter.figures_2d(image=True)

"""
__Log10__

Light profiles are often clearer in log10 space, which inputting `use_log10=True` into the `MatPlot2D` object
will do.

The same image can be set up manually via the `CMap`, `Contour` and `Colorbar` objects, but given this is a common
use-case, the `use_log10` input is provided for convenience.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=bulge, grid=grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
light_profile_plotter.figures_2d(image=True)

"""
__Errors__

Using a `LightProfilePDFPlotter`, we can make 1D plots that show the errors of a light model estimated via a model-fit. 

Here, the `light_profile_pdf_list` is a list of `LightProfile` objects that are drawn randomly from the PDF of a 
model-fit (the database tutorials show how these can be easily computed after a model fit). 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` with errors, which are plotted as vertical lines.

Below, we manually input two `LightProfiles` that clearly show these errors on the figure.
"""
bulge_0 = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=1.5,
    effective_radius=0.4,
    sersic_index=4.0,
)

bulge_1 = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=0.5,
    effective_radius=1.6,
    sersic_index=4.0,
)

light_profile_pdf_plotter = aplt.LightProfilePDFPlotter(
    light_profile_pdf_list=[bulge_0, bulge_1], grid=grid, sigma=3.0
)
light_profile_pdf_plotter.figures_1d(image=True)

"""
Finish.
"""

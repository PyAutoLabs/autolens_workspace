"""
Plots: TracerPlotter
====================

This example illustrates how to plot a `Tracer` using a `TracerPlotter`.

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
__Tracer__

First, lets create a `Tracer`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=0.4,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.2, 0.2)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Grid__

We also need an image-plane `Grid2D` which we'll ray-trace via the `Tracer`.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Figures__

We now pass the tracer` and grid to a `TracerPlotter` and call various `figure_*` methods to plot different attributes.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(
    image=True,
    convergence=True,
    potential=True,
    deflections_y=True,
    deflections_x=True,
    magnification=True,
)

"""
__Subplots__

A subplot of the above quantaties can be plotted.
"""
tracer_plotter.subplot_tracer()

"""
A subplot of the image-plane image and image in the source-plane of the galaxies in each plane can also be plotted 
(note that for  plane 0 the image-plane image and plane image are the same, thus the latter is omitted).
"""
tracer_plotter.subplot_galaxies_images()

"""
__Include__

A `Tracer` and its `Grid2D` contains the following attributes which can be plotted automatically via 
the `Include2D` object.

(By default, a `Grid2D` does not contain a `Mask2D`, we therefore manually created a `Grid2D` with a mask to illustrate
plotting its mask and border below).
"""
mask = al.Mask2D.circular(
    shape_native=grid.shape_native, pixel_scales=grid.pixel_scales, radius=2.0
)
masked_grid = al.Grid2D.from_mask(mask=mask)

include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
)
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=masked_grid, include_2d=include)
tracer_plotter.figures_2d(image=True, source_plane=True)

"""
__Log10__

A plane's light and mass profiles are often clearer in log10 space, which inputting `use_log10=True` into 
the `MatPlot2D` object will do.

The same image can be set up manually via the `CMap`, `Contour` and `Colorbar` objects, but given this is a common
use-case, the `use_log10` input is provided for convenience.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=masked_grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
tracer_plotter.figures_2d(image=True, convergence=True, potential=True)

"""
__Plane Image__

Whereas a `GalaxiesPlotter` had a method to plot its `plane_image`, it did not know the caustics of the source-plane as
they depend on the `MassProfile`'s of `Galaxy`'s in lower redshift planes. When we plot a plane image with a `Tracer`,
this information is now available and thus the caustics of the source-plane are now plotted.

The same is true of the `border, where the `border` plotted on the image-plane image has been ray-traced to the 
source-plane. This is noteworthy as it means in the source-plane we can see where our entire masked region traces too.

By default, this image is zoomed to the brightest pixels, so the galaxy can be clearly seen.
"""
tracer_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=True
)

"""
If we do not want the image to be zoomed, we can pass `zoom_to_brightest=False`. 

This shows the full extent of the grid used to create the source-plane image, and may also include the caustics 
which the zoomed image does not due to zooming inside of them.
"""
tracer_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=False
)

"""
Finish.
"""

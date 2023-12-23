"""
Plots: PlanePlotter
===================

This example illustrates how to plot a `Plane` using a `PlanePlotter`.

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
__Grid__

First, lets create a image-plane `Grid2D` and ray-trace it via `MassProfile` to create a source-plane `Grid2D`.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

mass_profile = al.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.1, 0.2), einstein_radius=1.0
)
deflections = mass_profile.deflections_yx_2d_from(grid=grid)
lens_galaxy = al.Galaxy(redshift=0.5, mass=mass_profile)

lensed_grid = grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

"""
__Plane__

We create a `Plane` representing a source-plane containing a `Galaxy` with a `LightProfile`.
"""
bulge = al.lp.Sersic(
    centre=(0.1, 0.1),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
    intensity=0.3,
    effective_radius=1.0,
    sersic_index=2.5,
)

source_galaxy = al.Galaxy(redshift=1.0, bulge=bulge)

image_plane = al.Plane(galaxies=[lens_galaxy])
source_plane = al.Plane(galaxies=[source_galaxy])

"""
__Figures__

We can plot the `image_plane` by passing it and our `grid to a` PlanePlotter` and calling various `figure_*` methods.

In this script our `lens_galaxy` only had a `MassProfile` so only methods like `figure_convergence` are
available.
"""
plane_plotter = aplt.PlanePlotter(plane=image_plane, grid=grid)
plane_plotter.figures_2d(convergence=True)

"""
__Subplots__

A subplot of the above quantaties can be plotted.
"""
plane_plotter.subplot_plane()

"""
A subplot of the image of the galaxies in the plane can also be plotted.
"""
plane_plotter.subplot_galaxy_images()

"""
We can also plot the `source_plane` by passing it with the `lensed_grid` to a `PlanePlotter`.

In this case, our `source_galaxy` only had a ` LightProfile` so only a plot of its image is available.
"""
plane_plotter = aplt.PlanePlotter(plane=source_plane, grid=lensed_grid)
plane_plotter.figures_2d(image=True)

"""
In addition to the lensed image of the source-plane, we can plot its unlensed image (e.g. how the source-galaxy 
appears in the source-plane before lensing) using the `figure_plane_image` method.

By default, this image is zoomed to the brightest pixels, so the galaxy can be clearly seen.
"""
plane_plotter.figures_2d(plane_image=True, zoom_to_brightest=True)

"""
If we do not want the image to be zoomed, we can pass `zoom_to_brightest=False`. This shows the full extent of the
grid used to create the source-plane image.
"""
plane_plotter.figures_2d(plane_image=True, zoom_to_brightest=False)

"""
__Galaxy Image__

We can also plot specific images of galaxies in the plane.
"""
plane_plotter.figures_2d_of_galaxies(image=True, galaxy_index=0)

"""
__Visuals__

It is feasible for us to plot the caustics in the source-plane. However, to calculate the `Caustics` we must manually
compute them from the image-plane `MassProfile` and pass them to the source-plane mat_plot_2d. 
"""
visuals = aplt.Visuals2D(
    tangential_caustics=image_plane.tangential_caustic_list_from(grid=grid),
    radial_caustics=image_plane.radial_caustic_list_from(grid=grid),
)
plane_plotter = aplt.PlanePlotter(
    plane=source_plane, grid=lensed_grid, visuals_2d=visuals
)
plane_plotter.figures_2d(plane_image=True)

"""
__Include__

For `PlanePlotter`'s, `GalaxyPlotter`'s and `LightProfilePlotter's that are plotting source-plane images, the only
way to plot the caustics is to manually extract them from the foreground `MassProfile`'s, as shown above. This is 
because these source-plane objects have no knowledge of what objects are in the image-plane.

`TracerPlotter`'s automatically extract and plot caustics on source-plane figures, given they have available the 
necessary information on the image-plane mass. This is shown in `autolens_workspace/plot/plotters/TracerPlotter.py`.

A `Plane` and its `Grid2D` contains the following attributes which can be plotted automatically via 
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

"""
Note that the image-plane has no `LightProfile`'s and does not plot any light-profile centres. Similarly, the 
source-plane has no `MassProfile`'s and plot no mass-profile centres.
"""
plane_plotter = aplt.PlanePlotter(
    plane=image_plane, grid=masked_grid, include_2d=include
)
plane_plotter.figures_2d(image=True)
plane_plotter = aplt.PlanePlotter(
    plane=source_plane, grid=masked_grid, include_2d=include
)
plane_plotter.figures_2d(image=True)

"""
Finish.
"""

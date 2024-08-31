"""
Plots: VoronoiDrawer
====================

This example illustrates how to customize the appearance of the Voronoi mesh of a Voronoi mesh using the
`VoronoiDrawer` object.

__Natural Neighbor Interpolation__

To run the Voronoi mesh plotter, you must install the Voronoi natural neighbor interpolation package by
following these instructions:

https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/util/nn


__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We now mask the `Imaging` data so we can fit it with an `Inversion`.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)

"""
__Tracer__

The `Inversion` maps pixels from the image-plane of our `Imaging` data to its source plane, via a lens model.

Lets create a `Tracer` which we will use to create the `Inversion`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.111111, 0.0), einstein_radius=1.6
    ),
)

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
    mesh=al.mesh.Voronoi(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Converting a `Tracer` to an `Inversion` performs a number of steps, which are handled by the `TracerToInversion` class. 

This class is where the data and tracer's galaxies are combined to fit the data via the inversion.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

"""
We can customize the filling of Voronoi cells using the `VoronoiDrawer` object which wraps the 
method `matplotlib.fill()`:

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill.html
"""
voronoi_drawer = aplt.VoronoiDrawer(edgecolor="b", linewidth=1.0, linestyle="--")

mat_plot = aplt.MatPlot2D(voronoi_drawer=voronoi_drawer)

"""
We now pass the inversion to a `InversionPlotter` which we will use to illustrate customization with 
the `VoronoiDrawer` object.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=mat_plot)

try:
    inversion_plotter.figures_2d_of_pixelization(
        pixelization_index=0, reconstruction=True
    )
    inversion_plotter.subplot_of_mapper(mapper_index=0)
except ImportError:
    print(
        "You have not installed the Voronoi natural neighbor interpolation package, see instructions at top of notebook."
    )

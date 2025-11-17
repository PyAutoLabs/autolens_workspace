"""
Plots: Plotters Double Einstein Ring
====================================

This example illustrates the API for plotting using `Plotter` objects for double Einstein ring systems which have
more than two planes at different redshifts.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotters work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, tracer, data) used to illustrate plotting.
**Fit Imaging:** Plot the fit of a tracer to an imaging dataset for a double Einstein ring system.
**Inversion:** Plot the inversion object which performs the linear algebra and other calculations which reconstruct the source galaxy for a double Einstein ring system.

__Setup__

To illustrate plotting, we require standard objects like a grid, tracer and dataset.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

dataset_name = "double_einstein_ring"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.15, -0.15), intensity=1.2, effective_radius=0.1
    ),
    mass=al.mp.IsothermalSph(centre=(-0.15, -0.15), einstein_radius=0.3),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.45, 0.45), intensity=0.6, effective_radius=0.07
    ),
)


tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Imaging__

The `FitImaging` object is a base object which represents the fit of a model to an imaging dataset, including the
residuals, chi-squared and model image.

We now pass the FitImaging to an `FitImagingPlotter` and call various `figure_*` methods to plot different attributes.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)

fit_plotter.set_mat_plots_for_subplot(is_for_subplot=False)

fit_plotter.figures_2d(
    data=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
It can plot of the model image of an input plane.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, model_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, model_image=True)
fit_plotter.figures_2d_of_planes(model_image=True, plane_index=2)

"""
It can plot the image of a plane with all other model images subtracted.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, subtracted_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, subtracted_image=True)
fit_plotter.figures_2d_of_planes(subtracted_image=True, plane_index=2)

"""
It can also plot the plane-image of a plane, that is what the source galaxy looks like without lensing (e.g.
for `plane_index=1` this is the source-plane image)
"""
fit_plotter.figures_2d_of_planes(plane_index=0)
fit_plotter.figures_2d_of_planes(plane_index=1)
fit_plotter.figures_2d_of_planes(plane_index=2)

"""
The `FitImagingPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=0)
fit_plotter.subplot_of_planes(plane_index=1)
fit_plotter.subplot_of_planes(plane_index=2)

"""
We can also plot a `FitImaging` which uses a `Pixelization`.
"""
source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        image_mesh=None,
        mesh=al.mesh.RectangularMagnification(),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    pixelization=al.Pixelization(
        image_mesh=None,
        mesh=al.mesh.RectangularMagnification(),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
We can even extract an `InversionPlotter` (described below) from the `FitImagingPlotter` and use it to plot all of its usual methods, 
which will now include the caustic and border.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)

inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=2)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)

"""
__Inversion__

The fit above has a property called an `inversion`, which contains all of the linear algebra, mesh calculations
and other key quantities used to reconstruct a source galaxy using a pixelization.

This has its own dedicated plotter, the `InversionPlotter`, which can be used to plot the inversion's attributes
and properties in a similar way to the `FitImagingPlotter`.

Converting a `Tracer` to an `Inversion` performs a number of steps, which are handled by the `TracerToInversion` class. 

This class is where the data and tracer's galaxies are combined to fit the data via the inversion.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

"""
We now pass the inversion to a `InversionPlotter` and call various `figure_*` methods to plot different attributes.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
An `Inversion` can have multiple mappers, which reconstruct multiple source galaxies at different redshifts and
planes (e.g. double Einstein ring systems).

To plot an individual source we must therefore specify the mapper index of the source we plot.
"""
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstructed_image=True, reconstruction=True
)

inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=1, reconstructed_image=True, reconstruction=True
)

"""
The `Inversion` attributes can also be plotted as a subplot.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)
inversion_plotter.subplot_of_mapper(mapper_index=1)

"""
Finish.
"""

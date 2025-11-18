"""
Plots: Plotters
===============

This example illustrates the API for plotting using `Plotter` objects, which enable quick visualization of all
key quantities.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotters work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, tracer, data) used to illustrate plotting.
**Array2D:** Plot an `Array2D` object, which is a base object representing any 2D quantity (e.g. images, convergence, data).
**Grid2D:** Plot a `Grid2D` object, which is a base object representing a (y,x) grid of coordinates in 2D space.
**Tracer:** Plot a `Tracer` object, representing a tracer of light through the universe, including the mass of lens galaxies.
**Imaging:** Plot an `Imaging` object, representing an imaging dataset, including the data, noise-map and PSF.
**Fit Imaging:** Plot a `FitImaging` object, representing the fit of a model to an imaging dataset (including residuals, chi-squared and model image).
**Light Profile:** Plot a `LightProfile` object, representing the light of a galaxy.
**Mass Profile:** Plot a `MassProfile` object, representing the mass of a galaxy.
**Galaxy:** Plot a `Galaxy` object, which is a collection of light and mass profiles.
**Galaxies:** Plot a `Galaxies` object, which is a collection of galaxies.
**Interferometer:** Plot an `Interferometer` object, representing an interferometer dataset, including the data, noise-map and UV wavelengths.
**Fit Interferometer:** Plot a `FitInterferometer` object, representing the fit of a model to an interferometer dataset (including residuals, chi-squared and model image).
**Point Dataset:** Plot a `PointDataset` object, representing a point source dataset (e.g. lensed quasar, supernova).
**Fit Point Dataset:** Plot a `FitPointDataset` object, representing the fit of a model to a point source dataset (including residuals and chi-squared).

__Setup__

To illustrate plotting, we require standard objects like a grid, tracer and dataset.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
from pathlib import Path
import autolens as al
import autolens.plot as aplt

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

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
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Array2D__

The `Array2D` object is base object which represents any 2D quantity (e.g. images, convergence, data).

It can be plotted using an `Array2DPlotter` and calling the `figure` method.
"""
array_plotter = aplt.Array2DPlotter(array=dataset.data)
array_plotter.figure_2d()

"""
__Grid2D__

The `Grid2D` object is a base object which represents a (y,x) grid of coordinates in 2D space, 9including image-plane
and source-plane grids.

It can be plotted using a `Grid2DPlotter` and calling the `figure` method.
"""
grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure_2d()

"""
We can ray-trace grids using a tracer (or galaxy, mass profile) and plot them.
"""
deflections = tracer.deflections_yx_2d_from(grid=grid)

lensed_grid = grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

grid_plotter = aplt.Grid2DPlotter(grid=lensed_grid)
grid_plotter.figure_2d()

"""
__Tracer__

The `Tracer` object is a base object which represents a tracer of light through the universe, including
multiple galaxies and their light and mass profiles.

We can pass a tracer and grid to a `TracerPlotter` and call various `figure_*` methods to plot different attributes.
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
A subplot of the above quantaties can be plotted.
"""
tracer_plotter.subplot_tracer()

"""
A subplot of the image-plane image and image in the source-plane of the galaxies in each plane can also be plotted 
(note that for  plane 0 the image-plane image and plane image are the same, thus the latter is omitted).
"""
tracer_plotter.subplot_galaxies_images()

"""
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
A tracer consists of light and mass profiles, and their centres can be extracted and plotted over the image. 

The `visuals.ipynb` notebook, under the sections `LightProfileCentreScatter` and `MassProfilesCentreScatter`,
describes how to plot these visuals over images.

If the tracer has a mass profile, it also has critical curves and caustics. The `visuals.ipynb` notebook, under the 
sections `CriticalCurvesLine` and `CausticsLine`, describes how to plot these visuals over images.

__Imaging__

The `Imaging` object is a base object which represents an imaging dataset, including the data, noise-map and PSF.

It can be plotted using an `ImagingPlotter` and calling the `figure_*` methods to plot different attributes.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(
    data=True,
    noise_map=True,
    psf=True,
)

"""
The `ImagingPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()

"""
__Fit Imaging__

The `FitImaging` object is a base object which represents the fit of a model to an imaging dataset, including the
residuals, chi-squared and model image.

It can be plotted using a `FitImagingPlotter` and calling the `figure_*` methods to plot different attributes.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)

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

"""
It can plot the image of a plane with all other model images subtracted.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, subtracted_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, subtracted_image=True)

"""
It can also plot the plane-image of a plane, that is what the source galaxy looks like without lensing (e.g.
for `plane_index=1` this is the source-plane image).

By default, this source-plane image is zoomed to the brightest pixels, so the galaxy can be clearly seen.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, plane_image=True)
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=True
)

"""
If we do not want the image to be zoomed, we can pass `zoom_to_brightest=False`.

This shows the full extent of the grid used to create the source-plane image, and may also include the caustics
which the zoomed image does not due to zooming inside of them.
"""
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=False
)

"""
The source is often much fainter than the lens galaxy, meaning we may want to brighten its appearance to fully see it.

We can do this by passing the `use_source_vmax` bool, which sets the maximum value of the colormap to the maximum
flux in the source-plane.

This is used by default in the subplots plotted below.
"""
fit_plotter.figures_2d(data=True, use_source_vmax=True)
fit_plotter.figures_2d_of_planes(plane_index=1, model_image=True, use_source_vmax=True)
fit_plotter.figures_2d_of_planes(
    plane_index=1, subtracted_image=True, use_source_vmax=True
)

"""
The `FitImagingPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
It also includes a log10 subplot option, which shows the same figures but with the colormap in log10 format to
highlight the fainter regions of the data.
"""
fit_plotter.subplot_fit_log10()

"""
By default, the `residual_map` and `normalized_residual_map` use a symmetric colormap.

This means the maximum normalization (`vmax`) an minimum normalziation (`vmin`) are the same absolute value.

This can be disabled via the `residuals_symmetric_cmap` input.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit, residuals_symmetric_cmap=False)
fit_plotter.figures_2d(
    residual_map=True,
    normalized_residual_map=True,
)

"""
__Light Profile__

Light profiles have dedicated plotters which can plot their attributes in 1D and 2D.

We first pass a light profile and grid to a `LightProfilePlotter` and call various `figure_*` methods to 
plot different attributes in 1D and 2D.
"""
bulge = tracer.galaxies[0].bulge

light_profile_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=grid)
light_profile_plotter.figures_2d(image=True)

"""
A light profile centre can be extracted and plotted over the image. The `visuals.ipynb` notebook, under the 
section `LightProfileCentreScatter` describes how to plot these visuals over images.

__Mass Profiles__

Mass profiles have dedicated plotters which can plot their attributes in 1D and 2D.

We first pass a mass profile and grid to a `MassProfilePlotter` and call various `figure_*` methods to
plot different attributes in 1D and 2D.
"""
mass = tracer.galaxies[0].mass

mass_profile_plotter = aplt.MassProfilePlotter(mass_profile=mass, grid=grid)
mass_profile_plotter.figures_2d(
    convergence=True,
    potential=True,
    deflections_y=True,
    deflections_x=True,
    magnification=True,
)

"""
A mass profile centre can be extracted and plotted over the image.  The `visuals.ipynb` notebook, under the 
section `MassProfilesCentreScatter`, describes how to plot these visuals over images.

A mass profile also has critical curves and caustics. The `visuals.ipynb` notebook, under the 
sections `CriticalCurvesLine` and `CausticsLine`, describes how to plot these visuals over images.

__Galaxy__

A `Galaxy` is a collection of light and mass profiles, and can be plotted using a `GalaxyPlotter`.

We first pass a galaxy and grid to a `GalaxyPlotter` and call various `figure_*` methods to plot different
attributes in 1D and 2D.
"""
galaxy = tracer.galaxies[0]

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
The `GalaxyPlotter` also has subplot method that plot each individual `Profile` in 2D as well as a 1D plot showing all
`Profiles` together.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)
galaxy_plotter.subplot_of_mass_profiles(
    convergence=True, potential=True, deflections_y=True, deflections_x=True
)

"""
A galaxy consists of light and mass profiles, and their centres can be extracted and plotted over the image. 
The `visuals.ipynb` notebook, under the sections `LightProfileCentreScatter` and `MassProfilesCentreScatter`,
describes how to plot these visuals over images.

If the galaxy has a mass profile, it also has critical curves and caustics. The `visuals.ipynb` notebook, under the 
sections `CriticalCurvesLine` and `CausticsLine`, describes how to plot these visuals over images.

__One Dimensional Plots__

We often want to calculative 1D quantities of a light or mass profile, for example to plot how its light changes as
a function of radius.

To do this, we must still input a 2D grid into the `image_2d_from` method, therefore we create a project 2D 
radial grid as follows which has shape [Number_of_1d_coordinates, 2] and where all [:,0] entries are the same.

For example, we may want the project grid which traces it major axis in uniform radial steps.

This is easily computed using the `grid_2d_radial_project_from` function and passing the `centre` and `angle`
of a light profile we can make it align with the light profile itself.

We can now plot the 1D radial profile of the light profile. This profile shows how the intensity of the light 
changes as a function of distance from the profile's center. This is a more informative way to visualize the light p
rofile's distribution.

When we plot 1D quantities, we do not use built-in plotting functions as in 2D, but instead use standard
matplotlib functionality.
"""
grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=galaxy.bulge.centre, angle=bulge.angle()
)

image_1d = galaxy.bulge.image_2d_from(grid=grid_2d_projected)

plt.plot(grid_2d_projected[:, 1], image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
Using a `Grid1D` which does not start from 0.0" plots the 1D quantity with both negative and positive radial 
coordinates.
"""
grid_1d = al.Grid1D.uniform_from_zero(shape_native=(10000,), pixel_scales=0.01)
image_1d = bulge.image_2d_from(grid=grid_1d)

plt.plot(grid_1d, image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
We can also plot decomposed 1D profiles, which display the 1D quantity of every individual light and / or mass profiles. 

For the 1D plot of each profile, the 1D grid of (x) coordinates is centred on the profile and aligned with the 
major-axis. This means that if the galaxy consists of multiple profiles with different centres or angles the 1D plots 
are defined in a common way and appear aligned on the figure.

We'll plot this using our masked grid above, which converts the 2D grid to a 1D radial grid used to plot every
profile individually.
"""
grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=bulge.centre, angle=bulge.angle()
)
bulge_image_1d = bulge.image_2d_from(grid=grid_2d_projected)

grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=source_galaxy.bulge.centre, angle=source_galaxy.bulge.angle()
)
source_image_1d = source_galaxy.bulge.image_2d_from(grid=grid_2d_projected)

plt.plot(grid_2d_projected[:, 1], bulge_image_1d, label="Bulge")
plt.plot(grid_2d_projected[:, 1], source_image_1d, label="Disk")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.legend()
plt.show()
plt.close()

"""
__Galaxies__

A `Galaxies` is a collection of galaxies, and can be plotted using a `GalaxiesPlotter`.

We first pass a galaxies and grid to a `GalaxiesPlotter` and call various `figure_*` methods to plot different
attributes in 1D and 2D.

We separate the `image_plane_galaxies` and `source_plane_galaxies` so that we can plot them separately, as they
are often at different redshifts and thus have different properties.
"""
image_plane_galaxies = al.Galaxies(galaxies=[tracer.galaxies[0]])
source_plane_galaxies = al.Galaxies(galaxies=[tracer.galaxies[1]])

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=image_plane_galaxies, grid=grid)
galaxies_plotter.figures_2d(convergence=True)

"""
A subplot of the above quantaties can be plotted.
"""
galaxies_plotter.subplot_galaxies()

"""
A subplot of the image of the galaxies in the plane can also be plotted.
"""
galaxies_plotter.subplot_galaxy_images()

"""
We can also plot the `source_plane_galaxies` by passing it with the `lensed_grid` to a `GalaxiesPlotter`.

In this case, our `source_galaxy` only had a ` LightProfile` so only a plot of its image is available.
"""
galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=source_plane_galaxies, grid=lensed_grid
)
galaxies_plotter.figures_2d(image=True)

"""
In addition to the lensed image of the source-plane, we can plot its unlensed image (e.g. how the source-galaxy 
appears in the source-plane before lensing) using the `figure_plane_image` method.

By default, this image is zoomed to the brightest pixels, so the galaxy can be clearly seen.
"""
galaxies_plotter.figures_2d(plane_image=True, zoom_to_brightest=True)

"""
If we do not want the image to be zoomed, we can pass `zoom_to_brightest=False`. This shows the full extent of the
grid used to create the source-plane image.
"""
galaxies_plotter.figures_2d(plane_image=True, zoom_to_brightest=False)

"""
We can also plot specific images of galaxies in the plane.
"""
galaxies_plotter.figures_2d_of_galaxies(image=True, galaxy_index=0)

"""
A galaxy consists of light and mass profiles, and their centres can be extracted and plotted over the image. 
The `visuals.ipynb` notebook, under the sections `LightProfileCentreScatter` and `MassProfilesCentreScatter`,
describes how to plot these visuals over images.

If the galaxy has a mass profile, it also has critical curves and caustics. The `visuals.ipynb` notebook, under the 
sections `CriticalCurvesLine` and `CausticsLine`, describes how to plot these visuals over images.

__Interferometer__

The `Interferometer` object is a base object which represents an interferometer dataset, including the data, noise-map,
and UV wavelengths.

First, we load one.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

"""
We now pass the interferometer to an `InterferometerPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(
    data=True,
    noise_map=True,
    u_wavelengths=True,
    v_wavelengths=True,
    uv_wavelengths=True,
    amplitudes_vs_uv_distances=True,
    phases_vs_uv_distances=True,
)

"""
The dirty images of the interferometer dataset can also be plotted, which use the transformer of the interferometer 
to map the visibilities, noise-map or other quantity to a real-space image.
"""
dataset_plotter.figures_2d(
    dirty_image=True,
    dirty_noise_map=True,
    dirty_signal_to_noise_map=True,
)

"""
The `InterferometerPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Fit Interferometer__

The `FitInterferometer` object is a base object which represents the fit of a model to an interferometer dataset,
including the residuals, chi-squared and model image.

We now create one.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)


"""
We now pass the FitInterferometer to an `FitInterferometerPlotter` and call various `figure_*` methods 
to plot different attributes.
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=fit)

fit_plotter.figures_2d(
    data=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_data=True,
    residual_map_real=True,
    residual_map_imag=True,
    normalized_residual_map_real=True,
    normalized_residual_map_imag=True,
    chi_squared_map_real=True,
    chi_squared_map_imag=True,
)

"""
The dirty images of the interferometer fit can also be plotted, which use the transformer of the interferometer
to map the visibilities, noise-map, residual-map or other quantitiy to a real-space image.

Bare in mind the fit itself uses the visibilities and not the dirty images, so these images do not provide a direct
visualization of the fit itself. However, they are easier to inspect than the fits plotted above which are in Fourier
space and make it more straight forward to determine if an unphysical lens model is being fitted.
"""
fit_plotter.figures_2d(
    dirty_image=True,
    dirty_noise_map=True,
    dirty_signal_to_noise_map=True,
    dirty_model_image=True,
    dirty_residual_map=True,
    dirty_normalized_residual_map=True,
    dirty_chi_squared_map=True,
)

"""
It can plot of the image of an input plane, where this image is the real-space image of the `Tracer`.
"""
fit_plotter.figures_2d(image=True)

"""
It can also plot the plane-image of a plane, that is what the source galaxy looks like without lensing (e.g.
for `plane_index=1` this is the source-plane image)
"""
fit_plotter.figures_2d_of_planes(plane_index=0, plane_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
The `FitInterferometerPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

"""
The plane images can be combined to plot the appearance of the galaxy in real-space.
"""
fit_plotter.subplot_fit_real_space()

"""
By default, the `ditry_residual_map` and `dirty_normalized_residual_map` use a symmetric colormap.

This means the maximum normalization (`vmax`) an minimum normalziation (`vmin`) are the same absolute value.

This can be disabled via the `residuals_symmetric_cmap` input.
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=fit, residuals_symmetric_cmap=False)
fit_plotter.figures_2d(
    dirty_residual_map=True,
    dirty_normalized_residual_map=True,
)

"""
__Point Dataset__

The `PointDataset` object is a base object which represents a point source dataset (e.g. lensed quasar, supernova),
including the multiple image positions, their errors and optionally fluxes and time delays.

We now load one to illustrate plotting.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

dataset = al.from_json(
    file_path=Path(dataset_path, "point_dataset.json"),
)

"""
We now pass the point dataset to a `PointDatasetPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
point_dataset_plotter = aplt.PointDatasetPlotter(dataset=dataset)
# point_dataset_plotter.figures_2d(positions=True, fluxes=True)

"""
The `PointDatasetPlotter` can also plot a subplot of all of these attributes.
"""
point_dataset_plotter.subplot_dataset()

"""
__Fit Point Dataset__

The `FitPointDataset` object is a base object which represents the fit of a model to a point source dataset,
including the residuals and chi-squared.

We now create one.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.8,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0, point_0=al.ps.PointFlux(centre=(0.0, 0.0), flux=0.8)
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

fit = al.FitPointDataset(dataset=dataset, tracer=tracer, solver=solver)


"""
We now pass the FitPointDataset to a `FitPointDatasetPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
fit_plotter = aplt.FitPointDatasetPlotter(fit=fit)
fit_plotter.figures_2d(positions=True, fluxes=True)


"""
__Probability Density Function (PDF) Plots__

We can make 1D plots that show the errors of the light and mass models estimated via a model-fit. 

Here, the `light_profile_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of a model-fit (the 
database tutorials show how these can be easily computed after a model fit). 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light or mass profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` with errors, which are plotted as vertical lines.

Below, we manually input two `Galaxy` objects with light profiles that clearly show these errors on the figure.
"""
import math

light_profile_pdf_list = [bulge, disk]

sigma = 3.0
low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

image_1d_list = []

for light_profile in light_profile_pdf_list:
    grid_projected = grid.grid_2d_radial_projected_from(
        centre=light_profile.centre, angle=light_profile.angle()
    )

    image_1d_list.append(light_profile.image_2d_from(grid=grid_projected))

min_index = min([image_1d.shape[0] for image_1d in image_1d_list])
image_1d_list = [image_1d[0:min_index] for image_1d in image_1d_list]

(
    median_image_1d,
    errors_image_1d,
) = ag.util.error_util.profile_1d_median_and_error_region_via_quantile(
    profile_1d_list=image_1d_list, low_limit=low_limit
)

plt.plot(
    grid_2d_projected[:min_index, 1], median_image_1d, label="Median Light Profile"
)
plt.fill_between(
    x=grid_2d_projected[:min_index, 1],
    y1=errors_image_1d[0],
    y2=errors_image_1d[1],
    color="lightgray",
    label=f"{sigma} Sigma Error Region",
)

"""
Finish.
"""

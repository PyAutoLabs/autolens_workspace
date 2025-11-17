"""
__Log Likelihood Function: Parametric__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Interferometer` data
with a lens light profile and source light profile (e.g. an elliptical Sersic lens and source).

This script has the following aims:

 - To provide a resource that authors can include in papers, so that readers can understand the likelihood
 function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, which we will fit 
with the model.

This includes the method used to Fourier transform the real-space image of the galaxy to the uv-plane and compare 
directly to the visibilities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities. We will discuss how the calculation of the likelihood
function changes for different methods of Fourier transforming in this guide.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
This guide uses in-built visualization tools for plotting. 

For example, using the `InterferometerPlotter` the dataset we perform a likelihood evaluation on is plotted.

The `subplot_dataset` displays the visibilities in the uv-plane, which are the raw data of the interferometer
dataset. These are what will ultimately be directly fitted in the Fourier space.

The `subplot_dirty_images` displays the dirty images of the dataset, which are the reconstructed images of visibilities
using an inverse Fourier transform to convert these to real-space. These dirty images are not the images we fit, but
visualization of the dirty images are often used in radio interferometry to show the data in a way that is more
interpretable to the human eye.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling is used, 
which evaluates light profiles on a higher resolution grid than the image data to ensure the calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.

__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

The dataset is defined in real-space, and is Fourier transformed to the uv-plane for the model-fit. The grid is
therefore paired to the `real_space_mask`.

The coordinates are given by `dataset.grids.lp`, which we can plot and see is a uniform grid of (y,x) Cartesian 
coordinates which have had the 3.0" circular mask applied.

Each (y,x) coordinate coordinates to the centre of each image-pixel in the dataset, meaning that when this grid is
used to evaluate a light profile the intensity of the profile at the centre of each image-pixel is computed, making
it straight forward to compute the light profile's image to the image data.
"""
grid_plotter = aplt.Grid2DPlotter(grid=dataset.grids.lp)
grid_plotter.figure_2d()

print(f"(y,x) coordinates of first ten unmasked image-pixels {dataset.grid[0:9]}")

"""
To perform lensing calculations we convert this 2D (y,x) grid of coordinates to elliptical coordinates:

 $\eta = \sqrt{(x - x_c)^2 + (y - y_c)^2/q^2}$

Where:

 - $y$ and $x$ are the (y,x) arc-second coordinates of each unmasked image-pixel, given by `dataset.grids.lp`.
 - $y_c$ and $x_c$ are the (y,x) arc-second `centre` of the light or mass profile used to perform lensing calculations.
 - $q$ is the axis-ratio of the elliptical light or mass profile (`axis_ratio=1.0` for spherical profiles).
 - The elliptical coordinates is rotated by position angle $\phi$, defined counter-clockwise from the positive 
 x-axis.

$q$ and $\phi$ are not used to parameterize a light profile but expresses these  as "elliptical components", 
or `ell_comps` for short:

$\epsilon_{1} =\frac{1-q}{1+q} \sin 2\phi, \,\,$
$\epsilon_{2} =\frac{1-q}{1+q} \cos 2\phi.$

Note that `Ell` is used as shorthand for elliptical and `Sph` for spherical.
"""
profile = al.EllProfile(centre=(0.1, 0.2), ell_comps=(0.1, 0.2))

"""
Transform `dataset.grids.lp` to the centre of profile and rotate it using its angle `phi`.
"""
transformed_grid = profile.transformed_to_reference_frame_grid_from(
    grid=dataset.grids.lp
)

grid_plotter = aplt.Grid2DPlotter(grid=transformed_grid)
grid_plotter.figure_2d()
print(
    f"transformed coordinates of first ten unmasked image-pixels {transformed_grid[0:9]}"
)

"""
Using these transformed (y',x') values we compute the elliptical coordinates $\eta = \sqrt{(x')^2 + (y')^2/q^2}$
"""
elliptical_radii = profile.elliptical_radii_grid_from(grid=transformed_grid)

print(
    f"elliptical coordinates of first ten unmasked image-pixels {elliptical_radii[0:9]}"
)

"""
__Light Profiles (Setup)__

To perform a likelihood evaluation we now compose our lens model.

We first define the light profiles which represents the lens galaxy's light, which will be used to fit the lens 
light.

A light profile is defined by its intensity $I (\eta_{\rm l}) $, for example the Sersic profile:

$I_{\rm  Ser} (\eta_{\rm l}) = I \exp \bigg\{ -k \bigg[ \bigg( \frac{\eta}{R} \bigg)^{\frac{1}{n}} - 1 \bigg] \bigg\}$

Where:

 - $\eta$ are the elliptical coordinates (see above) or the masked image-grid.
 - $I$ is the `intensity`, which controls the overall brightness of the Sersic profile.
 - $n$ is the ``sersic_index``, which via $k$ controls the steepness of the inner profile.
 - $R$ is the `effective_radius`, which defines the arc-second radius of a circle containing half the light.

In this example, we assume our lens is composed of one light profile, an elliptical Sersic which represent the 
bulge of the lens. 

It is uncommon for a lens galaxy observed with interferometer data to have luminous emission, but we show this example
to illustrate how the likelihood function works.
"""
bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

"""
Using the masked 2D grid defined above, we can calculate and plot images of each light profile component in real space.

(The transformation to elliptical coordinates above are built into the `image_2d_from` function and performed 
implicitly).
"""
image_2d_bulge = bulge.image_2d_from(grid=dataset.grid)

bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=dataset.grid)
bulge_plotter.figures_2d(image=True)

"""
__Lens Galaxy Mass__

We next define the mass profiles which represents the lens galaxy's mass, which will be used to ray-trace the 
image-plane 2D grid of (y,x) coordinates to the source-plane so that the source model can be evaluated.

In this example, we assume our lens is composed of an elliptical isothermal mass distribution and external shear.

A mass profile is defined by its convergence $\kappa (\eta)$, which is related to
the surface density of the mass distribution as

$\kappa(\eta)=\frac{\Sigma(\eta)}{\Sigma_\mathrm{crit}},$

where

$\Sigma_\mathrm{crit}=\frac{{\rm c}^2}{4{\rm \pi} {\rm G}}\frac{D_{\rm s}}{D_{\rm l} D_{\rm ls}},$

and

 - `c` is the speed of light.
 - $D_{\rm l}$, $D_{\rm s}$, and $D_{\rm ls}$ are respectively the angular diameter distances to the lens, to the 
 source, and from the lens to the source.

For readers less familiar with lensing, we can think of $\kappa(\eta)$ as a convenient and
dimensionless way to describe how light is gravitationally lensed after assuming a cosmology.

For the for the isothermal profile:

$\kappa(\eta) = \frac{1.0}{1 + q} \bigg( \frac{\theta_{\rm E}}{\eta} \bigg)$

Where:

 - $\theta_{\rm E}$ is the `einstein_radius` (which is rescaled compared to other einstein radius
 definitions).
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

mass_plotter = aplt.MassProfilePlotter(mass_profile=mass, grid=dataset.grid)
mass_plotter.figures_2d(convergence=True)

"""
From each mass profile we can compute its deflection angles, which describe how due to gravitational lensing
image-pixels are ray-traced to the source plane.

The deflection angles are computed by integrating $\kappa$: 

$\vec{{\alpha}}_{\rm x,y} (\vec{x}) = \frac{1}{\pi} \int \frac{\vec{x} - \vec{x'}}{\left | \vec{x} - \vec{x'} \right |^2} \kappa(\vec{x'}) d\vec{x'} \, ,$
"""
deflections_yx_2d = mass.deflections_yx_2d_from(grid=dataset.grid)

mass_plotter = aplt.MassProfilePlotter(mass_profile=mass, grid=dataset.grid)
mass_plotter.figures_2d(deflections_y=True, deflections_x=True)

"""
__Lens Galaxy__

We now combine the light and mass profiles into a single `Galaxy` object for the lens galaxy.

When computing quantities for the light and mass profiles from this object, it computes each individual quantity and 
adds them together. 

For example, for the `bulge`, when it computes their 2D images it computes each individually and then adds
them together.
"""
lens_galaxy = al.Galaxy(redshift=0.5, bulge=bulge, mass=mass, shear=shear)

"""
__Source Galaxy Light Profile__

The source galaxy is fitted using another analytic light profile, in this example another elliptical Sersic.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

"""
__Lens Light__

Compute a 2D image of the lens galaxy's light as the sum of its individual light profiles (the an MGE 
bulge). 

This computes the `lens_image_2d` of each `LightProfile` and adds them together. 
"""
lens_image_2d = lens_galaxy.image_2d_from(grid=dataset.grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Ray Tracing__

To perform lensing calculations we ray-trace every 2d (y,x) coordinate $\theta$ from the image-plane to its (y,x) 
source-plane coordinate $\beta$ using the summed deflection angles $\alpha$ of the mass profiles:

 $\beta = \theta - \alpha(\theta)$

The likelihood function of a source light profile ray-traces two grids from the image-plane to the source-plane:

 1) A 2D grid of (y,x) coordinates aligned with the imaging data's image-pixels.
 
 2) The 2D blurring grid (used for the lens light above) which accounts for pixels at the edge of the mask whose
 light blurs into the mask.
 
The function below computes the 2D deflection angles of the tracer's lens galaxies and subtracts them from the 
image-plane 2D (y,x) coordinates $\theta$ of each grid, thus ray-tracing their coordinates to the source plane to 
compute their $\beta$ values.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

# A list of every grid (e.g. image-plane, source-plane) however we only need the source plane grid with index -1.
traced_grid = tracer.traced_grid_2d_list_from(grid=dataset.grid)[-1]

mat_plot = aplt.MatPlot2D(axis=aplt.Axis(extent=[-1.5, 1.5, -1.5, 1.5]))

grid_plotter = aplt.Grid2DPlotter(grid=traced_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Source Image__

We pass the traced grid of coordinates to the source galaxy to evaluate its 2D image.
"""
source_image_2d = source_galaxy.image_2d_from(grid=traced_grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=traced_grid)
galaxy_plotter.figures_2d(image=True)

"""
__Lens + Source Light Addition__

We add the lens and source galaxy images together, to create an overall image of the strong lens.
"""
image = lens_image_2d + source_image_2d

array_2d_plotter = aplt.Array2DPlotter(array=image)
array_2d_plotter.figure_2d()

"""
If you are familiar with imaging data, you may have seen that a `blurring_image` of pixels surrounding the mask,
whose light is convolved into the masked, is also computed at this point.

For interferometer data, this is not necessary as the Fourier transform of the real-space image to the uv-plane 
does not require that the emission from outside the mask is accounted for.

__Fourier Transform__

Fourier Transform the 2D image of the galaxy above using the Non Uniform Fast Fourier Transform (NUFFT).
"""
visibilities = dataset.transformer.visibilities_from(
    image=image,
)

"""
The Fourier Transform converts the galaxy image from real-space, which is the observed 2D image of the galaxy we 
see with our eyes, to the uv-plane, where the visibilities are measured.

The visibilities are a grid of 2D values representing the real and imaginary components of the visibilities at each
uv-plane coordinate.

If you are not familiar with interferometer data and the uv-plane, you will need to read up on interferometry to
fully understand how this likelihood function works.
"""
grid_2d_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
grid_2d_plotter.figure_2d()


"""
__Likelihood Function__

We now quantify the goodness-of-fit of our galaxy model.

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for parametric galaxy modeling consists of two terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

We now explain what each of these terms mean.

__Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `visibilities`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not. 

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image 
for, corresponding to a fit with a lower likelihood.
"""
model_data = visibilities

residual_map = dataset.data - model_data
normalized_residual_map = residual_map / dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = al.Visibilities(visibilities=chi_squared_map)

grid_2d_plotter = aplt.Grid2DPlotter(grid=chi_squared_map.in_grid)
grid_2d_plotter.figure_2d()

"""
__Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term in the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the galaxy modeling process and has no impact on the 
model we infer.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the galaxy model, by combining the two terms computed above using
the likelihood function defined above.
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(figure_of_merit)

"""
__Fit__

This process to perform a likelihood function evaluation performed via the `FitInterferometer` object.
"""
fit = al.FitInterferometer(dataset=dataset, tracer=tracer)
fit_figure_of_merit = fit.figure_of_merit
print(fit_figure_of_merit)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.subplot_fit()


"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
but **PyAutoGalaxy** supports multiple MCMC and optimization algorithms. 

__Wrap Up__

We have presented a visual step-by-step guide to the parametric likelihood function, which uses 
analytic light profiles to fit the galaxy light.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in the `guides` package:

 - `over_sampling`: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 ray-traced to the source-plane and used to evaluate the light profile more accurately.
"""

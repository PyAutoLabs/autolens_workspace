"""
__Log Likelihood Function: Inversion (Parametric)__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Imaging` data with 
a parametric lens light profile and source light profile (e.g. an elliptical Sersic lens and source).

This script has the following aims:

 - To provide a resource that authors can include in papers using, so that readers can understand the likelihood 
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
from os import path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

In order to perform a likelihood evaluation, we first load a dataset.

This example fits a simulated galaxy where the imaging resolution is 0.1 arcsecond-per-pixel resolution.
"""
dataset_path = path.join("dataset", "imaging", "simple")

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
This guide uses in-built visualization tools for plotting. 

For example, using the `ImagingPlotter` the imaging dataset we perform a likelihood evaluation on is plotted.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

The likelihood is only evaluated using image pixels contained within a 2D mask, which we choose before performing
lens modeling.

Below, we define a 2D circular mask with a 3.0" radius.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
When we plot the masked imaging, only the circular masked region is shown.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=masked_dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, we disable over sampling in this guide by setting `sub_size=1`. 

a full description of over sampling and how to use it is given in `autolens_workspace/*/guides/over_sampling.py`.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sampling=al.OverSamplingDataset(uniform=al.OverSamplingUniform(sub_size=1))
)

"""
__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

These are given by `masked_dataset.grids.uniform`, which we can plot and see is a uniform grid of (y,x) Cartesian 
coordinates which have had the 3.0" circular mask applied.

Each (y,x) coordinate coordinates to the centre of each image-pixel in the dataset, meaning that when this grid is
used to perform ray-tracing and evaluate a light profile the intensity of the profile at the centre of each 
image-pixel is computed, making it straight forward to compute the light profile's image to the image data.
"""
grid_plotter = aplt.Grid2DPlotter(grid=masked_dataset.grids.uniform)
grid_plotter.figure_2d()

print(
    f"(y,x) coordinates of first ten unmasked image-pixels {masked_dataset.grid[0:9]}"
)

"""
To perform lensing calculations we convert this 2D (y,x) grid of coordinates to elliptical coordinates:

 $\eta = \sqrt{(x - x_c)^2 + (y - y_c)^2/q^2}$

Where:

 - $y$ and $x$ are the (y,x) arc-second coordinates of each unmasked image-pixel, given by `masked_dataset.grids.uniform`.
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
First we transform `masked_dataset.grids.uniform` to the centre of profile and rotate it using its angle `phi`.
"""
transformed_grid = profile.transformed_to_reference_frame_grid_from(
    grid=masked_dataset.grids.uniform
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
__Likelihood Setup: Lens Galaxy Light (Setup)__

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
"""
bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

"""
Using the masked 2D grid defined above, we can calculate and plot images of each light profile component.

(The transformation to elliptical coordinates above are built into the `image_2d_from` function and performed 
implicitly).
"""
image_2d_bulge = bulge.image_2d_from(grid=masked_dataset.grid)

bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=masked_dataset.grid)
bulge_plotter.figures_2d(image=True)

"""
__Likelihood Setup: Lens Galaxy Mass__

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

mass_plotter = aplt.MassProfilePlotter(mass_profile=mass, grid=masked_dataset.grid)
mass_plotter.figures_2d(convergence=True)

"""
From each mass profile we can compute its deflection angles, which describe how due to gravitational lensing
image-pixels are ray-traced to the source plane.

The deflection angles are computed by integrating $\kappa$: 

$\vec{{\alpha}}_{\rm x,y} (\vec{x}) = \frac{1}{\pi} \int \frac{\vec{x} - \vec{x'}}{\left | \vec{x} - \vec{x'} \right |^2} \kappa(\vec{x'}) d\vec{x'} \, ,$
"""
deflections_yx_2d = mass.deflections_yx_2d_from(grid=masked_dataset.grid)

mass_plotter = aplt.MassProfilePlotter(mass_profile=mass, grid=masked_dataset.grid)
mass_plotter.figures_2d(deflections_y=True, deflections_x=True)

"""
__Likelihood Setup: Lens Galaxy__

We now combine the light and mass profiles into a single `Galaxy` object for the lens galaxy.

When computing quantities for the light and mass profiles from this object, it computes each individual quantity and 
adds them together. 

For example, for the `bulge`, when it computes their 2D images it computes each individually and then adds
them together.
"""
lens_galaxy = al.Galaxy(redshift=0.5, bulge=bulge, mass=mass, shear=shear)

"""
__Likelihood Setup: Source Galaxy Light Profile__

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
__Likelihood Step 1: Lens Light__

Compute a 2D image of the lens galaxy's light as the sum of its individual light profiles (the `Sersic` 
bulge). 

This computes the `lens_image_2d` of each `LightProfile` and adds them together. 
"""
lens_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=masked_dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
To convolve the lens's 2D image with the imaging data's PSF, we need its `blurring_image`. This represents all flux 
values not within the mask, which are close enough to it that their flux blurs into the mask after PSF convolution.

To compute this, a `blurring_mask` and `blurring_grid` are used, corresponding to these pixels near the edge of the 
actual mask whose light blurs into the image:
"""
lens_blurring_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grids.blurring)

galaxy_plotter = aplt.GalaxyPlotter(
    galaxy=lens_galaxy, grid=masked_dataset.grids.blurring
)
galaxy_plotter.figures_2d(image=True)

"""
__Likelihood Step 2: Ray Tracing__

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
traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

mat_plot = aplt.MatPlot2D(axis=aplt.Axis(extent=[-1.5, 1.5, -1.5, 1.5]))

grid_plotter = aplt.Grid2DPlotter(grid=traced_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

traced_blurring_grid = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.blurring
)[-1]

mat_plot = aplt.MatPlot2D(axis=aplt.Axis(extent=[-1.5, 1.5, -1.5, 1.5]))

grid_plotter = aplt.Grid2DPlotter(grid=traced_blurring_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Likelihood Step 3: Source Image__

We pass the traced grid and blurring grid of coordinates to the source galaxy to evaluate its 2D image.
"""
source_image_2d = source_galaxy.image_2d_from(grid=traced_grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=traced_grid)
galaxy_plotter.figures_2d(image=True)

source_blurring_image_2d = source_galaxy.image_2d_from(grid=traced_blurring_grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_galaxy, grid=traced_blurring_grid)
galaxy_plotter.figures_2d(image=True)

"""
__Likelihood Step 4: Lens + Source Light Addition__

We add the lens and source galaxy images and blurring together, to create an overall image of the strong lens.
"""
image = lens_image_2d + source_image_2d

array_2d_plotter = aplt.Array2DPlotter(array=image)
array_2d_plotter.figure_2d()

blurring_image_2d = lens_blurring_image_2d + source_blurring_image_2d

array_2d_plotter = aplt.Array2DPlotter(array=blurring_image_2d)
array_2d_plotter.figure_2d()

"""
__Likelihood Step 5: Convolution__

Convolve the 2D image of the lens and source above with the PSF in real-space (as opposed to via an FFT) using 
a `Convolver`.
"""
convolved_image_2d = masked_dataset.convolver.convolve_image(
    image=image, blurring_image=blurring_image_2d
)

array_2d_plotter = aplt.Array2DPlotter(array=convolved_image_2d)
array_2d_plotter.figure_2d()

"""
__Likelihood Step 6: Likelihood Function__

We now quantify the goodness-of-fit of our lens and source model.

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for parametric lens modeling consists of two terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

We now explain what each of these terms mean.

__Likelihood Step 8: Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `convolved_image_2d`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not. 

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image 
for, corresponding to a fit with a lower likelihood.
"""
model_image = convolved_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = al.Array2D(values=chi_squared_map, mask=mask)

array_2d_plotter = aplt.Array2DPlotter(array=chi_squared_map)
array_2d_plotter.figure_2d()

"""
__Likelihood Step 7: Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term ins the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the lens modeling process and has no impact on the 
model we infer.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Likelihood Step 8: Calculate The Log Likelihood!__

We made it!

We can now, finally, compute the `log_likelihood` of the lens model, by combining the two terms computed above using
the likelihood function defined above.
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(figure_of_merit)

"""
__Fit__

This 11 step process to perform a likelihood function evaluation is what is performed in the `FitImaging` object.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)
fit_figure_of_merit = fit.figure_of_merit
print(fit_figure_of_merit)


"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `Nautilus` (https://github.com/joshspeagle/Nautilus)
but **PyAutoLens** supports multiple MCMC and optimization algorithms. 

__Wrap Up__

We have presented a visual step-by-step guide to the **PyAutoLens** parametric likelihood function, which uses analytic
light profiles to fit the lens and source light.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in this package. In brief, these describe:

 - **Sub-gridding**: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 ray-traced to the source-plane and used to evaluate the light profile more accurately.
"""

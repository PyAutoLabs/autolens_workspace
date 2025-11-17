"""
__Log Likelihood Function: Linear Light Profile__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Imaging` data with
parametric linear light profiles (e.g. a Sersic bulge and Exponential disk).

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood)
given the light profile's other parameters.

This script has the following aims:

 - To provide a resource that authors can include in papers, so that readers can understand the likelihood
 function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.

__Prerequisites__

The likelihood function of a linear light profile builds on that used for standard parametric light profiles,
therefore you must read the following notebooks before this script:

- `light_profile/likelihood_function.ipynb`.
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
__Dataset__

Following the `light_profile/log_likelihood_function.py` script, we load and mask an `Imaging` dataset and
set oversampling to 1.
"""
dataset_path = Path("dataset", "imaging", "simple")

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

masked_dataset = dataset.apply_mask(mask=mask)

masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

dataset_plotter = aplt.ImagingPlotter(dataset=masked_dataset)
dataset_plotter.subplot_dataset()

"""
__Masked Image Grid__

To perform galaxy calculations we used a 2D image-plane grid of (y,x) coordinates, which evaluated the
emission of galaxy light profiles created as `LightProfile` objects.

The code below repeats that used in `light_profile/log_likelihood_function.py` to show how this was done.
"""
bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=4.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

lens_galaxy = al.Galaxy(redshift=0.5, bulge=bulge, mass=mass, shear=shear)

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

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

image = tracer.image_2d_from(grid=masked_dataset.grids.lp)

"""
__Linear Light Profiles__

To use a linear light profile, whose `intensity` is computed via linear algebra, we simply use the `lp_Linear`
module instead of the `lp` module used throughout other example scripts. 

The `intensity` parameter of the light profile is no longer passed into the light profiles created via the
`lp_linear` module, as it is inferred via linear algebra.

In this example, we assume our galaxy is composed of two light profiles, an elliptical Sersic and Exponential (a Sersic
where `sersic_index=4`) which represent the bulge and disk of the galaxy. 
"""
bulge = al.lp_linear.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    effective_radius=0.6,
    sersic_index=3.0,
)

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

lens_galaxy = al.Galaxy(redshift=0.5, bulge=bulge, mass=mass, shear=shear)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

"""
Internally in the source code, linear light profiles have an `intensity` parameter, but its value is always set to 
1.0. It will be clear why this is later in the script.
"""
print("Bulge Internal Intensity:")
print(lens_galaxy.bulge.intensity)

print("Disk Internal Intensity:")
print(source_galaxy.bulge.intensity)

"""
Like standard light profiles, we can compute images of each linear light profile, but their overall
normalization is arbitrary given that the internal `intensity` value of 1.0 is used.
"""
image_2d_bulge = lens_galaxy.bulge.image_2d_from(grid=masked_dataset.grid)
image_2d_disk = source_galaxy.bulge.image_2d_from(grid=masked_dataset.grid)

"""
If we try and plot a linear light profile using a plotter, an exception is raised.

This is to ensure that a user does not plot and interpret the intensity of a linear light profile, as it is not a
physical quantity. Plotting only works after a linear light profile has had its `intensity` computed via linear
algebra.

Uncomment and run the code below to see the exception.
"""
print("This will raise an exception")

# bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=masked_dataset.grid)

"""
We now put them together in a `Tracer` object.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__LightProfileLinearObjFuncList__

For standard light profiles, we combined our linear light profiles into a single `Galaxies` object. The 
galaxies object computed each individual light profile's image and added them together.

This no longer occurs for linear light profiles, instead linear light profiles are passed into the 
`LightProfileLinearObjFuncList` object, which acts as an interface between the linear light profiles and the
linear algebra used to compute their intensity via the inversion.

The quantities used to compute the image, blurring image and blurred image of each light profiles (the
dataset grid, PSF, etc.) are passed to the `LightProfileLinearObjFuncList` object, because it internally uses these
to compute each linear light profile image to set up the linear algebra.

For lensing, this means we have to use a different `LightProfileLinearObjFuncList` object for each plane, because
each plane has its own ray-traced grid of (y,x) coordinates. Below, we set up the first `LightProfileLinearObjFuncList`,
which uses the image-plane grid and lens galaxy bulge.
"""
lp_linear_func_lens = al.LightProfileLinearObjFuncList(
    grid=masked_dataset.grids.lp,
    blurring_grid=masked_dataset.grids.blurring,
    light_profile_list=[tracer.galaxies[0].bulge],
    regularization=None,
)

"""
This has a property `params` which is the number of intensity values that are computed via the inversion,
which because we have 1 light profiles is equal to 1.

The `params` defines the dimensions of many of the matrices used in the linear algebra we discuss below.
"""
print("Number of Parameters (Intensity Values) in Linear Algebra:")
print(lp_linear_func_lens.params)

"""
__Mapping Matrix__

The `mapping_matrix` is a matrix where each column is an image of each linear light profiles (assuming its 
intensity is 1.0), not accounting for the PSF convolution.

It has dimensions `(total_image_pixels, total_linear_light_profiles)`.
"""
mapping_matrix = lp_linear_func_lens.mapping_matrix

"""
Printing the first column of the mapping matrix shows the image of the lens bulge light profile.
"""
bulge_image = mapping_matrix[:, 0]
print(bulge_image)
print(image_2d_bulge.slim)

"""
A 2D plot of the `mapping_matrix` shows each light profile image in 1D, which is a bit odd to look at but
is a good way to think about the linear algebra.
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
We now make the second `LightProfileLinearObjFuncList`, which uses the ray-traced source-plane grid and source
galaxy bulge light profile.
"""
traced_grids_of_planes_list = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.lp
)
traced_blurring_grids_of_planes_list = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.blurring
)

lp_linear_func_source = al.LightProfileLinearObjFuncList(
    grid=traced_grids_of_planes_list[-1],
    blurring_grid=traced_blurring_grids_of_planes_list[1],
    psf=masked_dataset.psf,
    light_profile_list=[tracer.galaxies[1].bulge],
    regularization=None,
)

"""
Printing the first column of the mapping matrix shows the image of the source bulge light profile.
"""
mapping_matrix = lp_linear_func_source.mapping_matrix

bulge_image = mapping_matrix[:, 0]
print(bulge_image)
print(image_2d_bulge.slim)

"""
__Combining Matrices__

The linear algebra system solves for all light profile `intensity` values at once, so we need to combine
each of their individual mapping matrices into a single matrix.

This is done via `hstack`, which stacks the two matrices horizontally to create a single matrix with dimensions
`(total_image_pixels, total_linear_light_profiles)`, where the latter dimension is now 2 because we have combined
two linear light profiles.
"""
mapping_matrix = np.hstack(
    [lp_linear_func_lens.mapping_matrix, lp_linear_func_source.mapping_matrix]
)

"""
__Blurred Mapping Matrix ($f$)__

The `mapping_matrix` does not account for the blurring of the light profile images by the PSF and therefore 
is not used directly to compute the likelihood.

Instead, we create a `blurred_mapping_matrix` which does account for this blurring. This is computed by 
convolving each light profile image with the PSF.

The `blurred_mapping_matrix` is a matrix analogous to the mapping matrix, but where each column is the image of each
light profile after it has been blurred by the PSF.

This operation does not change the dimensions of the mapping matrix, meaning the `blurred_mapping_matrix` also has
dimensions `(total_image_pixels, total_linear_light_profiles)`. 

The property is actually called `operated_mapping_matrix_override` for two reasons: 

1) The operated signifies that this matrix could have any operation applied to it, it just happens for imaging
   data that this operation is a convolution with the PSF.

2) The `override` signifies that in the source code is changes how the `operated_mapping_matrix` is computed internally. 
   This is important if you are looking at the source code, but not important for the description of the likelihood 
   function in this guide.
   
We have two separate `LightProfileLinearObjFuncList` objects, one for the lens and one for the source, we combine
the `blurred_mapping_matrix` of each via `hstack` to create a single `blurred_mapping_matrix` that represents
the linear system that will be solved for both.
"""
blurred_mapping_matrix = np.hstack(
    [
        lp_linear_func_lens.operated_mapping_matrix_override,
        lp_linear_func_source.operated_mapping_matrix_override,
    ],
)

"""
Printing the first column of the mapping matrix shows the blurred image of the bulge light profile, the
second the blurred image of the source light profile.
"""
print(blurred_mapping_matrix[:, 0])
print(blurred_mapping_matrix[:, 1])

"""
A 2D plot of the `mapping_matrix` shows each light profile image in 1D, with a PSF convolution applied.
"""
plt.imshow(
    blurred_mapping_matrix,
    aspect=(blurred_mapping_matrix.shape[1] / blurred_mapping_matrix.shape[0]),
)
plt.show()
plt.close()

"""
Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587) (hereafter WD03) introduce the linear inversion formalism 
used to compute the intensity values of the linear light profiles. In WD03, the science case is centred around strong
gravitational lensing and the galaxy is reconstructed on a rectangular grid of pixels, as opposed to linear light 
profiles.

However, the mathematics of the WD03 linear inversion formalism is the same as that used here, therefore this guide 
describes which quantities in the linear inversion formalism map to the equations given in WD03. The pixelized 
reconstruction methods, available in the code but described in the `pixelization` likelihood function guide, 
also follow the WD03 formalism.

The `blurred_mapping_matrix` is denoted $f_{ij}$ where $i$ maps over all $I$ linear light profiles and $j$ maps 
over all $J$ image pixels. 

For example: 

 - $f_{0, 1} = 0.3$ indicates that image-pixel $2$ maps to linear light profile $1$ with an intensity in that image 
   pixel of $0.3$ after PSF convolution.

The indexing of the `mapping_matrix` is reversed compared to the notation of WD03 (e.g. image pixels
are the first entry of `mapping_matrix` whereas for $f$ they are the second index).
"""
print(
    f"Mapping between image pixel 0 and linear light profile pixel 1 = {mapping_matrix[0, 1]}"
)

"""
__Data Vector (D)__

To solve for the linear light profile intensities we now pose the problem as a linear inversion.

This requires us to convert the `blurred_mapping_matrix` and our `data` and `noise map` into matrices of certain 
dimensions. 

The `data_vector`, $D$, is the first matrix and it has dimensions `(total_linear_light_profiles,)`.

In WD03 (https://arxiv.org/abs/astro-ph/0302587) the data vector is given by: 

 $\vec{D}_{i} = \sum_{\rm  j=1}^{J}f_{ij}(d_{j} - b_{j})/\sigma_{j}^2 \, \, .$

Where:

 - $d_{\rm j}$ are the image-pixel data flux values.
 - $b_{\rm j}$ are the image values of all standard light profiles (therefore $d_{\rm  j} - b_{\rm j}$ is 
 the data minus any standard light profiles).
 - $\sigma{\rm _j}^2$ are the statistical uncertainties of each image-pixel value.

$i$ maps over all $I$ linear light profiles and $j$ maps over all $J$ image pixels. 

This equation highlights a first aspect of linear inversions, if we are combining standard light profiles (which
have an input `intensity` value) with linear light profiles, the inversion is performed on the data minus
the standard light profile images. In this example, we have no standard light profiles and therefore the data
vector uses the data directly.
"""
data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(masked_dataset.data),
    noise_map=np.array(masked_dataset.noise_map),
)

"""
$D$'s meaning is a bit abstract, it essentially weights each linear light profile's `intensity` based on how it
maps to the data, so that the linear algebra can compute the `intensity` values that best-fit the data.

We can plot $D$ as a column vector:
"""
plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

"""
The dimensions of $D$ are the number of linear light profiles, which in this case is 2.
"""
print("Data Vector:")
print(data_vector)
print(data_vector.shape)

"""
__Curvature Matrix (F)__

The `curvature_matrix` $F$ is the second matrix and it has 
dimensions `(total_linear_light_profiles, total_linear_light_profiles)`.

In WD03 (https://arxiv.org/abs/astro-ph/0302587) the curvature matrix is a 2D matrix given by:

 ${F}_{ik} = \sum_{\rm  j=1}^{J}f_{ij}f_{kj}/\sigma_{j}^2 \, \, .$

NOTE: this notation implicitly assumes a summation over $K$, where $k$ runs over all linear light profile indexes $K$.

Note how summation over $J$ runs over $f$ twice, such that every entry of $F$ is the sum of the multiplication
between all values in every two columns of $f$.

For example, $F_{0,1}$ is the sum of every blurred image pixels values in $f$ of linear light profile 0 multiplied by
every blurred image pixel value of linear light profile 1.

$F$'s meaning is also a bit abstract, but it essentially quantifies how much each linear light profile's image
overlaps with every other linear light profile's image, weighted by the noise in the data. This is what combined with
the `data_vector` allows the inversion to compute the `intensity` values that best-fit the data.
"""
curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()


"""
__Reconstruction (Positive-Negative)__

The following chi-squared is minimized when we perform the inversion and reconstruct the galaxy:

$\chi^2 = \sum_{\rm  j=1}^{J} \bigg[ \frac{(\sum_{\rm  i=1}^{I} s_{i} f_{ij}) + b_{j} - d_{j}}{\sigma_{j}} \bigg]$

Where $s$ is the `intensity` values in all $I$ linear light profile images.

The solution for $s$ is therefore given by (equation 5 WD03):

 $s = F^{-1} D$

We can compute this using NumPy linear algebra and the `solve` function.

However, this function allows for the solved `intensity` values to be negative. For linear light profiles which
are a good fit to the data, this is unlikely to happen and the `intensity` values will be positive. However, 
for more complex models this may not be the case. Below, we describes how we can ensure the `intensity` values
are positive.
"""
reconstruction = np.linalg.solve(curvature_matrix, data_vector)

"""
The `reconstruction` is a 1D vector of length equal to the number of linear light profiles, which in this case is 2.

Each value represents the intensity of the linear light profile.

In this example, both values are positive, but remember that this is not guaranteed for all linear inversions
that are solve using this method.
"""
print("Reconstruction (S) of Linear Light Profiles Intensity:")
print(reconstruction)

"""
__Reconstruction (Positive Only)__

The linear algebra can be solved for with the constraint that all solutions, and therefore all `intensity` values,
are positive. 

This could be achieved by using the `scipy` `nnls` non-negative least squares solver.

The nnls poses the problem slightly different than the code above. It solves for the `intensity` values in an
iterative manner meaning that it is slower. It does not use `data_vector` $D$ and `curvature_matrix` $F$ but instead
works directly with the `blurred_mapping_matrix` $f$ and the data and noise-map.

The `nnls` function is therefore computationally slow, especially for cases where there are many linear light profiles 
or even  more complex linear inversions like a pixelized reconstruction.

The source code therefore uses a "fast nnls" algorithm, which is an adaptation of the algorithm found at
this URL: https://github.com/jvendrow/fnnls

Unlike the scipy nnls function, the fnnls method uses the `data_vector` $D$ and `curvature_matrix` $F$ to solve for
the `intensity` values. This provides it with additional information about the linear algebra problem, which is
why it is faster.

The function `reconstruction_positive_only_from` uses the `fnnls` algorithm to compute the `intensity` values
of the linear light profiles, ensuring they are positive.
"""
reconstruction = al.util.inversion.reconstruction_positive_only_from(
    data_vector=data_vector,
    curvature_reg_matrix=curvature_matrix,  # ignore _reg_ tag in this guide
)

print(reconstruction)

"""
__Image Reconstruction__

Using the reconstructed `intensity` values we can map the reconstruction back to the image plane (via 
the `blurred mapping_matrix`) and produce a reconstruction of the image data.
"""
mapped_reconstructed_image_2d = (
    al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
)

mapped_reconstructed_image_2d = al.Array2D(
    values=mapped_reconstructed_image_2d, mask=mask
)

array_2d_plotter = aplt.Array2DPlotter(array=mapped_reconstructed_image_2d)
array_2d_plotter.figure_2d()


"""
__Likelihood Function__

We now quantify the goodness-of-fit of our galaxy model.

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for parametric galaxy modeling, even if linear light profiles are used, consists of two terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

We now explain what each of these terms mean.

__Chi Squared__

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
model_image = mapped_reconstructed_image_2d

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
__Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term in the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the galaxy modeling process and has no impact on the 
model we infer.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the galaxy model, by combining the two terms computed above using
the likelihood function defined above.
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(figure_of_merit)


"""
__Fit__

This process to perform a likelihood function evaluation is what is performed in the `FitImaging` object.
"""
fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=False, use_border_relocator=True
    ),
)
fit_log_evidence = fit.log_evidence
print(fit_log_evidence)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
The fit contains an `Inversion` object, which handles all the linear algebra we have covered in this script.
"""
print(fit.inversion)
print(fit.inversion.data_vector)
print(fit.inversion.curvature_matrix)
print(fit.inversion.reconstruction)
print(fit.inversion.mapped_reconstructed_image)

"""
The `Inversion` object can be computed from a tracer and a dataset, by passing them to the `TracerToInversion` object.

This objects handles a lot of extra functionality that we have not covered in this script, such as:

- Separating out the linear light profiles from the standard light profiles.
- Separating out objects which reconstruct the galaxy using a pixelized reconstruction, which are passed into
  the `Inversion` object as well.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=masked_dataset,
)

inversion = tracer_to_inversion.inversion

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
but **PyAutoLens** supports multiple MCMC and optimization algorithms. 

For linear light profiles, the reduced number of free parameters (e.g. the `intensity` values are solved for
via linear algebra and not a dimension of the non-linear parameter space) means that the sampler converges in fewer
iterations and is less likely to infer a local maximum.

__Wrap Up__

We have presented a visual step-by-step guide to the parametric linear light profile likelihood function, which uses 
analytic light profiles to fit the galaxies light and solve for the `intensity` values via linear algebra.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in the `guides` package:

 - `over_sampling`: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 ray-traced to the source-plane and used to evaluate the light profile more accurately.
"""

"""
__Log Likelihood Function: Multi Gaussian Expansion__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Imaging` data with
a multi-Gaussian expansion (MGE), which is a superposition of multiple 2D Gaussian linear light profiles.

You should be familiar with the `log_likelihood_function` of a linear light profile before reading this script,
which is described in the `log_likelihood_function/imaging/linear_light_profile/likelihood_function.ipynb` notebook.

This script has the following aims:

 - To provide a resource that authors can include in papers, so that readers can understand the likelihood
 function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.

__Prerequisites__

The likelihood function of a multi Gaussian expansion builds on that used for standard light profiles and
linear light profiles, therefore you must read the following notebooks before this script:

- `light_profile/likelihood_function.ipynb`.
- `linear_light_profile/likelihood_function.ipynb`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Following the `linear_light_profile/log_likelihood_function.py` script, we load and mask an `Imaging` dataset and
set oversampling to 1.

This example fits a simulated galaxy where galaxy has an asymmetric light distribution, which cannot be accurately 
fitted with `Sersic` profile and therefore requires a multi-Gaussian expansion to fit accurately.
"""
dataset_name = "lens_light_asymmetric"
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
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

image = tracer.image_2d_from(grid=masked_dataset.grids.lp)

"""
__Multiple Gaussians & Linear Light Profiles__

To use a linear light profile, whose `intensity` is computed via linear algebra, we simply use the `lp_Linear`
module instead of the `lp` module used throughout other example scripts. 

The `intensity` parameter of the light profile is no longer passed into the light profiles created via the
`lp_linear` module, as it is inferred via linear algebra.

For a multi-Gaussian expansion, we use 30 linear light profile `Gaussian`'s, which is easily achieved by creating a
list of `Gaussian` objects via a for loop.
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

basis_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        sigma=10 ** log10_sigma_list[i],
    )

    basis_gaussian_list.append(gaussian)

"""
__Basis__

For a multi-Gaussian expansion (and other mdoels where the light profile is a superposition of multiple light profiles),
the list of linear light profiles is passed to the `Basis` class.

The `Basis` basically stores all these light profiles into a single object such that they can collectively be used to
perform the fit.
"""
basis = al.lp_basis.Basis(profile_list=basis_gaussian_list)

"""
The `Basis` is composed of many Gaussians, each with different sizes (the `sigma` value) and therefore capturing
emission on different scales.

These Gaussians are visualized below using a `BasisPlotter`, which shows that the Gaussians expand in size as the
sigma value increases, in log10 increments.

This figure is a brilliant way to visualize the multi-Gaussian expansion, showing the 30 different Gaussian light
profiles that will be used perform the expansion on the data.

Below, we will discuss how linear light profiles cannot be visualized (an exception is raised if you try). Therefore
below we make a separate `Basis` object of `Gaussians` using standard light profiles with input `intensity` values,
which we can visualize.
"""
basis_plot_gaussian_list = []

for i in range(total_gaussians):
    gaussian = al.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=10 ** log10_sigma_list[i],
    )

    basis_plot_gaussian_list.append(gaussian)

basis_plot = al.lp_basis.Basis(profile_list=basis_plot_gaussian_list)

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

basis_plotter = aplt.BasisPlotter(basis=basis_plot, grid=grid)
basis_plotter.subplot_image()

"""
Internally in the source code, linear light profiles have an `intensity` parameter, but its value is always set to 
1.0. 

This can be seen by printing the intensity of the first two Gaussians in the basis.
"""
print("Basis Internal Intensity Of First Gaussian:")
print(basis.light_profile_list[0].intensity)

print("Basis Internal Intensity Of Second Gaussian:")
print(basis.light_profile_list[1].intensity)

"""
Like standard light profiles, we can compute images of each linear light profile in the basis, but their overall
normalization is arbitrary given that the internal `intensity` value of 1.0 is used.
"""
image_2d_basis_0 = basis.light_profile_list[0].image_2d_from(grid=masked_dataset.grid)
image_2d_basis_1 = basis.light_profile_list[1].image_2d_from(grid=masked_dataset.grid)

"""
If we try and plot a linear light profile using a plotter, an exception is raised.

This is to ensure that a user does not plot and interpret the intensity of a linear light profile, as it is not a
physical quantity. Plotting only works after a linear light profile has had its `intensity` computed via linear
algebra.

Uncomment and run the code below to see the exception.

Note that the `BasisPlotter` used above did not raise an exception, because its intended purpose is to visualize
the basis light profiles and not the intensity of the light profiles.
"""
print("This will raise an exception")

# basis_plotter = aplt.LightProfilePlotter(light_profile=basis, grid=masked_dataset.grid)

"""
We now set up a `Tracer` using the MGE for the lens galaxy.
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

lens_galaxy = al.Galaxy(redshift=0.5, bulge=basis, mass=mass, shear=shear)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Comparison To Linear Light Profiles Example__

The text below is nearly identical to the `linear_light_profile/likelihood_function.ipynb` example, because the 
linear algebra and likelihood function of a multi-Gaussian expansion is essentially identical to that of a single 
linear light profile.

The key difference between the linear light profile and multi-Gaussian expansion calculation is essentially the
following:

- The `mapping_matrix`, which for 2 linear light profiles had dimensions `(total_image_pixels, 2)`, now has dimensions
 `(total_image_pixels, 30)`, corresponding to the 30 different Gaussian light profiles.
 
- Each column of this `mapping_matrix` is the image of each Gaussian light profile, as opposed to the Sersic and
  Exponential light profiles used in the previous example.

- The use of the positive only solver for the reconstruction is more important for an MGE, because MGEs can otherwise
  infer unphysical solutions where the Gaussians alternate between large positive and large negative values.

Other than the above change, the calculation is performed in an identical manner to the linear light profile example,
with the `data_vector`, `curvature_matrix`, `reconstruction` and `log_likelihood` all computed in the same way
with the same dimensions. 

__LightProfileLinearObjFuncList__

For standard light profiles, we combined our linear light profiles into a single `Galaxies` object. The 
galaxies object computed each individual light profile's image and added them together.

This no longer occurs for linear light profiles, instead linear light profiles are passed into the 
`LightProfileLinearObjFuncList` object, which acts as an interface between the linear light profiles and the
linear algebra used to compute their intensity via the inversion.

For an MGE, we input the whole `Basis` object into the `LightProfileLinearObjFuncList` object, which
contains all the Gaussian linmear light profiles.

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
    psf=masked_dataset.psf,
    light_profile_list=basis.light_profile_list,
    regularization=None,
)

"""
This has a property `params` which is the number of intensity values that are computed via the inversion,
which because we have 30 Gaussian linear light profiles is equal to 30.

The `params` defines the dimensions of many of the matrices used in the linear algebra we discuss below.
"""
print("Number of Parameters (Intensity Values) in Linear Algebra:")
print(lp_linear_func_lens.params)

"""
__Combining Matrices__

In the `linear_light_profile/log_likelihood_function.py` example, we used two `LightProfileLinearObjFuncList` to set
up the linear algebra for the different planes of the `Tracer`, which we do again below.

In this example the source is a single Sersic linear light profile, but it could easily be an MGE itself.
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
__Mapping Matrix__

The `mapping_matrix` is a matrix where each column is an image of each Gaussian linear light profiles (assuming its 
intensity is 1.0), not accounting for the PSF convolution.

We combine the `mapping_matrix` of the lens and source plane into a single matrix, which is used to compute the
`blurred_mapping_matrix` and the `data_vector` below.

It has dimensions `(total_image_pixels, total_linear_light_profiles)` = `(total_image_pixels, 31)`.
"""
mapping_matrix = np.hstack(
    [lp_linear_func_lens.mapping_matrix, lp_linear_func_source.mapping_matrix]
)

"""
Printing the first column of the mapping matrix shows the image of the basis light profile.
"""
basis_image = mapping_matrix[:, 0]
print(basis_image)
print(image_2d_basis_0.slim)

"""
A 2D plot of the `mapping_matrix` shows each light profile image in 1D, which is a bit odd to look at but
is a good way to think about the linear algebra.
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
__Blurred Mapping Matrix ($f$)__

The `mapping_matrix` does not account for the blurring of the light profile images by the PSF and therefore 
is not used directly to compute the likelihood.

Instead, we create a `blurred_mapping_matrix` which does account for this blurring. This is computed by 
convolving each light profile image with the PSF.

The `blurred_mapping_matrix` is a matrix analogous to the mapping matrix, but where each column is the image of each
light profile after it has been blurred by the PSF.

This operation does not change the dimensions of the mapping matrix, meaning the `blurred_mapping_matrix` also has
dimensions `(total_image_pixels, total_rectangular_pixels)`. 

The property is actually called `operated_mapping_matrix_override` for two reasons: 

1) The operated signifies that this matrix could have any operation applied to it, it just happens for imaging
   data that this operation is a convolution with the PSF.

2) The `override` signifies that in the source code is changes how the `operated_mapping_matrix` is computed internally. 
   This is important if you are looking at the source code, but not important for the description of the likelihood 
   function in this guide.
"""
blurred_mapping_matrix = np.hstack(
    [
        lp_linear_func_lens.operated_mapping_matrix_override,
        lp_linear_func_source.operated_mapping_matrix_override,
    ],
)

"""
Printing the first column of the mapping matrix shows the blurred image of the basis light profile.
"""
basis_image = blurred_mapping_matrix[:, 0]
print(basis_image)

"""
A 2D plot of the `mapping_matrix` shows each light profile image in 1D, with a PSF convolution applied.
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
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
    f"Mapping between image pixel 0 and Gaussian linear light profile pixel 1 = {mapping_matrix[0, 1]}"
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

However, this function allows for the solved `intensity` values to be negative, which are unphysical values for
describing the light profile of a galaxy. 

For a multi-Gaussian expansion, it is common for the inferred solution to contain negative `intensity` values. A common
solution is one where the Gaussians alternate between large positive and large negative values, creating an almost
"ringing" effect in the reconstruction. This is a very unphysical solution and one we want to avoid.

We are able to illustrate this now, first by solving the linear algebra and then printing the `intensity` values.
"""
reconstruction = np.linalg.solve(curvature_matrix, data_vector)

"""
The `reconstruction` is a 1D vector of length equal to the number of Gaussian linear light profiles, which in this case 
is 30.

Each value represents the solved for `intensity` of the Gaussian linear light profile.

In this example, the values alternate between positive and negative, indicating a solution that is not physical
and one we must avoid.
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
or even more complex linear inversions like a pixelized reconstruction.

The source code therefore uses a "fast nnls" algorithm, which is an adaptation of the algorithm found at
this URL: https://github.com/jvendrow/fnnls

Unlike the scipy nnls function, the fnnls method uses the `data_vector` $D$ and `curvature_matrix` $F$ to solve for
the `intensity` values. This provides it with additional information about the linear algebra problem, which is
why it is faster.

The function `reconstruction_positive_only_from` uses the `fnnls` algorithm to compute the `intensity` values
of the linear light profiles, ensuring they are positive.

However, the code below by itself actually produces a `LinAlgError` because the `curvature_matrix` is singular. Uncomment
the code below to see this error.
"""
# reconstruction = al.util.inversion.reconstruction_positive_only_from(
#     data_vector=data_vector,
#     curvature_reg_matrix=curvature_matrix,  # ignore _reg_ tag in this guide
# )
#
# print(reconstruction)

"""
To make the `curvature_matrix` non-singular, we simply add small numerical values to its diagonal elements. 

This is effectively add a small degree of "zeroth order" regularization to the inversion, which is sufficient to make
the matrix non-singular and ensure the inversion can be performed.

There are a variety of ways to regularize the inversion, and these can be manually input into the `Basis` object.
However, for a multi-Gaussian expansion, testing has shown that adding a small degree of zeroth order regularization
in conjunction with a positive-only solution is sufficient to ensure the inversion is robust for all reasonable
science cases.

In practise, the code only adds these small numerical values to the diagonal of the curvature matrix for elements
which have no other regularization applied to them. Therefore, in the function call below we input 
`no_regularization_index_list=range(30)`, which tells the function to add small numerical values to all 30
diagonal values corresponding to the 30 Gaussian linear light profiles.
"""
curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix,
    noise_map=masked_dataset.noise_map,
    add_to_curvature_diag=True,
    no_regularization_index_list=list(range(30)),
)

"""
The `reconstruction` can now be computed successfully without a linear algebra error.
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
galaxy = al.Galaxy(
    redshift=0.5,
    basis=basis,
)

galaxies = al.Galaxies(galaxies=[galaxy])

fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_border_relocator=True),
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
__Galaxy Modeling__

To fit a galaxy model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
multiple MCMC and optimization algorithms are supported.

For an MGE, the reduced number of free parameters (e.g. the `intensity` values are solved for
via linear algebra and not a dimension of the non-linear parameter space) means that the sampler converges in fewer
iterations and is less likely to infer a local maximum. 

Furthermore, the size of the lens galaxy, controlled by the `sigma` values of the Gaussians, are also all fixed
and not non-linear free parameters. This further simplifies the non-linear parameter space.

__Wrap Up__

We have presented a visual step-by-step guide to the multi Gaussian expansion likelihood function, which uses 
many 2D Gaussians to fit the galaxy light and solve for the `intensity` values via linear algebra.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in the `guides` package:

 - `over_sampling`: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 ray-traced to the source-plane and used to evaluate the light profile more accurately.
"""

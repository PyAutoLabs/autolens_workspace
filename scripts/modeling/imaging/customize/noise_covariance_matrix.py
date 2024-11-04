"""
Customize: Noise Covariance Matrix
==================================

This example demonstrates how to account for correlated noise in a dataset, using a noise covariance matrix. This
changes the goodness-of-fit measure (the chi-squared term of the log likelihood function).

__Advantages__

For datasets with high amounts of correlated noise, this will give a more accurate analysis.

__Disadvantages__

It can be challenging to properly measure the noise covariance matrix and for high resolution datasets
can pose issues in terms of storing the matrix in memory.

__Visualization__

It is difficult to visualize quantities like the `normalized_residual_map` and `chi_squared_map` in a way that
illustrates the noise covariance.

These quantities are therefore visualized using the diagonal of the `noise_covariance_matrix`.

Because these visuals do not account for noise covariance, they are not fully representative of the overall fit to
the data.

__Inversions__

Only fits using regular light profiles support noise covariance fits. Inversions (e.g. using linear light profiles
or a pixelization) do not support noise covariance, as it is not currently accounted for in the linear algebra
calculations.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Noise Covariance Matrix__

We define our noise covariance matrix, which has shape of [image_pixels, image_pixels] in the data. 

For this simple example, we define a noise covariance matrix where: 

 - All values on the diagonal are 1.
 - All values on the neighboring diagonals are 0.5, meaning there is covariance between image-pixels are their
 neighbors.
 - All other values are zero, meaning covariance does not stretch beyond only neighboring image-pixels.

For your science, you will have likely estimated the noise-covariance matrix during the data reductions and would load
it below from a saved NumPy array or other storage format.
"""
image_shape_native = (100, 100)
total_image_pixels = image_shape_native[0] * image_shape_native[1]

noise_covariance_matrix = np.zeros(shape=(total_image_pixels, total_image_pixels))

for i in range(total_image_pixels):
    noise_covariance_matrix[i, i] = 1.0

for i in range(total_image_pixels - 1):
    noise_covariance_matrix[i + 1, i] = 0.5
    noise_covariance_matrix[i, i + 1] = 0.5

"""
__Dataset__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files  zoom_around_mask: true            # If True, plots of data structures with a mask automatically zoom in the masked region.

Note how below we include the noise covariance matrix as part of the input.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
    noise_covariance_matrix=noise_covariance_matrix,
)

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.

The mask is also applied to the `noise_covariance_matrix`, to ensure only covariance within the mask is accounted for.

This changes the `noise_covariance_matrix` from `shape=(total_image_pixels, total_image_pixels)` to 
`shape=`pixels_in_mask, pixels_in_mask`).
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

print(dataset.noise_covariance_matrix.shape)

dataset = dataset.apply_mask(mask=mask)

print(dataset.noise_covariance_matrix.shape)

"""
__Model + Search + Analysis__ 

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!

This model-fit implicitly uses the noise covariance matrix when computing the chi-squared and log likelihood!
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=path.join("imaging", "customize"),
    name="noise_covariance_matrix",
    unique_tag=dataset_name,
)

analysis = al.AnalysisImaging(dataset=dataset)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

The covariance matrix is used for every iteration of the model-fit, being fully accounted for in 
the `log_likelihood_function`.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

By plotting the maximum log likelihood `FitImaging` object we can confirm the custom mask was used.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finish.
"""

"""
Manual Preprocessing: Mask Irregular
====================================
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
This tool allows one to mask a bespoke mask for a given image of a strong lens, which can then be loaded
before a model-fit.

This tool creates an irregular mask, which can form any shape and is not restricted to circles, annuli, ellipses,
etc. This mask is created as follows:

1) Blur the observed image with a Gaussian kernel of specified FWHM.
2) Compute the absolute S/N map of that blurred image and the noise-map.
3) Create the mask for all pixels with a S/N above a theshold value.

For strong lenses without a lens light component this masks create a source-only mask. If the lens light is included
it includes the lens light and source.

The following parameters determine the behaviour of this function:

The sigma value (e.g. FWHM) of the Gaussian the image is blurred with and the S/N threshold defining above which a 
image-pixel value must be to not be masked.
"""
blurring_gaussian_sigma = 0.1
snr_cut = 10.0

"""
Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/data_preparation/imaging/simple__no_lens_light`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.1
)
noise_map = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), pixel_scales=0.1
)

"""
Returns the 2D Gaussian that the image is blurred with. This blurring smooths over noise in the image, which will 
otherwise lead unmasked values with in individual pixels if not smoothed over correctly.
"""
blurring_gaussian = al.Kernel2D.from_gaussian(
    shape_native=(31, 31),
    pixel_scales=data.pixel_scales,
    sigma=blurring_gaussian_sigma,
)

"""
Blur the image with this Gaussian smoothing kernel and plot the resulting image.
"""
blurred_image = blurring_gaussian.convolved_array_from(array=data)
aplt.Array2DPlotter(array=blurred_image)

"""
Now compute the absolute signal-to-noise map of this blurred image, given the noise-map of the observed dataset.
"""
blurred_signal_to_noise_map = blurred_image / noise_map
aplt.Array2DPlotter(array=blurred_signal_to_noise_map)

"""
Now create the mask in 2ll pixels where the signal to noise is above some threshold value.
"""
mask = np.where(blurred_signal_to_noise_map.native > snr_cut, False, True)
mask = al.Mask2D(mask=mask, pixel_scales=data.pixel_scales)

visuals = aplt.Visuals2D(mask=mask)
array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
Now we`re happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""
mask.output_to_fits(file_path=path.join(dataset_path, "mask.fits"), overwrite=True)

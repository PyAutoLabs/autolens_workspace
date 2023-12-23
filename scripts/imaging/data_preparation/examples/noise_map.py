"""
Data Preparation: Noise-map
===========================

The noise-map defines the uncertainty in every pixel of your lens image, where values are defined as the
RMS standard deviation in every pixel (not the variances, HST WHT-map values, etc.).

You MUST be certain that the noise-map is the RMS standard deviations or else your analysis will be incorrect!

This tutorial describes preprocessing your dataset`s noise-map to adhere to the units and formats required
by **PyAutoLens**.

__Pixel Scale__

The "pixel_scale" of the image (and the data in general) is pixel-units to arcsecond-units conversion factor of
your telescope. You should look up now if you are unsure of the value.

The pixel scale of some common telescopes is as follows:

 - Hubble Space telescope 0.04" - 0.1" (depends on the instrument and wavelength).
 - James Webb Space telescope 0.06" - 0.1" (depends on the instrument and wavelength).
 - Euclid 0.1" (Optical VIS instrument) and 0.2" (NIR NISP instrument).
 - VRO / LSST 0.2" - 0.3" (depends on the instrument and wavelength).
 - Keck Adaptive Optics 0.01" - 0.03" (depends on the instrument and wavelength).

It is absolutely vital you use the correct pixel scale, so double check this value!

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

# %matplotlib inline
from os import path
import autolens as al
import autolens.plot as aplt

"""
__Loading Data From Individual Fits Files__

Load a noise-map from .fits files (a format commonly used by Astronomers) via the `Array2D` object. 

This noise-map represents a good data-reduction that conforms **PyAutoLens** formatting standards!
"""
dataset_path = path.join("dataset", "imaging", "simple")

noise_map = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), pixel_scales=0.1
)

array_plotter = aplt.Array2DPlotter(array=noise_map)
array_plotter.figure_2d()

"""
This noise-map conforms to **PyAutoLens** standards for the following reasons:

 - Units: Like its corresponding image, it is in units of electrons per second (as opposed to electrons, counts, 
   ADU`s etc.). Internal **PyAutoLens** functions for computing quantities like a galaxy magnitude assume the data and 
   model light profiles are in electrons per second.

 - Values: The noise-map values themselves are the RMS standard deviations of the noise in every pixel. When a model 
   is fitted to data in **PyAutoLens** and a likelihood is evaluated, this calculation assumes that this is the
   corresponding definition of the noise-map. The noise map therefore should not be the variance of the noise, or 
   another definition of noise.
   
 - Poisson: The noise-map includes the Poisson noise contribution of the image (e.g. due to Poisson count statistics
   in the lens and source galaxies), in addition to the contribution of background noise from the sky background. 
   Data reduction pipelines often remove the Poisson noise contribution, but this is incorrect and will lead to
   incorrect results.
   
Given the image should be centred and cut-out around the lens and source galaxies, so should the noise-map.

If your noise-map conforms to all of the above standards, you are good to use it for an analysis (but must also check
you image and PSF conform to standards first!).

If it does not conform to standards, this script illustrates **PyAutoLens** functionality which can be used to 
convert it to standards. 

__1) Tools Illustrated In Image__

The script `data_prepatation/examples/image.ipynb` illustrates the following preparation steps:

1) Converted it from counts / ADUs / other units to electrons per second.
2) Trimmed / padded the image.
3) Recentered the image.

You can perform identical operations on your noise-map (assuming it is in the same units and has the dimensions as the
image.

__Noise Conversions__

There are many different ways the noise-map can be reduced, and it varies depending on the telescope and its
specific data reduction. 

The preprocess module contains example functions for computing noise-maps, which may help you calculate your noise-map
from the data you currently have (if it is not already RMS values including the Poisson noise contribution and 
background sky contribution).

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/preprocess.py

Functions related to the noise map are:

- `noise_map_via_data_eps_and_exposure_time_map_from` 
- `noise_map_via_weight_map_from`
- `noise_map_via_inverse_noise_map_from`
- `noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from`
- `noise_map_via_data_eps_exposure_time_map_and_background_variances_from`
- `poisson_noise_via_data_eps_from
`
"""

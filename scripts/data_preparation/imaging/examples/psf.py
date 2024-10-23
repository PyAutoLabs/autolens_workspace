"""
Data Preparation: PSF
=====================

The Point Spread Function (PSF) describes blurring due the optics of your dataset`s telescope. It is used by
PyAutoLens when fitting a dataset to include these effects, such that does not bias the model.

It should be estimated from a stack of stars in the image during data reduction or using a PSF simulator (e.g. TinyTim
for Hubble).

This tutorial describes preprocessing your dataset`s psf to adhere to the units and formats required by PyAutoLens.

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

# %matplotlib
from os import path
import autolens as al
import autolens.plot as aplt

"""
Setup the path the datasets we'll use to illustrate preprocessing, which is the folder `dataset/data_preparation/imaging`.
"""
dataset_path = path.join("dataset", "imaging", "simple")

"""
__Loading Data From Individual Fits Files__

Load a PSF from .fits files (a format commonly used by Astronomers) via the `Array2D` object. 

This image represents a good data-reduction that conforms **PyAutoLens** formatting standards!
"""
psf = al.Kernel2D.from_fits(
    file_path=path.join(dataset_path, "psf.fits"), hdu=0, pixel_scales=0.1
)

array_plotter = aplt.Array2DPlotter(array=psf)
array_plotter.figure_2d()

"""
This psf conforms to **PyAutoLens** standards for the following reasons.

 - Size: The PSF has a shape 21 x 21 pixels, which is large enough to capture the PSF core and thus capture the 
   majority of the blurring effect, but not so large that the convolution slows down the analysis. Large 
   PSFs (e.g. 51 x 51) are supported, but will lead to much slower run times. The size of the PSF should be carefully 
   chosen to ensure it captures the majority of blurring due to the telescope optics, which for most instruments is 
   something around 11 x 11 to 21 x 21.

 - Oddness: The PSF has dimensions which are odd (an even PSF would for example have shape 20 x 20). The 
   convolution of an even PSF introduces a small shift in the modle images and produces an offset in the inferred
   model parameters.
   
 - Normalization: The PSF has been normalized such that all values within the kernel sum to 1 (note how all values in 
   the example PSF are below zero with the majority below 0.01). This ensures that flux is conserved when convolution 
   is performed, ensuring that quantities like a galaxy's magnitude are computed accurately.

 - Centering: The PSF is at the centre of the array (as opposed to in a corner), ensuring that no shift is introduced
   due to PSF blurring on the inferred model parameters.

If your PSF conforms to all of the above standards, you are good to use it for an analysis (but must also check
you noise-map and image conform to standards first!).

If it does not conform to standards, this script illustrates **PyAutoLens** functionality which can be used to 
convert it to standards. 

__1) PSF Size__

The majority of PSF blurring occurs at its central core, which is the most important region for lens modeling. 

By default, the size of the PSF kernel in the .fits is used to perform convolution. The larger this stamp, the longer 
this convolution will take to run. Large PSFs (e.g. > 51 x 51) could have significantly slow down on run-time. 

In general we recommend the PSF size is 21 x 21. The example below is 11 x 11, which for this simulated data is just 
about acceptable but would be on the small side for many real telescopes.
"""
psf = al.Kernel2D.from_fits(
    file_path=path.join(dataset_path, "psf.fits"), hdu=0, pixel_scales=0.1
)

array_plotter = aplt.Array2DPlotter(array=psf)
array_plotter.figure_2d()

"""
We can resize a psf the same way that we resize an image.

Below, we resize the PSF to 5 x 5 pixels, which is too small for a realistic analysis and just for demonstration 
purposes.
"""
trimmed_psf = al.preprocess.array_with_new_shape(array=psf, new_shape=(5, 5))

array_plotter = aplt.Array2DPlotter(array=trimmed_psf)
array_plotter.figure_2d()

"""
__PSF Dimensions__

The PSF dimensions must be odd x odd (e.g. 21 x 21), because even-sized PSF kernels introduce a half-pixel offset in 
the convolution routine which can lead to systematics in the lens analysis. 

The preprocess module contains functions for converting an even-sized PSF to an odd-sized PSF.

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/preprocess.py

- `psf_with_odd_dimensions_from`

However, this uses an interpolation routine that will not be perfect. The best way to create an odd-sized PSF is to do 
so via the data reduction procedure. If this is a possibility, do that, this function is only for when you have no
other choice.

__PSF Normalization__

The PSF should also be normalized to unity. That is, the sum of all values in the kernel 
should sum  to 1. This ensures that the PSF convolution does not change the overall normalization of the image.

PyAutoLens automatically normalized PSF when they are passed into a `Imaging` or `SimulatedImaging` object, so you 
do not actually need to normalize your PSF. However, it is better to do it now, just in case.

Below, we show how to normalize a PSF when it is loaded from a .fits file, by simply including the `normalize=True`
argument (the default value is `False`).
"""
psf = al.Kernel2D.from_fits(
    file_path=path.join(dataset_path, "psf.fits"),
    hdu=0,
    pixel_scales=0.1,
    normalize=True,
)

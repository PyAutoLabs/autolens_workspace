"""
Data Preparation: Image
=======================

The image is the image of your galaxy, which comes from a telescope like the Hubble Space telescope (HST).

This tutorial describes preprocessing your dataset`s image to adhere to the units and formats required by PyAutoLens.

__Contents__

**Pixel Scale:** The "pixel_scale" of the image (and the data in general) is pixel-units to arcsecond-units.
**Loading Data From Individual Fits Files:** Load an image from .fits files (a format commonly used by Astronomers) via the `Array2D` object.
**Converting Data To Electrons Per Second:** Brightness units: the image`s flux values should be in units of electrons per second (as opposed to.
**Resizing Data:** The bigger the postage stamp cut-out of the image the more memory it requires to store.
**Background Subtraction:** The background of an image is the light that is not associated with the lens or source galaxies we.

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

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from autoconf import jax_wrapper  # Sets JAX environment before other imports


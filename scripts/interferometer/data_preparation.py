"""
Interferometer: Data Preparation
================================

When an interferometer dataset is analysed, it must conform to certain standards in order for
the analysis to be performed correctly. This tutorial describes these standards and links to more detailed scripts
which will help you prepare your dataset to adhere to them if it does not already.

__Contents__

**SLACK:** The interferometer data preparation scripts are currently being developed and are not yet complete.
**Pixel Scale:** When fitting an interferometer dataset, the images of the lens and source galaxies are first.
**Visibilities:** The image is the image of your strong lens, which comes from a telescope like the Hubble Space.
**UV Wavelengths:** The uv-wavelengths define the baselines of the interferometer.
**Real Space Mask:** The `modeling` scripts also define a real-space mask, which defines where the image is evalated in.
**Data Processing Complete:** If your visibilities, noise-map, uv_wavelengths and real space mask conform the standards above.

__SLACK__

The interferometer data preparation scripts are currently being developed and are not yet complete. If you are
unsure of how to prepare your dataset, please message us on Slack and we will help you directly!

__Pixel Scale__

When fitting an interferometer dataset, the images of the lens and source galaxies are first evaluated in real-space
using a grid of pixels, which is then Fourier transformed to the uv-plane.

The "pixel_scale" of an interferometer dataset is this pixel-units to arcsecond-units conversion factor. The value
depends on the instrument used to observe the lens, the wavelength of the light used to observe it and size of
the baselines used (e.g. longer baselines means higher resolution and therefore a smaller pixel scale).

The pixel scale of some common interferometers is as follows:

 - ALMA: 0.02" - 0.1" / pixel
 - SMA: 0.05" - 0.1" / pixel
 - JVLA: 0.005" - 0.01" / pixel

It is absolutely vital you use a sufficently small pixel scale that all structure in the data is resolved after the
Fourier transform. If the pixel scale is too large, the Fourier transform will smear out the data and the lens model.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from autoconf import jax_wrapper  # Sets JAX environment before other imports


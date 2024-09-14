"""
Flux
====

Absolute flux calibration in Astronomy is the process of converting the number of photons detected by a telescope into
a physical unit of luminosity or a magnitude. For example, a luminosity might be given in units of solar luminosities
or the brightness of a galaxy quoted as a magnitude in units of AB magnitudes.

The conversion of a light profile, that has been fitted to data, to physical units can be non-trivial, as careful
consideration must be given to the units that are involved.

The key quantity is the `intensity` of the light profile, the units of which match the units of the data that is fitted.
For example, if the data is in units of electrons per second, the intensity will also be in units of electrons per
second per pixel.

The conversion of this intensity to a physical unit, like solar luminosities, therefore requires us to make a number
of conversion steps that go from electrons per second to the desired physical unit or magnitude.

This guide gives example conversions for units commonly used in astronomy, such as converting the intensity of a
light profile from electrons per second to solar luminosities or AB magnitudes. Once we have values in a more standard
unit, like a solar luminosity or AB magnitude, it becomes a lot more straightforward to follow Astropy tutorials
(or other resources) to convert these values to other units or perform calculations with them.

__Zero Point__

In astronomy, a zero point refers to a reference value used in photometry and spectroscopy to calibrate the brightness
of celestial objects. It sets the baseline for a magnitude system, allowing astronomers to compare the brightness of
different stars, galaxies, or other objects.

For example, the zero point in a photometric system corresponds to the magnitude that a standard star (or a theoretical
object) would have if it produced a specific amount of light at a particular wavelength. It provides a way to convert
the raw measurements of light received by a telescope into meaningful values of brightness (magnitudes).

The conversions below all require a zero point, which is typically provided in the documentation of the telescope or
instrument that was used to observe the data.
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
__Mega Janskys / steradian (MJy/sr): James Webb Space Telescope__

James Webb Space Telescope (JWST) NIRCam data is often provided in units of Mega Janskys per steradian (MJy/sr).
We therefore show how to convert the intensity of a light profile to MJy/sr.

This calculation is well documented in the JWST documentation, and we are following the steps in the following
webpage:

https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints#gsc.tab=0

First, we need a light profile, which we'll assume is a Sersic profilee. If you're analyzing real JWST data, you'll
need to use the light profile that was fitted to the data.
"""
light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=2.0,  # in units of MJy/sr
    effective_radius=0.1,
    sersic_index=3.0,
)

"""
According to the document above, flux density in MJy/sr can be converted to AB magnitude using the following formula:

 mag_AB = -6.10 - 2.5 * log10(flux[MJy/sr]*PIXAR_SR[sr/pix] ) = ZP_AB – 2.5 log10(flux[MJy/sr])

Where ZP_AB is the zeropoint:  

 ZP_AB = –6.10 – 2.5 log10(PIXAR_SR[sr/pix]). 

For example, ZP_AB = 28.0 for PIXAR_SR = 2.29e-14 (corresponding to pixel size 0.0312").

For data in units of MJy/sr, computing the total flux that goes into the log10 term is straightforward, it is
simply the sum of the image of the light profile. 

We compute this using a grid, which must be large enough that all light from the light profile is included. Below,
we use a grid which extends to 10" from the centre of the light profile, which is sufficient for this example,
but you may need to increase this size for your own data.
"""
grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = light.image_2d_from(grid=grid)

total_flux = np.sum(image)

"""
We now convert this total flux to an AB magnitude using the zero point of the JWST NIRCam filter we are analyzing.

As stated above, the zero point is given by:

 ZP_AB = –6.10 – 2.5 log10(PIXAR_SR[sr/pix])
 
Where the value of PIXAR_SR is provided in the JWST documentation for the filter you are analyzing. 

The Pixar_SR values for JWST (James Webb Space Telescope) NIRCam filters refer to the pixel scale in steradians (sr) 
for each filter, which is a measure of the solid angle covered by each pixel. These values are important for 
calibrating and understanding how light is captured by the instrument.

For the F444W filter, which we are using in this example, the value is 2.29e-14 (corresponding to a pixel size o
f 0.0312").
"""
pixar_sr = 2.29e-14

zero_point = -6.10 - 2.5 * np.log10(pixar_sr)

magnitude_ab = zero_point - 2.5 * np.log10(total_flux)

"""
Finish.
"""

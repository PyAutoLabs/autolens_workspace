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

__Contents__

- **Zero Point:** In astronomy, a zero point refers to a reference value used in photometry and spectroscopy to.
- **Total Flux:** A key quantity for computing the magnitudes of galaxies is the total flux of a light profile.
- **Latent Variables:** Reading the same total flux directly from the `latent.csv` of a completed fit.

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
# ENV: full_datasets
# Guides load committed full-resolution FITS; SMALL_DATASETS would
# mismatch the pre-existing 100x100 data shape.

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
from scipy.special import gamma

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Total Flux__

A key quantity for computing the magnitudes of galaxies is the total flux of a light profile.

The most simple way to compute the total flux of a light profile is to create a grid of (y,x) coordinates over which
we compute the image of the light profile, and then sum the image. 

The units of the light profile `intensity` are the units of the data the light profile was fitted to. For example, 
HST data is often electrons per second, so the intensity is in units of electrons per second per pixel.
"""
light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=2.0,  # in units of e- pix^-1 s^-1, representative of HST data in units of electrons per second
    effective_radius=0.1,
    sersic_index=3.0,
)

"""
The total flux, in units of electrons per second, is computed by summing the image of the light profile over all pixels.

Note that we can use a `grid` of any shape and pixel scale here, the important thing is that it is so large
and high enough resolution that it captures all the light from the light profile.
"""
grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = light.image_2d_from(grid=grid)

total_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

"""
For a spherical Sersic function, there is an analytic expression for the total flux, shown below.

However, because the light profile is in units of pix^-1, the total flux computed via this expression is in slightly
strange units we need to account for afterwards of e- s^-1 arcsec^2 pix^-1.
"""
total_flux_strange_units = (
    light.intensity
    * (light.effective_radius**2)
    * 2
    * np.pi
    * light.sersic_index
    * (
        np.exp(light.sersic_constant)
        / (light.sersic_constant ** (2 * light.sersic_index))
    )
    * gamma(2 * light.sersic_index)
)

"""
To get the total flux in units of e- s^-1, we divide by the total grid area (in arcsec^2) and multiply by the total
number of pixels, which are provided by the grid.
"""
total_flux = (total_flux_strange_units / grid.total_area) * grid.total_pixels

"""
The two calculations come out very close to one another, and become closer the more pixels we use in the grid to 
compute the total flux.

If possible, you should use analytic expressions to compute the total flux of a light profile, as this is exact, 
especially if computing magnitudes precisely is important for your science case.

However, for many light profiles the total flux cannot easily be computed analytically, and the summed image approach
sufficient.

__Mega Janskys / steradian (MJy/sr): James Webb Space Telescope__

James Webb Space Telescope (JWST) NIRCam data is often provided in units of Mega Janskys per steradian (MJy/sr).
We therefore show how to convert the intensity of a light profile from MJy/sr to absolute AB magnitudes.

This calculation is well documented in the JWST documentation, and we are following the steps in the following
webpage:

https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints#gsc.tab=0

First, we need a light profile, which we'll assume is a Sersic profilee. If you're analyzing real JWST data, you'll
need to use the light profile that was fitted to the data.
"""
light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=2.0,  # in units of MJy sr^-1 pix^-1
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

total_flux = np.sum(image)  # In units of MJy sr^-1 as summed over pixels

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
With an absolute magnitude and quantity of light in physical units, you should now be able to convert these values to
whatever units you need for your science case.

__Electrons Per Second (e s^-1): Hubble Space Telescope__

Hubble Space Telescope (HST) data is often provided in units of electrons per second (e- s^-1). 

We therefore show how to convert the intensity of a light profile from electrons per second to an absolute magnitude.
"""
light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=2.0,  # in units of e- pix^-1 s^-1
    effective_radius=0.1,
    sersic_index=3.0,
)

"""
We first compute the total flux in electrons per second by summing the image of the light profile.
"""
grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = light.image_2d_from(grid=grid)  # in units e- s^-1 as summed over pixels

total_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

"""
We now use the zero point of the HST filter we are analyzing to convert this total flux to an AB magnitude.

The zero point for the F814W filter, which we are using in this example, is 25.943.

Zero points of the HST ACS filter are provided here: https://acszeropoints.stsci.edu, for other filters you should
consult the HST documentation.

The zero point below is defined in units such that it converts the total flux from input units of electrons per second,
you should make sure your HST data is in these units and that the zero point you are using follows the same convention.
"""
zero_point_f814w = 25.943

magnitude_ab = zero_point_f814w - 2.5 * np.log10(total_flux)

"""
With an absolute magnitude and quantity of light in physical units, you should now be able to convert these values to
whatever units you need for your science case.

For HST, a few quantitites that may be useful and worth looking into are:

- The HST PHOTFLAM value, in units of erg cm^-2 s^-1 A^-1 e-^-1, which is used to convert to ergs, which radio 
  astronomers may be interested in.
  
- The HST PHOTNU value, in units of Jy (e s^-1), which converts to Janskys, which is often used by SED fitting
  software.

__Latent Variables: Total Flux Directly from the Fit__

The examples above all computed total flux by hand: build a light profile, sample it on a grid, sum the image, then
apply the zero point. PyAutoLens does exactly this automatically as part of every fit and records the result as a
latent variable in the `latent/samples.csv` file beside the search output. You can skip the manual recipe entirely
and just read the column.

Three lensing flux latents ship default-on (they need no instrument inputs and run on every fit unless disabled in
`config/latent.yaml`):

- `total_lens_flux` — total integrated flux of the lens galaxy (the sum of `fit.tracer.galaxies[0]`'s model image),
  in the *raw* image units the fit was performed in. For HST data in e- s^-1, this is e- s^-1; for JWST data in
  MJy/sr, this is MJy/sr.

- `total_lensed_source_flux` — image-plane integrated flux of the source galaxy after lensing. This is the source
  flux as observed: stretched, multiplied, and distorted by the lens. Same raw units.

- `total_source_flux` — source-plane intrinsic flux of the source galaxy: the flux the source would have if the
  lens weren't there. Computed via `fit.tracer_linear_light_profiles_to_light_profiles` so that linear light
  profiles (MGEs, pixelizations) work correctly. The ratio
  `total_lensed_source_flux / total_source_flux` is the integrated magnification (also recorded directly as the
  `magnification` latent).

To convert any of these to AB magnitudes or microjanskies, apply the same zero-point recipe used above. Suppose
you have an HST F814W fit with the F814W zero point of 25.943; reading the lens galaxy flux from your result and
converting goes:
"""
from autogalaxy.imaging.model.latent import (
    ab_mag_via_flux_from,
    flux_mujy_via_ab_mag_from,
)

# Stand-in for what you'd read from `latent.csv` — in a real script this is one column of one row, e.g.
#   total_lens_flux = pd.read_csv(search.paths.output_path / "latent" / "samples.csv")["total_lens_flux"].iloc[-1]
total_lens_flux = 1234.5  # e- s^-1

zero_point_f814w = 25.943
ab_mag_lens = ab_mag_via_flux_from(flux=total_lens_flux, magzero=zero_point_f814w)
flux_mujy_lens = flux_mujy_via_ab_mag_from(ab_mag=ab_mag_lens)

"""
The two helpers used above are the same ones the library uses internally to populate the `_mujy` variants of the
latents (`total_lens_flux_mujy`, `total_lensed_source_flux_mujy`, `total_source_flux_mujy`). Those variants are
default-off because they need a `magzero` you supply per-instrument. If you have a single fixed zero-point you can
flip them on by:

1. Setting `total_lens_flux_mujy: true` (and friends) in your project's `config/latent.yaml`.
2. Passing `magzero=<value>` when constructing the analysis:
   `analysis = al.AnalysisImaging(dataset=dataset, magzero=25.943)`.

The latent dispatcher then writes the converted µJy columns into `latent/samples.csv` directly, so you don't have
to run the conversion in post. If you enable the `_mujy` latents but forget the `magzero` keyword, the columns are
populated with NaN and a single warning per process notes that the conversion was skipped — the fit itself is
unaffected.

Finish.
"""

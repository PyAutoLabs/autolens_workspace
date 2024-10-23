"""
Imaging: Data Preparation
=========================

When a CCD imaging dataset is analysed, it must conform to certain standards in order for the analysis
to be performed correctly. This tutorial describes these standards and links to more detailed scripts which will help
you prepare your dataset to adhere to them if it does not already.

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
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Image__

The image is the image of your strong lens, which comes from a telescope like the Hubble Space telescope (HST).

Lets inspect an image which conforms to **PyAutoLens** standards:
"""
dataset_path = path.join("dataset", "imaging", "simple")

data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.1
)

array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

"""
This image conforms to **PyAutoLens** standards for the following reasons.

 - Units: The image flux is in units of electrons per second (as opposed to electrons, counts, ADU`s etc.). 
   Internal **PyAutoLens** functions for computing quantities like galaxy magnitudes assume the data and model
   light profiles are in electrons per second.
   
 - Centering: The lens galaxy is at the centre of the image (as opposed to in a corner). Default **PyAutoLens**
   parameter priors assume the lens galaxy is at the centre of the image.
   
 - Stamp Size: The image is a postage stamp cut-out of the lens, but does not include many pixels around the edge of
   the lens. It is advised to cut out a postage stamp of the lens, as opposed to the entire image, as this reduces
   the amount of memory **PyAutoLens** uses, speeds up the analysis and ensures visualization zooms around the lens
   and source. However, conforming to this standard is not necessary to ensure an accurate **PyAutoLens** analysis.
   
If your image conforms to all of the above standards, you are good to use it for an analysis (but must also check
you noise-map and PSF conform to standards first!).

**Links / Resources:**

 - `data_preparation/imaging/examples/data.ipynb`: tools to process the data to conform to these standards.

__Noise Map__

The noise-map defines the uncertainty in every pixel of your strong lens image, where values are defined as the 
RMS standard deviation in every pixel (not the variances, HST WHT-map values, etc.). 

Lets inspect a noise-map which conforms to **PyAutoLens** standards:
"""
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

If you are not certain what the definition of the noise-map you have available to you is, or do not know how to
compute a noise-map at all, you should refer to the instrument handbook of the telescope your data is from. It is
absolutely vital that the noise-map is correct, as it is the only way **PyAutoLens** can quantify the goodness-of-fit.

A sanity check for a reliable noise map is that the signal-to-noise of the lens galaxy is somewhere between a value of 
10 - 300 and source around 5 - 50, however this is not a definitive test.
   
Given the image should be centred and cut-out around the lens galaxy, so should the noise-map.

If your noise-map conforms to all of the above standards, you are good to use it for an analysis (but must also check
you image and PSF conform to standards first!).

**Links / Resources:**

 - `data_preparation/imaging/examples/noise_map.ipynb`: tools to process the noise-map to conform to these standards.

__PSF__

The Point Spread Function (PSF) describes blurring due the optics of your dataset`s telescope. It is used when fitting 
a dataset to include these effects, such that does not bias the model.

It should be estimated from a stack of stars in the image during data reduction or using a PSF simulator (e.g. TinyTim
for Hubble).

Lets inspect a PSF which conforms to **PyAutoLens** standards:
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
   convolution of an even PSF introduces a small shift in the model images and produces an offset in the inferred
   lens model parameters. Inputting an even PSF will lead **PyAutoLens** to raise an error.

 - Normalization: The PSF has been normalized such that all values within the kernel sum to 1 (note how all values in 
   the example PSF are below zero with the majority below 0.01). This ensures that flux is conserved when convolution 
   is performed, ensuring that quantities like a galaxy's magnitude are computed accurately.

 - Centering: The PSF is at the centre of the array (as opposed to in a corner), ensuring that no shift is introduced
   due to PSF blurring on the inferred model parameters.

If your PSF conforms to all of the above standards, you are good to use it for an analysis (but must also check
you noise-map and image conform to standards first!).

**Links / Resources:**

 - `data_preparation/imaging/examples/psf.ipynb`: tools to process the PSF to conform to these standards.

__Data Processing Complete__

If your image, noise-map and PSF conform the standards above, you are ready to analyse your dataset!

Below, we provide an overview of optional data preparation steos which prepare other aspects of the analysis. 

New users are recommended to skim-read the optional steps below so they are aware of them, but to not perform them 
and instead analyse their dataset now. You can come back to the data preparation scripts below if it becomes necessary.

__Mask (Optional)__

The mask removes the regions of the image where the lens and source galaxy are not present, typically the edges of the 
image.

Example modeling scripts internally create a 3.0" circular mask and therefore do not require that a mask has been 
created externally via a data preparation script. 

This script shows how to create customize masked (e.g. annular, ellipses) which are tailored to match the lens or
lensed source emission. 

If you have not analysed your dataset yet and do not know of a specific reason why you need the bespoke masks 
created by this script, it is recommended that you simply use the default ~3.0" circular mask internally made in each
script and omit this data preparation tutorial.

**Links / Resources:**

- `data_preparation/examples/optional/mask.ipynb`: tools to create a bespoke mask for your dataset.
- `data_preparation/examples/gui/mask.ipynb`: use a Graphical User Interface (GUI) to create a bespoke mask.

__Positions (Optional)__

The script allows you to mark the (y,x) arc-second positions of the multiply imaged lensed source galaxy in 
the image-plane, under the assumption that they originate from the same location in the source-plane.

A non-linear search (e.g. Nautilus) can then use these positions to preferentially choose mass models where these 
positions trace close to one another in the source-plane. This speeding up the initial fitting of lens models and 
removes unwanted solutions from parameter space which have too much or too little mass in the lens galaxy.

If you create positions for your dataset, you must also update your modeling script to use them by loading them 
and passing them to the `Analysis` object via a `PositionsLH` object. 

If your **PyAutoLens** analysis is struggling to converge to a good lens model, you should consider using positions
to help the non-linear search find a good lens model.

**Links / Resources:**

Position-based lens model resampling is particularly important for fitting pixelized source models, for the
reasons disucssed in the following readthedocs 
webapge  https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

- `data_preparation/examples/optional/positions.ipynb`: input the positions manually into a Python script.
- `data_preparation/gui/positions.ipynb` use a Graphical User Interface (GUI) to mark the positions.
- `modeling/imaging/customize/positions.py` for an example.of how to use positions in a `modeling` script.


__Lens Light Centre (Optional)__

This script allows you to mark the (y,x) arcsecond locations of the lens galaxy light centre(s) of the strong lens
you are analysing. These can be used as fixed values for the lens light and mass models in a model-fit.

This  reduces the number of free parameters fitted for in a lens model and removes inaccurate solutions where
the lens mass model centre is unrealistically far from its true centre.

Advanced `chaining` scripts often use these input centres in the early fits to infer an accurate initial lens model,
amd then make the centres free parameters in later searches to ensure a general and accurate lens model is inferred.

If you create a `light_centre` for your dataset, you must also update your modeling script to use them.

If your **PyAutoLens** analysis is struggling to converge to a good lens model, you should consider using a fixed
lens light and / or mass centre to help the non-linear search find a good lens model.

**Links / Resources:**

- `data_preparation/examples/optional/lens_light_centre.py`: input the lens galaxy light centre manually into a Python script.
- `data_preparation/gui/lens_light_centre.ipynb` use a Graphical User Interface (GUI) to mask the lens galaxy light centre.


__Extra Galaxies (Optional)__

There may be galaxies nearby the lens and source galaxies, whose emission blends with that of the lens and source
and whose mass may contribute to the ray-tracing and lens model.

We can include these galaxies in the lens model, either as light profiles, mass profiles, or both, using the
modeling API, where these nearby objects are denoted `extra_galaxies`.

The script `extra_galaxies_centres.py` marks the (y,x) arcsecond locations of these extra galaxies, so that when they 
are included in the lens model the centre of these extra galaxies light and / or mass profiles are fixed to these 
values (or their priors are  initialized surrounding these centres).

The example `mask_extra_galaxies.py` (see below) masks the regions of an image where extra galaxies are present. 
This mask is used  to remove their signal from the data and increase their noise to make them not impact the fit. 
This means their  luminous emission does not need to be included in the model, reducing the number of free parameters 
and speeding up the analysis. It is still a choice whether their mass is included in the model.

**Links / Resources:**

- `data_preparation/examples/optional/extra_galaxies_centres.py`: input the extra galaxy centres manually into a 
  Python script.
- `data_preparation/gui/extra_galaxies_centres.ipynb`: use a Graphical User Interface (GUI) to mark the extra galaxy centres.
- `features/extra_galaxies.py` how to use extra galaxies in a model-fit, including loading the extra galaxy centres.


__Mask Extra Galaxies (Optional)__

There may be regions of an image that have signal near the lens and source that is from other galaxies not associated 
with the strong lens we are studying. The emission from these images will impact our model fitting and needs to be 
removed from the analysis.

This script creates a mask of these regions of the image, called the `mask_extra_galaxies`, which can be used to
prevent them from impacting a fit. This mask may also include emission from objects which are not technically galaxies,
but blend with the galaxy we are studying in a similar way. Common examples of such objects are foreground stars
or emission due to the data reduction process.

The mask can be applied in different ways. For example, it could be applied such that the image pixels are discarded
from the fit entirely, Alternatively the mask could be used to set the image values to (near) zero and increase their
corresponding noise-map to large values.

The exact method used depends on the nature of the model being fitted. For simple fits like a light profile a mask
is appropriate, as removing image pixels does not change how the model is fitted. However, for more complex models
fits, like those using a pixelization, masking regions of the image in a way that removes their image pixels entirely
from the fit can produce discontinuities in the pixelixation. In this case, scaling the data and noise-map values
may be a better approach.

**Links / Resources:**

- `data_preparation/examples/optional/mask_extra_galaxies.py`: create the extra galaxies mask manually via a Python script.
- `data_preparation/gui/extra_galaxies_mask.ipynb` use a Graphical User Interface (GUI) to create the extra galaxies mask.
- `features/extra_galaxies.py` how to use the extra galaxies mask in a model-fit.

__Info (Optional)__

Auxiliary information about a strong lens dataset may used during an analysis or afterwards when interpreting the 
 modeling results. For example, the redshifts of the source and lens galaxy. 

By storing these as an `info.json` file in the lens's dataset folder, it is straight forward to load the redshifts 
in a modeling script and pass them to a fit, such that **PyAutoLens** can then output results in physical 
units (e.g. kpc instead of arc-seconds).

For analysing large quantities of  modeling results, **PyAutoLens** has an sqlite database feature. The info file 
may can also be loaded by the database after a model-fit has completed, such that when one is interpreting
the results of a model fit additional data on a lens can be used to. 

For example, to plot the model-results against other measurements of a lens not made by PyAutoLens. Examples of such 
data might be:

- The velocity dispersion of the lens galaxy.
- The stellar mass of the lens galaxy.
- The results of previous strong lens models to the lens performed in previous papers.

**Links / Resources:**

- `data_preparation/examples/optional/info.py`: create the info file manually via a Python script.
"""

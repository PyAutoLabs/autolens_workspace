"""
Data Preparation: Image
=======================

The image is the image of your galaxy, which comes from a telescope like the Hubble Space telescope (HST).

This tutorial describes preprocessing your dataset`s image to adhere to the units and formats required by PyAutoLens.

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

Load an image from .fits files (a format commonly used by Astronomers) via the `Array2D` object. 

This image represents a good data-reduction that conforms **PyAutoLens** formatting standards!
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
   Internal **PyAutoLens** functions for computing quantities like a galaxy magnitude assume the data and model
   light profiles are in electrons per second.
   
 - Centering: The lens galaxy is at the centre of the image (as opposed to in a corner). Default **PyAutoLens**
   parameter priors assume the galaxy is at the centre of the image.
   
 - Stamp Size: The image is a postage stamp cut-out of the galaxy, but does not include many pixels around the edge of
   the galaxy. It is advisible to cut out a postage stamp of the galaxy, as opposed to the entire image, as this reduces
   the amount of memory **PyAutoLens** uses, speeds up the analysis and ensures visualization zooms around the galaxy. 
   Conforming to this standard is not necessary to ensure an accurate analsyis.
    
  - Background Sky Subtraction: The image has had its background sky subtracted. 
   
If your image conforms to all of the above standards, you are good to use it for an analysis (but must also check
you noise-map and PSF conform to standards first!).

If it does not conform to standards, this script illustrates **PyAutoLens** functionality which can be used to 
convert it to standards. 

__Converting Data To Electrons Per Second__

Brightness units: the image`s flux values should be in units of electrons per second (as opposed to electrons, 
counts, ADU`s etc.). 

Although **PyAutoLens** can technically perform an analysis using other units, the default setup assumes electrons per 
second (e.g. the priors on `LightProfile` intensity and `Regularization` parameters). Thus, images not in electrons per 
second should be converted!

The data loaded above is in units of electrons per second, lets convert it to counts to illustrate how this is done.

Converting from electrons per second to counts (and visa versa) means we must know the exposure time of our observation, 
which will either be in the .fits header information of your data or be an output of your data reduction pipeline.

We create an `Array2D` of the exposure time map, which is the exposure time of each pixel in the image assuming that
all pixels have the same exposure time. This is a good approximation for most HST observations, but not for all.
"""
exposure_time = 1000.0

exposure_time_map = al.Array2D.full(
    fill_value=exposure_time,
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
)

data_counts = al.preprocess.array_eps_to_counts(
    array_eps=data, exposure_time_map=exposure_time_map
)

"""
By plotting the image in counts, we can see that the flux values are now much higher values (e.g. ~1000 or above)
compared to the data in electrons per second (e.g. ~1 or below).
"""
array_plotter = aplt.Array2DPlotter(array=data_counts)
array_plotter.figure_2d()

"""
It is therefore straightforward to convert an image to electrons per second from counts.
"""
data_eps = al.preprocess.array_counts_to_eps(
    array_counts=data_counts, exposure_time_map=exposure_time_map
)

array_plotter = aplt.Array2DPlotter(array=data_eps)
array_plotter.figure_2d()

"""
If the effective exposure-time map is output as part of the data reduction, you can use this to convert the image to 
electrons per second instead.

[The code below is commented out because the simulated data does not have an effective exposure time map in .fits 
format.]
"""
# exposure_time_map = al.Array2D.from_fits(
#     file_path=path.join(dataset_path, "exposure_time_map.fits"),
#     pixel_scales=data_eps.pixel_scales,
# )
#
# data_eps = al.preprocess.array_counts_to_eps(
#     array_counts=data_counts, exposure_time_map=exposure_time_map
# )
#
# array_plotter = aplt.Array2DPlotter(array=data_eps)
# array_plotter.figure_2d()

"""
**PyAutoLens** can also convert data to / from units of electrons per second to ADUs, which uses both the exposure 
time andinstrumental gain of the data.
"""
data_in_adus = al.preprocess.array_eps_to_adus(
    array_eps=data, gain=4.0, exposure_time_map=exposure_time_map
)

array_plotter = aplt.Array2DPlotter(array=data_in_adus)
array_plotter.figure_2d()

data_eps = al.preprocess.array_adus_to_eps(
    array_adus=data_in_adus, gain=4.0, exposure_time_map=exposure_time_map
)

array_plotter = aplt.Array2DPlotter(array=data_eps)
array_plotter.figure_2d()

"""
In `autolens_workspace/*/data_preparation/noise_map.py` we show that a noise-map must also be in units of 
electrons per second, and that the same functions as above can be used to do this.

__Resizing Data__

The bigger the postage stamp cut-out of the image the more memory it requires to store. Visualization will be less 
ideal too, as the lens galaxy will be a smaller blob in the centre relative to the large surrounding edges of the image. Why 
keep the edges surrounding the lens and sourcegalaxy if they are masked out anyway?

Lets look at an example of a very large postage stamp - we can barely even see the lens and source galaxy!
"""
dataset_path = path.join("dataset", "imaging", "simple__big_stamp")

data_large_stamp = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.1
)

array_plotter = aplt.Array2DPlotter(array=data_large_stamp)
array_plotter.figure_2d()

"""
If you have a large postage stamp you can trim it using the preprocess module, which is centered on the image.
"""
data_large_stamp_trimmed = al.preprocess.array_with_new_shape(
    array=data_large_stamp, new_shape=(130, 130)
)

array_plotter = aplt.Array2DPlotter(array=data_large_stamp_trimmed)
array_plotter.figure_2d()

"""
Stamps can also be too small, if the mask you input to the analysis is larger than the postage stamp extends.

In this case, you either need to reproduce the data with a larger postage stamp, or use a smaller mask.

__Background Subtraction__

The background of an image is the light that is not associated with the lens or source galaxies we are 
interested in. This is due to light from the sky, zodiacal light, and light from other galaxies in the 
field of view. The background should have been subtracted from the image before it was reduced, but 
sometimes this is not the case.

It is recommend you use data processing tools outside of **PyAutoLens** to subtract the background from your image,
as these have been optimized for this task. However, if you do not have access to these tools, **PyAutoLens** has
functions in the `preprocess` module that can estimate and subtract the background of an image.

The preprocess module is found here: 

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/preprocess.py

Functions related to background subtraction are:

- `background_sky_level_via_edges_from`
- `background_noise_map_via_edges_from`
"""

# __Centering__

########## IVE INCLUDED THE TEXT CAN BE AWARE OF CENTERING, BUT THE BUILT IN FUNCTIONALITY FOR #####
########## RECENTERING CURRENTLY DOES NOT WORK :( ###########

# galaxy Centering - The galaxy should be in the centre of the image as opposed to a corner. This ensures
# the origin of the galaxy's light and `MassProfile`'s are near the origin (0.0", 0.0") of the grid used to perform
# ray-tracing. The defaults priors on light and `MassProfile`'s assume a origin of (0.0", 0.0").

# Lets look at an off-center image - clearly both the galaxy and Einstein ring are offset in the positive y and x d
# directions.

# dataset_path = f"{dataset_path}/imaging_offset_centre"

# imaging_offset_centre = al.Imaging.from_fits(data_path=path+`image.fits`, pixel_scales=0.1,
#                                   noise_map_path=path+`noise_map.fits`,
#                                   psf_path=path+`psf.fits`)
# aplt.Imaging.subplot(imaging=imaging_offset_centre)

# We can address this by using supplying a new centre for the image, in pixels. We also supply the resized shape, to
# instruct the code whether it should trim the image or pad the edges that now arise due to recentering.

# imaging_recentred_pixels = al.Imaging.from_fits(data_path=path+`image.fits`, pixel_scales=0.1,
#                                             noise_map_path=path+`noise_map.fits`,
#                                             psf_path=path+`psf.fits`,
#                                             resized_imaging_shape=(100, 100),
#                                             resized_imaging_centre_pixels=(0, 0))
# #                                            resized_imaging_centre_arc_seconds=(1.0, 1.0))
# print(imaging_recentred_pixels.shape)
# aplt.Imaging.subplot(imaging=imaging_recentred_pixels)

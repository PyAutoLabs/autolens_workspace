"""
Interferometer: Data Preparation
================================

When an interferometer dataset is analysed, it must conform to certain standards in order for
the analysis to be performed correctly. This tutorial describes these standards and links to more detailed scripts
which will help you prepare your dataset to adhere to them if it does not already.

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
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

# %matplotlib inline
from os import path
import autolens as al
import autolens.plot as aplt

dataset_path = path.join("dataset", "interferometer", "simple")

"""
__Visibilities__

The image is the image of your strong lens, which comes from a telescope like the Hubble Space telescope (HST).

Lets inspect an image which conforms to **PyAutoLens** standards:
"""
visibilities = al.Visibilities.from_fits(
    file_path=path.join(dataset_path, "data.fits"), hdu=0
)

array_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
array_plotter.figure_2d()

"""
These visibilities conforms to **PyAutoLens** standards, because they come from a standard CASA data reduction
procedure. 

More details of this procedure are given in the `examples/case_to_autolens.ipynb` notebook.

__Noise-Map__

The noise-map is the real and complex noise in each visiblity of the interferometer dataset. It is used to weight
the visibilities when a lens model is fitted to the data via a chi-squared statistic.

It is common for all visibilities to have the same noise value, depending on the instrument used to observe the
the data.
"""
visibilities = al.VisibilitiesNoiseMap.from_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), hdu=0
)

array_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
array_plotter.figure_2d()

"""
__UV Wavelengths__

The uv-wavelengths define the baselines of the interferometer. They are used to Fourier transform the image to the
uv-plane, which is where the lens model is evaluated.
"""
uv_wavelengths = al.util.array_2d.numpy_array_2d_via_fits_from(
    file_path=path.join(dataset_path, "uv_wavelengths.fits"), hdu=0
)

uv_wavelengths = al.Grid2DIrregular.from_yx_1d(
    y=uv_wavelengths[:, 1] / 10**3.0,
    x=uv_wavelengths[:, 0] / 10**3.0,
)

grid_plotter = aplt.Grid2DPlotter(grid=uv_wavelengths)
grid_plotter.figure_2d()

"""
These uv wavelengths conform to **PyAutoLens** standards, because they come from a standard CASA data reduction
procedure. 

More details of this procedure are given in the `examples/case_to_autolens.ipynb` notebook.

__Real Space Mask__

The `modeling` scripts also define a real-space mask, which defines where the image is evalated in real space 
before it is Fourier transformed.

You must double check that the real-space mask you use:
 
 - Spatially covers the lensed source galaxy, such that the source is not truncated by the mask.
 - Is high enough resolution that the lensed source galaxy is not smeared via the Fourier transform.
 
__Run Times__

If you are analysing an interfeometer dataset with many visibilities (e.g. 1 million and above) and a high 
resolution real-space mask (e.g. 0.01" / pixel), the analysis can take a long time to run. 

The `examples/run_times.ipynb` script shows how to profile and setup your analysis to ensure it have fast enough
run times.

__Data Processing Complete__

If your visibilities, noise-map, uv_wavelengths and real space mask conform the standards above, you are ready to analyse 
your dataset!

Below, we provide an overview of optional data preparation steps which prepare other aspects of the analysis. 

New users are recommended to skim-read the optional steps below so they are aware of them, but to not perform them 
and instead analyse their dataset now. You can come back to the data preparation scripts below if it becomes necessary.

The following scripts are used to prepare components of an interferometer dataset, however they are used in an
identical fashion for dataset datasets.

Therefore, they are not located in the `interferometer/data_preparation` package, but instead in the
`data_preparation/imaging` package, so refer there for a description of their usage.

Note that in order to perform some tasks (e.g. mark on the image where the source is), you will need to use an image
of the interferometer data even though visibilities are used for the analysis.


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

The example `data_preparation/examples/optional/lens_light_centre.py` shows how to input the lens galaxy light centre
manually into a Python script.

The script `data_preparation/gui/lens_light_centre.ipynb` shows how to use a Graphical User Interface (GUI) to mask the
lens galaxy light centres.


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

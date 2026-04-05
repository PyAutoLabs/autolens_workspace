"""
GUI Preprocessing: Extra Galaxies Mask (Optional)
=================================================

There may be regions of an image that have signal near the lens and source that is from other galaxies not associated
with the strong lenswe are studying. The emission from these images will impact our model fitting and needs to be
removed from the analysis.

The example `imaging/data_preparation/example/optional/extra_galaxies_mask.py` provides a full description of
what the extra galaxies are and how they are used in the model-fit. You should read this script first before
using this script.

This script uses a GUI to mark the regions of the image where these extra galaxies are located, in contrast to the
example above which requires you to input these values manually.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Scribbler:** Load the Scribbler GUI for spray painting the scaled regions of the dataset.
**Output:** The new image is plotted for inspection.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Dataset__

The path where the extra galaxy mask is output, which is `dataset/imaging/extra_galaxies`.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` data, where the extra galaxies are visible in the data.
"""
data = al.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

data = al.Array2D(
    values=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), mask=data.mask
)

cmap = "jet"

"""
__Mask__

Create a 3.0" mask to plot over the image to guide where extra galaxy light needs its emission removed and noise scaled.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales, radius=mask_radius
)

"""
__Scribbler__

Load the Scribbler GUI for spray painting the scaled regions of the dataset. 

Push Esc when you are finished spray painting.
"""
scribbler = al.Scribbler(image=data.native, cmap=cmap, mask_overlay=mask)
mask = scribbler.show_mask()
mask = al.Mask2D(mask=mask, pixel_scales=pixel_scales)

"""
The GUI has now closed and the extra galaxies mask has been created.

Apply the extra galaxies mask to the image, which will remove them from visualization.
"""
data = data.apply_mask(mask=mask)

"""
__Output__

The new image is plotted for inspection.
"""
aplt.plot_array(array=data, title="")

"""
Plot the data with the new mask, in order to check that the mask removes the regions of the image corresponding to the
extra galaxies.
"""
aplt.plot_array(array=data, title="")

"""
__Output__

Output to a .png file for easy inspection.
"""
aplt.plot_array(array=data, title="")

"""
Output the extra galaxies mask, which will be load and used before a model fit.
"""
aplt.fits_array(
    array=mask,
    file_path=Path(dataset_path, "mask_extra_galaxies.fits"), overwrite=True
)

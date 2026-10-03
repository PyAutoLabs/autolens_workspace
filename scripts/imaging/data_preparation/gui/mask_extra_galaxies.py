"""
GUI Preprocessing: Extra Galaxies Mask (Optional)
=================================================

There may be regions of an image that have signal near the lens and source that is from other galaxies not associated
with the strong lenswe are studying. The emission from these images will impact our model fitting and needs to be
removed from the analysis.

The example `imaging/data_preparation/examples/optional/mask_extra_galaxies.py` provides a full description of
what the extra galaxies are and how they are used in the model-fit. You should read this script first before
using this script.

This script uses a GUI to mark the regions of the image where these extra galaxies are located, in contrast to the
example above which requires you to input these values manually.

__Contents__

- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Scribbler:** Load the Scribbler GUI for spray painting the scaled regions of the dataset.
- **Refining An Existing Mask:** Reopen a saved mask as a proposal and add to / erase from it.
- **Output:** The new image is plotted for inspection.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

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
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/extra_galaxies/simulator.py"],
        check=True,
    )

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

Create a 4.0" `guide_mask` to plot over the image to guide where extra galaxy light needs its emission removed and
noise scaled. It is only a visual guide: the extra galaxies mask itself is what you draw in the GUI.

It is better to draw the extra galaxies mask over too large an area than too small: a mask can always be made
smaller during the analysis, but making it larger will require the extra galaxies mask to be re-drawn.
"""
mask_radius = 4.0

guide_mask = al.Mask2D.circular(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales, radius=mask_radius
)

"""
__Scribbler__

Load the Scribbler GUI for spray painting the scaled regions of the dataset.

Two brushes are available: press `1` for the white brush, which ADDS pixels to the mask, and `2` for the black
brush, which ERASES them. Press `=` / `-` to make the brush bigger / smaller (each press scales it by 1.4x), `z` to
undo the last stroke and Esc when you are finished.

`mask_from()` returns everything painted white that was not painted black.

__Refining An Existing Mask__

To adjust an extra galaxies mask drawn previously instead of starting from a blank image, set
`refine_existing = True`. If the saved mask exists it is loaded and passed to the GUI as a `proposal`, instead of
opening a blank canvas. Its boundary is outlined in black over the image, and `mask_from()` then returns the
proposal plus whatever you paint white, minus whatever you paint black. With `refine_existing = False` (or no saved
mask yet) the GUI opens on the blank image.

The same route lets a mask drawn for one waveband of a multi-wavelength dataset seed the next, provided the two
images share a pixel grid.
"""
refine_existing = False

mask_path = Path(dataset_path, "mask_extra_galaxies.fits")

if refine_existing and mask_path.exists():
    previous = al.Mask2D.from_fits(file_path=mask_path, pixel_scales=pixel_scales)
    scribbler = al.Scribbler(
        image=data.native,
        proposal=np.asarray(previous),
        cmap=cmap,
        mask_overlay=guide_mask,
        title="Mask extra galaxies, refining the saved mask",
    )
else:
    scribbler = al.Scribbler(
        image=data.native,
        cmap=cmap,
        mask_overlay=guide_mask,
        title="Mask extra galaxies",
    )

mask = al.Mask2D(mask=scribbler.mask_from(), pixel_scales=pixel_scales)

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
    array=mask, file_path=Path(dataset_path, "mask_extra_galaxies.fits"), overwrite=True
)

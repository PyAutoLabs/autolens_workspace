"""
Data Preparation: Extra Galaxies Mask (Optional)
================================================

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

This script outputs a `mask_extra_galaxies.fits` file, which can be loaded and used before a model fit, in whatever
way is appropriate for the model being fitted.

__Links / Resources__

The script `data_preparation/gui/extra_galaxies_mask.ipynb` shows how to use a Graphical User Interface (GUI) to create
the extra galaxies mask.

__Contents__

**Output:** Output to a .png file for easy inspection.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np

import autolens as al
import autolens.plot as aplt

"""
The path where the extra galaxy mask is output, which is `dataset/imaging/extra_galaxies`.

The corresponding simulator (`scripts/imaging/features/extra_galaxies/simulator.py`) already writes a default
`mask_extra_galaxies.fits` automatically. This script demonstrates how to override that default with your own
centres + radii — useful when working with real data where the extra galaxy locations are not known in advance.
"""
dataset_type = "imaging"
dataset_name = "extra_galaxies"
dataset_path = Path("dataset", dataset_type, dataset_name)

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
Load the dataset image, so that the location of galaxies is clear when scaling the noise-map.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.1)

aplt.plot_array(array=data, title="")

"""
Define the extra galaxies mask as the union of circles centred on each extra galaxy.

`Mask2D.circular` honours `PYAUTO_SMALL_DATASETS=1` (caps to 15x15 at 0.6"/px) and works on the actual loaded
data shape, so the same code path runs under both normal and small-dataset modes without modification.
"""
extra_galaxies_mask = np.zeros(data.shape_native, dtype=bool)

for centre, radius in [
    ((1.0, 3.5), 1.5),
    ((-2.0, -3.5), 2.4),
]:
    circle = al.Mask2D.circular(
        shape_native=data.shape_native,
        pixel_scales=data.pixel_scales,
        centre=centre,
        radius=radius,
        invert=True,  # True inside the circle (i.e. masked region)
    )
    extra_galaxies_mask = np.logical_or(extra_galaxies_mask, circle.native)

mask = al.Mask2D(mask=extra_galaxies_mask, pixel_scales=data.pixel_scales)

"""
Apply the extra galaxies mask to the image, which will remove them from visualization.
"""
data = data.apply_mask(mask=mask)

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

"""
The workspace also includes a GUI for image and noise-map scaling, which can be found at
`autolens_workspace/*/imaging/data_preparation/gui/mask_extra_galaxies.py`.

This tools allows you `spray paint` on the image where an you want to scale, allow irregular patterns (i.e. not
rectangles) to be scaled.
"""

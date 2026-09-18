"""
GUI Preprocessing: Mask
=======================

This tool allows one to mask a bespoke mask for a given image of a strong lens using an interactive GUI. This mask
can then be loaded before a pipeline is run and passed to that pipeline so as to become the default masked used by a
search (if a mask function is not passed to that search).

This GUI is adapted from the following code: https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab

__Contents__

- **Dataset:** Load and plot the strong lens dataset.
- **Scribbler:** Load the Scribbler GUI for drawing the mask.
- **Refining An Existing Mask:** Reopen a saved mask as a proposal and add to / erase from it.
- **Output:** Now lets plot the image and mask, so we can check that the mask includes the regions of the image.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Dataset__

Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/simple__no_lens_light`.
"""
dataset_name = "simple__no_lens_light"
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
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` dataset, so that the mask can be plotted over the strong lens image.
"""
data = al.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

"""
__Scribbler__

Load the Scribbler GUI for drawing the mask, painting over the region of the image you want to fit.

Two brushes are available: press `1` for the green brush, which ADDS pixels to the painted region, and `2` for the red
brush, which ERASES them. Press `=` / `-` to make the brush bigger / smaller (each press scales it by 1.4x), `z` to
undo the last stroke and Esc when you are finished.

`mask_from()` returns everything painted green that was not painted red.
"""
scribbler = al.Scribbler(image=data.native)
mask = scribbler.mask_from()
mask = al.Mask2D(mask=np.invert(mask), pixel_scales=pixel_scales)

"""
__Refining An Existing Mask__

To adjust a mask drawn previously instead of starting from a blank image, load it and pass it to the GUI as a
`proposal`. Its boundary is outlined in white over the image, and `mask_from()` then returns the proposal plus
whatever you paint green, minus whatever you paint red.

The `.fits` written at the end of this script stores the region to *exclude* (`True` = masked), so it is inverted
back to the painted region before being passed as the proposal, and inverted again afterwards. Set
`refine_existing = True` to use this instead of the blank-canvas draw above.
"""
refine_existing = False

mask_path = Path(dataset_path, "mask_gui.fits")

if refine_existing and mask_path.exists():
    previous = al.Mask2D.from_fits(file_path=mask_path, pixel_scales=pixel_scales)
    scribbler = al.Scribbler(
        image=data.native, proposal=np.invert(np.asarray(previous))
    )
    mask = al.Mask2D(mask=np.invert(scribbler.mask_from()), pixel_scales=pixel_scales)

"""
__Output__

Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
"""
aplt.plot_array(array=data, title="")

"""
Output this image of the mask to a .png file in the dataset folder for future reference.
"""
aplt.plot_array(array=data, title="")

"""
Output it to the dataset folder of the lens, so that we can load it from a .fits in our modeling scripts.
"""
aplt.fits_array(
    array=mask, file_path=Path(dataset_path, "mask_gui.fits"), overwrite=True
)

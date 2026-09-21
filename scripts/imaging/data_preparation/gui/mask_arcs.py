"""
GUI Preprocessing: Arc Mask (Optional)
======================================

This tool paints the lensed ARCS and multiple images of the source, and keeps only those: everything you do not
paint, lens light included, is masked. The product isolates the source, which is what a source-only fit, an arc
signal-to-noise measurement or a source-plane analysis should look at.

It is the OPPOSITE polarity to `mask_extra_galaxies.py`, and the third of the three mask-up steps:

1. `mask_extra_galaxies.py`: paint what to REMOVE (contaminants).
2. `positions.py`: click the multiple images.
3. `mask_arcs.py` (this script): paint the arcs to KEEP.

Steps 1 and 2 are shown while you paint, if their products exist: the contaminant mask's edge as black crosses,
so a neighbour already judged a contaminant is not mistaken for a lensed image, and each marked position as a dark
cross, so the arc mask is drawn around the same multiple images.

The lens galaxy's light is subtracted from the DISPLAY (`subtract_radial=True`) because on most lenses the arcs are
invisible under it. The subtracted image is shown on the left and the image as observed on the right; you can paint
on either. This changes only what you see, not what a stroke masks.

__Contents__

- **Dataset:** Load the strong lens dataset and, if present, the contaminant mask and positions from steps 1 and 2.
- **Scribbler:** Paint the arcs over the lens-light-subtracted display.
- **Refining An Existing Mask:** Reopen a saved arc mask as a proposal and add to / erase from it.
- **Output:** Save the arc mask in `Mask2D` convention (True = masked) and plot the arcs it isolates.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Dataset__

The dataset folder the arc mask is read from and written to, `dataset/imaging/simple`, which has lens light and
a faint extra galaxy.
"""
dataset_name = "simple"
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
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` data.
"""
data = al.Array2D.from_fits(
    file_path=dataset_path / "data.fits", pixel_scales=pixel_scales
)

data = al.Array2D(
    values=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), mask=data.mask
)

"""
Load the products of the earlier steps where they exist. The contaminant mask (step 1) is shown as a `mask_overlay`,
whose edge is scattered over the image; the positions (step 2) are shown as crosses. Neither enters the arc mask.
"""
mask_extra_galaxies_path = dataset_path / "mask_extra_galaxies.fits"

mask_extra_galaxies = (
    al.Mask2D.from_fits(file_path=mask_extra_galaxies_path, pixel_scales=pixel_scales)
    if mask_extra_galaxies_path.exists()
    else None
)

positions_path = dataset_path / "positions.json"

positions = al.from_json(file_path=positions_path) if positions_path.exists() else None

"""
__Scribbler__

Paint ONLY the arcs and multiple images. The left panel has the lens galaxy's radial light profile subtracted so
the arcs stand out; the right panel is the image as observed. Paint on either.

Two brushes are available: press `1` for the green brush, which ADDS pixels to the arc region, and `2` for the red
brush, which ERASES them. Press `=` / `-` to make the brush bigger / smaller, `z` to undo the last stroke and Esc
when you are finished.

`mask_from()` returns the painted arc region; it is INVERTED before saving so that, like every other mask, `True`
means excluded from the fit.
"""
scribbler = al.Scribbler(
    image=data.native,
    cmap="jet",
    subtract_radial=True,
    mask_overlay=mask_extra_galaxies,
    positions=positions,
)

arc_region = scribbler.mask_from()
mask = al.Mask2D(mask=np.invert(arc_region), pixel_scales=pixel_scales)

"""
__Refining An Existing Mask__

To adjust an arc mask drawn previously, load it, invert it back to the painted arc region and pass that as a
`proposal`: its boundary is outlined in white and `mask_from()` returns the proposal plus whatever you paint green,
minus whatever you paint red. Set `refine_existing = True` to use this instead of the blank-canvas draw above.
"""
refine_existing = False

mask_arcs_path = dataset_path / "mask_arcs.fits"

if refine_existing and mask_arcs_path.exists():
    previous = al.Mask2D.from_fits(file_path=mask_arcs_path, pixel_scales=pixel_scales)
    scribbler = al.Scribbler(
        image=data.native,
        cmap="jet",
        subtract_radial=True,
        mask_overlay=mask_extra_galaxies,
        positions=positions,
        proposal=np.invert(np.asarray(previous)),
    )
    arc_region = scribbler.mask_from()
    mask = al.Mask2D(mask=np.invert(arc_region), pixel_scales=pixel_scales)

"""
__Output__

Plot the data with the arc mask applied, so only the painted arcs remain, to check it isolates the source.
"""
aplt.plot_array(array=data.apply_mask(mask=mask), title="")

"""
Output the arc mask to the dataset folder. Load it with `al.Mask2D.from_fits`; `~mask` is the arc region.
"""
aplt.fits_array(array=mask, file_path=mask_arcs_path, overwrite=True)

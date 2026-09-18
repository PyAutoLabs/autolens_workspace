"""
GUI Preprocessing: Extra Galaxies Mask, Multi-Wavelength (Optional)
===================================================================

A multi-wavelength dataset needs one extra galaxies mask per waveband: what should be masked is not the same in
every filter (a contaminant can be bright in one and absent in another), and the wavebands are often on different
pixel grids, so a mask cannot simply be copied across.

This script draws the mask on the first waveband and then, instead of starting the next waveband from a blank
image, REGRIDS the first mask onto its pixel grid and opens it as a `proposal` to review there: the proposal is
outlined over the image, and you add to it with the green brush and erase from it with the red one.

It also shows the GUI's `subtract_radial` display, which removes the galaxy's azimuthally-averaged light profile
from the displayed image (the mask itself is unaffected) so that faint structure hidden under the galaxy is
visible while you paint.

The example `imaging/data_preparation/gui/mask_extra_galaxies.py` covers the single-waveband GUI and should be
read first.

__Contents__

- **Dataset:** Load two wavebands of the COSMOS-Web ring, which are on different pixel grids.
- **First Waveband:** Draw the extra galaxies mask on the first waveband using the radial-subtracted display.
- **Next Waveband:** Regrid that mask onto the next waveband, review it there and apply, reject or skip it.
- **Output:** Save one mask per waveband.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Dataset__

Multi-wavelength James Webb Space Telescope imaging of the COSMOS-Web ring, the same dataset
`multi_dataset/start_here.py` models. F150W is on a 0.03" pixel grid and F277W on a 0.06" grid, so a mask drawn
on one must be regridded to fit the other.
"""
waveband_list = ["F150W", "F277W"]

pixel_scale_dict = {
    "F150W": 0.03,
    "F277W": 0.06,
}

dataset_name = "cosmos_web_ring"
dataset_path = Path("dataset") / "imaging" / dataset_name / "wavebands"

data_dict = {}

for waveband in waveband_list:
    data = al.Array2D.from_fits(
        file_path=dataset_path / waveband / "data.fits",
        pixel_scales=pixel_scale_dict[waveband],
    )

    data_dict[waveband] = al.Array2D(
        values=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), mask=data.mask
    )

"""
__First Waveband__

Draw the mask on the first waveband from a blank image.

`subtract_radial=True` shows two panels: on the LEFT the image with its azimuthally-averaged radial profile
subtracted, which lifts faint arcs and companions out from under the galaxy's light; on the RIGHT the image as
observed, where a contaminant's true extent is judged. You can paint on either panel, since both land on the same
pixels.

Two brushes are available: press `1` for the green brush, which ADDS pixels to the mask, and `2` for the red
brush, which ERASES them. Press `=` / `-` to make the brush bigger / smaller, `z` to undo the last stroke and
Esc when you are finished.
"""
waveband = waveband_list[0]
data = data_dict[waveband]

scribbler = al.Scribbler(image=data.native, cmap="jet", subtract_radial=True)
mask = al.Mask2D(mask=scribbler.mask_from(), pixel_scales=data.pixel_scales)

mask_dict = {waveband: mask}

aplt.plot_array(array=data.apply_mask(mask=mask), title=f"{waveband} mask")

"""
__Next Waveband__

The next waveband starts from the first waveband's mask rather than a blank image, but never blindly: the mask is
first regridded onto this waveband's pixel grid (nearest neighbour in arc-second coordinates, so it covers the
same sky) and then opened as a `proposal`, outlined in white over the image, for you to correct with the two
brushes.
"""
from autogalaxy.gui.display_util import mask_2d_regridded_from

previous_waveband = waveband_list[0]
waveband = waveband_list[1]
data = data_dict[waveband]

proposal = mask_2d_regridded_from(
    mask=mask_dict[previous_waveband],
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
)

scribbler = al.Scribbler(
    image=data.native, cmap="jet", subtract_radial=True, proposal=np.asarray(proposal)
)

"""
After the GUI closes, decide what to do with the reviewed proposal:

- `a` (default): apply the proposal plus your edits.
- `d`: reject the proposal and keep only what you painted green (minus red).
- `s`: skip this waveband and write nothing for it.
"""
choice = input("[a]pply proposal + edits / [d]rawn only / [s]kip: ").strip().lower()

if choice.startswith("s"):
    print(f"{waveband}: skipped, no mask written")
elif choice.startswith("d"):
    mask = scribbler.mask_from(proposal=np.zeros(data.shape_native, dtype=bool))
    mask_dict[waveband] = al.Mask2D(mask=mask, pixel_scales=data.pixel_scales)
else:
    mask_dict[waveband] = al.Mask2D(
        mask=scribbler.mask_from(), pixel_scales=data.pixel_scales
    )

if waveband in mask_dict:
    aplt.plot_array(
        array=data.apply_mask(mask=mask_dict[waveband]), title=f"{waveband} mask"
    )

"""
For a dataset with more than two wavebands, repeat the block above for each further waveband, regridding from
the waveband whose mask you trust most.

__Output__

Output one mask per waveband, in that waveband's folder, so a modeling script can load each with
`al.Mask2D.from_fits` using that waveband's `pixel_scales`.
"""
for waveband, mask in mask_dict.items():
    aplt.fits_array(
        array=mask,
        file_path=dataset_path / waveband / "mask_extra_galaxies_gui.fits",
        overwrite=True,
    )

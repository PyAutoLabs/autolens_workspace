"""
Data Preparation: Datacube
============================

Most users come to datacube modeling with a single 4D FITS file from CASA, with shape
``(n_pol, n_chan, n_vis, 2)`` — for example, an ALMA observation of a CO emission line might be
``(2, 34, 16984, 2)``: two polarisations, 34 spectral channels, ~17k visibilities per channel, real/imag.

This script shows how to bridge from that on-disk shape to the canonical input PyAutoLens expects:

  * **In memory**: a Python ``list`` of ``al.Interferometer`` objects, one per channel.
  * **On disk** (canonical 3D layout): three FITS files ``visibilities_cube.fits``,
    ``noise_map_cube.fits`` and ``uv_wavelengths_cube.fits``, all of shape ``(n_chan, n_vis, 2)``.

Two preprocessing steps are usually needed before the 3D layout is reached:

  1. **Polarisation collapse** — averaging or concatenating the two polarisation entries.
  2. **Optional `uv_wavelengths` / `noise_map` reduction** — both quantities change very little
     channel-to-channel, so for the purposes of modeling they're often averaged across channels and
     stored as a single ``(n_vis, 2)`` array shared across the cube.

The script is structured as runnable explanation. The reusable bit is
``dataset_list_from_3d_fits()`` near the bottom — copy that into your own script if you have a 3D
cube already laid out on disk.

__Contents__

- **What ALMA Gives You:** The 4D shape `(n_pol, n_chan, n_vis, 2)` straight from CASA.
- **Polarisation Handling:** Average vs concatenate; tradeoffs and one-line numpy.
- **Canonical 3D Shape:** `(n_chan, n_vis, 2)` is what every loader below assumes.
- **Shared vs Per-Channel uv_wavelengths / noise_map:** When channel-invariant quantities can be a
  single shared `(n_vis, 2)` array, and how the loader handles either case.
- **3D-FITS Loader:** A self-contained `dataset_list_from_3d_fits()` function that builds the
  in-memory `dataset_list`.
- **Worked Example:** Loads the reference cube (written by `simulator.py`) and confirms the
  resulting `dataset_list` matches what the per-channel-folder loader would produce.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np

import autolens as al

"""
__What ALMA Gives You__

A typical ALMA spectral-line `visibilities.fits` from CASA has shape ``(n_pol, n_chan, n_vis, 2)``.
For Hannah's data this is ``(2, 34, 16984, 2)``: two polarisations, 34 channels of an emission line,
about 17k visibilities per channel, real/imag.

`noise_map.fits` and `uv_wavelengths.fits` typically share the polarisation and channel dimensions
(though `uv_wavelengths` may collapse to `(n_pol, n_vis, 2)` if your reduction has decided the
baselines are channel-invariant — which is usually a good approximation for narrow emission lines).

The simulator in this folder writes a 4D `(n_pol, n_chan, n_vis, 2)` cube alongside the per-channel
folders and the 3D `(n_chan, n_vis, 2)` autolens-native cube. We load the simulator's actual 4D
output below so the polarisation-collapse demonstration runs on the same data the rest of the
modeling scripts will fit. (Hint: for this synthetic simulator the two polarisations are identical —
real CASA data has independent noise realisations between pols, which is why averaging real data
reduces effective noise by `sqrt(2)`. The collapse code below is the same in both cases.)
"""
dataset_label = "datacube"
dataset_name = "sim_simple"
dataset_path = Path("dataset") / "interferometer" / dataset_label / dataset_name

if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/features/datacube/simulator.py"],
        check=True,
    )

visibilities_4d = al.ndarray_via_fits_from(
    file_path=dataset_path / "visibilities_4d_cube.fits", hdu=0
)
noise_map_4d = al.ndarray_via_fits_from(
    file_path=dataset_path / "noise_map_4d_cube.fits", hdu=0
)
uv_wavelengths_4d = al.ndarray_via_fits_from(
    file_path=dataset_path / "uv_wavelengths_4d_cube.fits", hdu=0
)

print(f"On-disk shapes (loaded from simulator's *_4d_cube.fits):")
print(f"  visibilities_4d:    {visibilities_4d.shape}  (n_pol, n_chan, n_vis, 2)")
print(f"  noise_map_4d:       {noise_map_4d.shape}     (n_pol, n_chan, n_vis, 2)")
print(f"  uv_wavelengths_4d:  {uv_wavelengths_4d.shape}     (n_pol, n_chan, n_vis, 2)")

"""
__Polarisation Handling__

Two options for collapsing the polarisation axis. Pick whichever your reduction was designed for —
the workspace doesn't bake a choice in.

- **Average** preserves ``n_vis``. Best when the two polarisations carry the same source signal and
  you want to suppress polarisation-mode noise:

      visibilities_collapsed = visibilities_4d.mean(axis=0)        # (n_chan, n_vis, 2)

  For the noise map use inverse-variance weighting, which for equal-noise polarisations reduces to
  ``noise_map.mean(axis=0) / sqrt(2)``:

      noise_map_collapsed = noise_map_4d.mean(axis=0) / np.sqrt(2)  # (n_chan, n_vis, 2)

- **Concatenate** doubles ``n_vis``. Best when you want to keep every visibility independent and let
  the model fit each polarisation separately (or when the polarisations carry slightly different
  signal — the average would smear that out):

      visibilities_collapsed = np.concatenate([visibilities_4d[0], visibilities_4d[1]], axis=1)  # (n_chan, 2*n_vis, 2)
      noise_map_collapsed   = np.concatenate([noise_map_4d[0], noise_map_4d[1]], axis=1)         # (n_chan, 2*n_vis, 2)
      uv_wavelengths_collapsed = np.concatenate([uv_wavelengths_4d[0], uv_wavelengths_4d[1]], axis=1)  # (n_chan, 2*n_vis, 2)

We use the averaging path below for clarity. Swap in the concatenate version if that's what your
reduction expects.
"""
visibilities_3d = visibilities_4d.mean(axis=0)
noise_map_3d = noise_map_4d.mean(axis=0) / np.sqrt(2)
uv_wavelengths_3d = uv_wavelengths_4d.mean(axis=0)
# uv_wavelengths is channel-invariant in this synthetic simulator, so we can also collapse the
# channel axis if you prefer the shared 2D form. The loader supports either:
uv_wavelengths_2d = uv_wavelengths_3d.mean(axis=0)

print(f"\nAfter polarisation averaging:")
print(f"  visibilities_3d:    {visibilities_3d.shape}    (n_chan, n_vis, 2)")
print(f"  noise_map_3d:       {noise_map_3d.shape}    (n_chan, n_vis, 2)")
print(
    f"  uv_wavelengths_3d:  {uv_wavelengths_3d.shape}    (n_chan, n_vis, 2)  [per-channel]"
)
print(
    f"  uv_wavelengths_2d:  {uv_wavelengths_2d.shape}        (n_vis, 2)  [shared, optional]"
)

"""
__Canonical 3D Shape__

After polarisation collapse, every loader below assumes the canonical post-preprocessing shapes:

- ``visibilities``:    ``(n_chan, n_vis, 2)`` — required.
- ``noise_map``:       ``(n_chan, n_vis, 2)`` — required, can be channel-invariant if you replicate.
- ``uv_wavelengths``:  ``(n_chan, n_vis, 2)`` for per-channel baselines, **or** ``(n_vis, 2)`` for
  channel-invariant baselines (the loader broadcasts).

If your reduction stores `noise_map` as channel-invariant `(n_vis, 2)`, the loader broadcasts the
same way.
"""

"""
__Shared vs Per-Channel `uv_wavelengths` / `noise_map`__

For most ALMA narrow-line cubes both `uv_wavelengths` and `noise_map` change very little
channel-to-channel and can safely be stored as a single shared `(n_vis, 2)` array. The deferred
shared-`Lᵀ W̃ L` optimisation Aris designed (see issue #120) will only fire when these arrays are
actually shared, so there's some real performance reason to use the shared form when you can.

The loader below supports both: if the input `noise_map` or `uv_wavelengths` array is 2D, it gets
broadcast to all `n_chan` channels; if it's 3D, each channel gets its own slice.
"""

"""
__3D-FITS Loader__

Self-contained loader function. Copy this into your own script if you have a 3D cube on disk.

The function takes paths to the three FITS files (or numpy arrays directly — see the `_arrays`
variant below if you want to skip the disk trip), splits them per-channel, and constructs the list
of `al.Interferometer` objects.
"""


def dataset_list_from_3d_fits(
    visibilities_path: Path,
    noise_map_path: Path,
    uv_wavelengths_path: Path,
    real_space_mask: al.Mask2D,
    transformer_class=al.TransformerNUFFT,
):
    """Load a 3D-FITS datacube into a list of `al.Interferometer` objects.

    Parameters
    ----------
    visibilities_path
        Path to a `.fits` file of shape `(n_chan, n_vis, 2)` storing real/imag visibilities for
        every channel. Polarisations should already be collapsed.
    noise_map_path
        Path to a `.fits` file of shape `(n_chan, n_vis, 2)` (per-channel) or `(n_vis, 2)` (shared
        across channels). The shared form is broadcast to every channel.
    uv_wavelengths_path
        Path to a `.fits` file of shape `(n_chan, n_vis, 2)` (per-channel) or `(n_vis, 2)` (shared).
    real_space_mask
        The 2D real-space mask used by the Fourier transformer.
    transformer_class
        `al.TransformerNUFFT` (default, JAX-native via `nufftax`, scales to many millions of visibilities)
        or `al.TransformerDFT`.

    Returns
    -------
    list[al.Interferometer]
        One `Interferometer` per channel, in cube order.
    """
    visibilities_3d = al.ndarray_via_fits_from(file_path=visibilities_path, hdu=0)
    noise_map_arr = al.ndarray_via_fits_from(file_path=noise_map_path, hdu=0)
    uv_wavelengths_arr = al.ndarray_via_fits_from(file_path=uv_wavelengths_path, hdu=0)

    if visibilities_3d.ndim != 3:
        raise ValueError(
            f"visibilities array must be 3D (n_chan, n_vis, 2); got shape {visibilities_3d.shape}"
        )

    n_chan = visibilities_3d.shape[0]

    # Broadcast 2D shared arrays to (n_chan, n_vis, 2) so the per-channel loop is uniform.
    if noise_map_arr.ndim == 2:
        noise_map_arr = np.broadcast_to(
            noise_map_arr[None], visibilities_3d.shape
        ).copy()
    if uv_wavelengths_arr.ndim == 2:
        uv_wavelengths_arr = np.broadcast_to(
            uv_wavelengths_arr[None], visibilities_3d.shape
        ).copy()

    return [
        al.Interferometer(
            data=al.Visibilities(visibilities=visibilities_3d[c]),
            noise_map=al.VisibilitiesNoiseMap(visibilities=noise_map_arr[c]),
            uv_wavelengths=uv_wavelengths_arr[c],
            real_space_mask=real_space_mask,
            transformer_class=transformer_class,
        )
        for c in range(n_chan)
    ]


"""
__Worked Example__

Load the reference cube (already used above for the 4D-collapse demo) and confirm the 3D-FITS
loader produces a `dataset_list` numerically identical to the per-channel-folder loader. This is
a sanity check — once you trust this, you can drop the per-channel-folder pattern entirely if you
prefer the 3D layout.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=3.5,
)

dataset_list_3d = dataset_list_from_3d_fits(
    visibilities_path=dataset_path / "visibilities_cube.fits",
    noise_map_path=dataset_path / "noise_map_cube.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths_cube.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

# Cross-check against the per-channel-folder loader.
channel_paths = sorted(
    p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("channel_")
)
dataset_list_folders = [
    al.Interferometer.from_fits(
        data_path=channel_path / "data.fits",
        noise_map_path=channel_path / "noise_map.fits",
        uv_wavelengths_path=channel_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )
    for channel_path in channel_paths
]

assert len(dataset_list_3d) == len(dataset_list_folders), "channel count mismatch"

for c, (d3d, dfolder) in enumerate(zip(dataset_list_3d, dataset_list_folders)):
    np.testing.assert_allclose(
        np.asarray(d3d.data.real),
        np.asarray(dfolder.data.real),
        rtol=1e-12,
        err_msg=f"channel {c}: data.real mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(d3d.data.imag),
        np.asarray(dfolder.data.imag),
        rtol=1e-12,
        err_msg=f"channel {c}: data.imag mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(d3d.uv_wavelengths),
        np.asarray(dfolder.uv_wavelengths),
        rtol=1e-12,
        err_msg=f"channel {c}: uv_wavelengths mismatch",
    )

print(
    f"\n3D-FITS loader and per-channel-folder loader agree on {len(dataset_list_3d)} channels."
)
print(f"  visibilities/channel:  {dataset_list_3d[0].data.shape[0]}")
print(f"  uv_wavelengths shape:  {np.asarray(dataset_list_3d[0].uv_wavelengths).shape}")

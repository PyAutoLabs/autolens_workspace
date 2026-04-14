"""
Interferometer: CASA Reduction
==============================

This script is a practical walkthrough of how to reduce an ALMA / JVLA interferometer
observation with **CASA** and export it into the three FITS files that **PyAutoLens**
expects for lens modeling:

- `data.fits`         — complex visibilities, shape `(n_vis,)`
- `noise_map.fits`    — complex per-visibility RMS (sigma), shape `(n_vis,)`
- `uv_wavelengths.fits` — `(u, v)` baselines in units of *wavelengths*, shape `(n_vis, 2)`

The script is deliberately a hybrid: the sections that must run **inside the CASA
environment** (`split`, `statwt`, `uvcontsub`, `tb.open`, etc.) are shown as CASA
snippets you copy into a CASA session. The plain Python helpers at the bottom use
`numpy` and `astropy` to convert CASA-exported columns into the shapes PyAutoLens
requires, and can be run either inside CASA (which has a Python interpreter) or
in a normal Python environment once the FITS files have been produced.

The script is **a starting point**, not a finished tool. Every real dataset has
quirks (single-field vs. mosaic, spectral line vs. continuum, different numbers
of spectral windows, flagged antennas, polarisation products, ...). You will
almost certainly need to tweak it. If you get stuck, please contact us on the
PyAutoLens **SLACK** — interferometry support is actively evolving, and your
feedback directly shapes these examples.

__Contents__

**PyAutoLens Requirements:** What the `al.Interferometer` object expects as input.
**CASA in Five Steps:** The canonical CASA reduction pipeline for PyAutoLens.
**Step 1 — Split by Field / SPW:** Isolate the lens target and each spectral window.
**Step 2 — Channel Averaging:** Reduce visibility count by averaging channels.
**Step 3 — Continuum Subtraction:** Optional `uvcontsub` for line observations.
**Step 4 — Rescale Sigmas:** `statwt` makes the SIGMA column reflect real scatter.
**Step 5 — Export to FITS:** Python helpers to read the .ms and write .fits.
**Combining Spectral Windows:** Concatenate per-SPW arrays into a single dataset.
**Polarisations:** Averaging XX/YY (or RR/LL) into a single complex visibility.
**Building the Interferometer Object:** Load the FITS files into `al.Interferometer`.
**Troubleshooting:** Common pitfalls and what to check.
**SLACK:** Where to ask for help.

__PyAutoLens Requirements__

`al.Interferometer.from_fits(...)` loads three FITS files. **On disk**, they are
stored as plain real-valued arrays with a single `n_vis` axis, after polarisations
and channels have been collapsed into one long visibility list:

- `data.fits`           : real array, shape `(n_vis, 2)` — col 0 real, col 1 imag
- `noise_map.fits`      : real array, shape `(n_vis, 2)` — col 0 sigma_real, col 1 sigma_imag
- `uv_wavelengths.fits` : real array, shape `(n_vis, 2)` — cols are (u, v) in wavelengths

PyAutoLens converts the first two into complex arrays internally when the
`Interferometer` object is built.

Critically:

- The `uv` coordinates must be in **wavelengths**, not metres. The CASA `UVW`
  column stores metres, so you must multiply by `frequency / c` per channel.
- You must flatten *all* polarisation and channel axes into the single `n_vis`
  axis before writing the FITS. The standard order is:
  **(1) average polarisations → (2) reshape channels into n_vis → (3) concatenate SPWs**.
  If you skip step (1), the polarisations end up interleaved with the channels
  and the noise-map will be wrong.

__CASA in Five Steps__

A typical ALMA calibrated Measurement Set `my_obs.ms.split.cal` becomes PyAutoLens
input via this pipeline:

    1. split out the field and (optionally) per-SPW   → field/spw isolated .ms
    2. channel-average to reduce n_vis                 → chanaveraged .ms
    3. (line only) uvcontsub to remove continuum       → .ms.contsub
    4. statwt to rescale SIGMA                         → .ms.statwt
    5. export DATA / UVW / CHAN_FREQ / SIGMA to FITS   → data/noise/uv .fits

The rest of this script walks through each of these steps in order. All `split(...)`,
`statwt(...)`, `uvcontsub(...)`, `tb.open(...)` calls must be executed from within
a running CASA session (either interactively or via `casa -c my_script.py`).

__Step 1 — Split by Field / SPW__

ALMA `.ms` datasets typically contain multiple fields (calibrators + science
target) and multiple spectral windows (SPWs). Start by splitting out just the
science field. You can also split per-SPW at this stage, which makes later
channel-averaging simpler because different SPWs can have different numbers
of channels.
"""

# CASA:
# split(
#     vis="my_obs.ms.split.cal",
#     outputvis="my_obs_field_SPT-0418_spw0.ms",
#     keepmms=True,
#     field="SPT-0418",
#     spw="0",
#     datacolumn="data",     # use "corrected" if CORRECTED_DATA exists
#     keepflags=False,
# )

"""
Repeat the `split` for each SPW you want to include (e.g. spw="1", "2", "3").
If your `.ms` only has a single field and you want *all* SPWs in one file, you
can omit the `spw` argument.

__Step 2 — Channel Averaging__

ALMA observations often have thousands of channels per SPW, producing far more
visibilities than PyAutoLens can comfortably model (the light-profile path tops
out around ~10,000 visibilities; the pixelized source path scales to millions
but is more advanced). For continuum-style lens modeling you typically average
all channels in an SPW down to one (or a handful) by setting `width`.

Note that `width` must not exceed the number of channels in the SPW — check with
the `get_num_chan` helper below if unsure.
"""

# CASA:
# split(
#     vis="my_obs_field_SPT-0418_spw0.ms",
#     outputvis="my_obs_field_SPT-0418_spw0_chanavg.ms",
#     keepmms=True,
#     width=128,             # average 128 channels together
#     datacolumn="data",
#     keepflags=False,
# )

"""
__Step 3 — Continuum Subtraction (line observations only)__

If you're modeling a **spectral line** (e.g. CO(9-8) from a high-z lensed galaxy),
you want to remove the underlying continuum first using `uvcontsub`. Specify
the line-free channel ranges via `fitspw` and the channels to keep via `spw`.
Skip this step for pure continuum lens modeling.
"""

# CASA:
# uvcontsub(
#     vis="my_obs_field_SPT-0418_spw0.ms",
#     outputvis="my_obs_field_SPT-0418_spw0.ms.contsub",
#     fitspw="0:10~200;800~1000",    # line-free channel ranges for the fit
#     spw="0",                       # channels to keep in the output
#     fitorder=1,
# )

"""
If you ran `uvcontsub`, use the resulting `.ms.contsub` file as input to Step 4
and Step 5 instead of the original.

__Step 4 — Rescale Sigmas with statwt__

The `SIGMA` / `WEIGHT` columns that come out of ALMA calibration are often
set to nominal values rather than reflecting the true scatter in the visibilities.
`statwt` measures the scatter per-baseline and rescales SIGMA accordingly. This
is **strongly recommended** before exporting the noise-map — without it, your
per-visibility error bars will be wrong and the likelihood will be biased.

`statwt` modifies the `.ms` in place, so copy it first if you want to keep the
original SIGMAs for comparison.
"""

# CASA (from the shell / inside a CASA session):
# import os
# os.system("cp -r my_obs_field_SPT-0418_spw0_chanavg.ms "
#           "my_obs_field_SPT-0418_spw0_chanavg.ms.statwt")
# statwt(
#     vis="my_obs_field_SPT-0418_spw0_chanavg.ms.statwt",
#     datacolumn="data",
# )

"""
__Step 5 — Export DATA / UVW / CHAN_FREQ / SIGMA to FITS__

With the reduced `.ms` in hand we now read the relevant columns out with the
CASA `tb` tool and save them as FITS files. The helpers below can be saved
as a separate `.py` file and executed inside CASA via

    casa -c export_to_fits.py

or pasted directly into an interactive CASA session. They use the global `tb`
object CASA injects, together with numpy and astropy.
"""

import os
import numpy as np

try:
    from astropy import units, constants
    from astropy.io import fits
    astropy_is_imported = True
except ImportError:
    astropy_is_imported = False


def getcol_wrapper(ms, table, colname):
    """
    Open a CASA Measurement Set sub-table and return a squeezed column.

    Parameters
    ----------
    ms : str
        Path to the .ms directory.
    table : str
        Sub-table name (e.g. "SPECTRAL_WINDOW", "DATA_DESCRIPTION"). Empty
        string for the main table.
    colname : str
        Column name (e.g. "DATA", "UVW", "CHAN_FREQ", "SIGMA", "NUM_CHAN").

    Returns
    -------
    np.ndarray
        The requested column, with trivial dimensions removed.
    """
    if not os.path.isdir(ms):
        raise IOError(f"{ms} does not exist")

    tb.open(f"{ms}/{table}" if table else ms)  # noqa: F821 — `tb` is the CASA tool
    col = np.squeeze(tb.getcol(colname))        # noqa: F821
    tb.close()                                  # noqa: F821
    return col


def get_num_chan(ms):
    """Number of channels per SPW (shape `(n_spw,)` or scalar)."""
    return getcol_wrapper(ms=ms, table="SPECTRAL_WINDOW", colname="NUM_CHAN")


def get_spw_ids(ms):
    """SPW ids present in the main table (shape `(n_spw,)` or scalar)."""
    return getcol_wrapper(ms=ms, table="DATA_DESCRIPTION", colname="SPECTRAL_WINDOW_ID")


def get_frequencies(ms):
    """Channel frequencies in Hz (shape `(n_chan,)` or `(n_chan, n_spw)`)."""
    return getcol_wrapper(ms=ms, table="SPECTRAL_WINDOW", colname="CHAN_FREQ")


def _write_array(filename, data):
    """Write a numpy array as FITS if astropy is available, else `.npy`."""
    if astropy_is_imported:
        fits.writeto(filename=filename + ".fits", data=data, overwrite=True)
    else:
        with open(filename + ".numpy", "wb") as f:
            np.save(f, data)


"""
__Visibilities__

The `DATA` column has complex dtype and shape `(n_pol, n_chan, n_vis)` for a
single-SPW split. We stack the real and imaginary parts along a new trailing
axis so downstream code can load them as a real-valued FITS image and
recombine into a complex array via `vis[..., 0] + 1j * vis[..., 1]`.
"""


def get_visibilities(ms):
    data = getcol_wrapper(ms=ms, table="", colname="DATA")
    return np.stack(arrays=(data.real, data.imag), axis=-1)


def export_visibilities(ms, filename):
    if os.path.isfile(filename + ".fits") or os.path.isfile(filename + ".numpy"):
        print(f"{filename} already exists — skipping")
        return
    visibilities = get_visibilities(ms=ms)
    print("shape (visibilities):", visibilities.shape)
    _write_array(filename=filename, data=visibilities)


"""
__UV Wavelengths__

`UVW` in CASA is stored in **metres**, with shape `(3, n_vis)`. PyAutoLens
needs `(u, v)` in **wavelengths**, so we multiply by `frequency / c` for each
channel. This produces a `(2, n_chan, n_vis)` array of u, v wavelengths, which
is later reshaped to `(n_vis_total, 2)` once you flatten over channels.
"""


def convert_array_to_wavelengths(array, frequency):
    if astropy_is_imported:
        return (
            (array * units.m) * (frequency * units.Hz) / constants.c
        ).decompose().value
    return array * frequency / 299792458.0


def get_uv_wavelengths(ms):
    uvw = getcol_wrapper(ms=ms, table="", colname="UVW")
    chan_freq = get_frequencies(ms=ms)

    if np.shape(chan_freq):
        n_chan = np.shape(chan_freq)[0]
        u_wavelengths = np.zeros((n_chan, uvw.shape[1]))
        v_wavelengths = np.zeros((n_chan, uvw.shape[1]))
        for i in range(n_chan):
            u_wavelengths[i] = convert_array_to_wavelengths(uvw[0], chan_freq[i])
            v_wavelengths[i] = convert_array_to_wavelengths(uvw[1], chan_freq[i])
    else:
        u_wavelengths = convert_array_to_wavelengths(uvw[0], chan_freq)
        v_wavelengths = convert_array_to_wavelengths(uvw[1], chan_freq)

    return np.stack(arrays=(u_wavelengths, v_wavelengths), axis=-1)


def export_uv_wavelengths(ms, filename):
    if os.path.isfile(filename + ".fits") or os.path.isfile(filename + ".numpy"):
        print(f"{filename} already exists — skipping")
        return
    uv_wavelengths = get_uv_wavelengths(ms=ms)
    print("shape (uv_wavelengths):", uv_wavelengths.shape)
    _write_array(filename=filename, data=uv_wavelengths)


"""
__Sigma (Noise Map)__

The `SIGMA` column in CASA has shape `(n_pol, n_vis)` — one value per polarisation
per visibility — and CASA assigns the *same* sigma to the real and imaginary
components. We broadcast it across the channel axis and duplicate the last
axis so the final shape matches the visibilities: `(n_pol, n_chan, n_vis, 2)`.

For this to be meaningful you **must** have run `statwt` first, otherwise the
sigmas are nominal placeholders rather than real scatter estimates.
"""


def get_sigma(ms):
    sigma = getcol_wrapper(ms=ms, table="", colname="SIGMA")
    chan_freq = np.atleast_1d(get_frequencies(ms=ms))
    n_chan = chan_freq.shape[0]

    # Re-introduce the polarisation axis if np.squeeze removed it (n_pol == 1).
    if sigma.ndim == 1:
        sigma = sigma[np.newaxis, :]

    sigma = np.tile(sigma[:, np.newaxis, :], (1, n_chan, 1))
    return np.stack(arrays=(sigma, sigma), axis=-1)


def export_sigma(ms, filename):
    if os.path.isfile(filename + ".fits") or os.path.isfile(filename + ".numpy"):
        print(f"{filename} already exists — skipping")
        return
    sigma = get_sigma(ms=ms)
    print("shape (sigma):", sigma.shape)
    _write_array(filename=filename, data=sigma)


"""
__Polarisations (do this FIRST)__

Raw ALMA data has two polarisations (XX, YY) on axis 0 of `DATA` and `SIGMA`.
For lens modeling the standard approach is to average the two into a single
complex visibility with combined sigma, since the source emission is not
polarised for the vast majority of lensed galaxies. **Do this before
flattening channels or concatenating SPWs**, otherwise the polarisations end
up interleaved with the channel axis and the noise-map no longer corresponds
to the visibility it was measured from.

    # vis: (n_pol, n_chan, n_vis, 2) — real/imag on last axis
    vis_avg = 0.5 * (vis[0] + vis[1])                      # -> (n_chan, n_vis, 2)

    # sig: (n_pol, n_chan, n_vis, 2) — sigma_real/sigma_imag on last axis
    sig_avg = 0.5 * np.sqrt(sig[0] ** 2 + sig[1] ** 2)     # -> (n_chan, n_vis, 2)

If your science *does* care about polarisation, keep each pol as a separate
dataset and fit them jointly — contact us on Slack for the current state of
joint-polarisation modeling.

__Combining Spectral Windows__

Once you have per-SPW pol-averaged arrays, flatten channels into the n_vis
axis and concatenate across SPWs. All three arrays (visibilities, sigma,
uv-wavelengths) use the same ordering so they stay aligned.

    # Per-SPW (pol-averaged) shapes:
    #   vis_avg  : (n_chan_spw, n_vis_spw, 2)
    #   sig_avg  : (n_chan_spw, n_vis_spw, 2)
    #   uv       : (n_chan_spw, n_vis_spw, 2)

    vis_all = np.concatenate(
        [v.reshape(-1, 2) for v in per_spw_vis], axis=0
    )
    sig_all = np.concatenate(
        [s.reshape(-1, 2) for s in per_spw_sig], axis=0
    )
    uv_all = np.concatenate(
        [u.reshape(-1, 2) for u in per_spw_uv], axis=0
    )

After this step all three arrays have shape `(n_vis_total, 2)` — exactly
what `al.Interferometer.from_fits` expects.

__Building the Interferometer Object__

Write the three concatenated arrays to FITS (using `astropy.io.fits.writeto`
or `autoconf.fitsable.output_to_fits`) and load them via the canonical
workspace pattern used in `start_here.py`:

    import autolens as al

    # Small, fast real-space mask — keep shape_native and radius modest while
    # you are iterating on the reduction. Increase later for production fits.
    real_space_mask = al.Mask2D.circular(
        shape_native=(64, 64),
        pixel_scales=0.05,
        radius=1.5,
    )

    dataset = al.Interferometer.from_fits(
        data_path="data.fits",
        noise_map_path="noise_map.fits",
        uv_wavelengths_path="uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,   # NUFFT scales to large n_vis
    )

The 64x64 / 0.05" mask above keeps the Fourier transform and any dirty-image
sanity plots fast while you iterate — a first-pass check of your reduction
should finish in seconds, not minutes. Bump `shape_native` up (e.g. 256x256)
and drop `pixel_scales` (e.g. 0.02") for the real modeling run. Choosing
these numbers sensibly for your instrument is covered in
`scripts/interferometer/data_preparation.py`.

__Troubleshooting__

- **`uv_wavelengths` values look way too big/small** — you forgot the
  metres → wavelengths conversion, or you used the wrong frequency column.
- **Dirty image is offset / upside-down** — the `(u, v)` sign convention or
  the `(y, x)` axis order may disagree with your real-space mask. Plot the
  dirty image with `aplt.subplot_interferometer_dirty_images` as a sanity
  check.
- **Sigmas produce a flat chi-squared map** — you probably forgot `statwt`.
- **n_vis is enormous** — use a larger `width` in `split`, or switch from
  `TransformerDFT` to `TransformerNUFFT` (and the pixelized source workflow
  if n_vis > 10,000).
- **Only one polarisation present** — if CASA has flagged one pol the
  averaging code above will produce NaNs. Check with `get_visibilities`
  and branch accordingly.

__SLACK__

These scripts are a starting point, not a polished pipeline. If your
observation has mosaicking, heterogeneous SPWs, unusual correlators, or you
just can't get the numbers to line up — ping us on the PyAutoLens Slack.
Direct user feedback is actively shaping this workflow and we're happy to
help debug real datasets.

__Running as a Script__

The block below is illustrative — edit the paths/widths for your own reduction
and run inside CASA with `casa -c scripts/interferometer/casa_reduction.py`.
It assumes you have already run `split`, `uvcontsub` (optional) and `statwt`
by hand on the `.ms`.
"""

if __name__ == "__main__":

    uid = "my_obs_field_SPT-0418"
    width = 128

    ms_chanavg = f"{uid}_spw0_chanavg.ms"
    ms_statwt = f"{ms_chanavg}.statwt"

    export_uv_wavelengths(ms=ms_chanavg, filename=f"uv_wavelengths_{uid}_width_{width}")
    export_visibilities(ms=ms_chanavg, filename=f"visibilities_{uid}_width_{width}")
    export_sigma(ms=ms_statwt, filename=f"sigma_{uid}_width_{width}_statwt")

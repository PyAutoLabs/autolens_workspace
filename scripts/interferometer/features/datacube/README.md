The `datacube` folder contains example scripts showing how to model an ALMA-style spectral-line interferometer datacube as a list of per-channel `Interferometer` datasets sharing a single lens model.

# Files

The following example scripts illustrate datacube lens modeling where:

- `start_here`: A first walkthrough — load a 4-channel cube, build the FactorGraph, fit with Nautilus.
- `simulator`: Simulate a representative cube. Source `intensity` follows a Gaussian emission line; source `centre` shifts linearly across channels to mimic a kinematic gradient. Writes three layouts side by side: per-channel folders (`channel_NNN/{data,noise_map,uv_wavelengths}.fits`), a 3D-FITS cube (`{visibilities,noise_map,uv_wavelengths}_cube.fits`, `(n_chan, n_vis, 2)`), and a CASA-like 4D cube (`{visibilities,noise_map,uv_wavelengths}_4d_cube.fits`, `(n_pol, n_chan, n_vis, 2)`). Also writes `positions.json` and `cube_summary.json`.
- `data_preparation`: How to bridge from CASA's 4D `(n_pol, n_chan, n_vis, 2)` visibilities to the autolens-canonical `(n_chan, n_vis, 2)` layout — polarisation handling (average vs concatenate), run on the simulator's actual 4D output rather than a synthetic example. Includes a self-contained `dataset_list_from_3d_fits()` loader function (copy that into your own script if you have a 3D cube on disk).
- `modeling`: Focused FactorGraph + `RectangularBilinearAdaptDensity` pixelization fit, ready to point at your own cube.
- `modeling_parametric`: Same wiring with a parametric `Sersic` source — shared morphology, per-channel intensity. Faster than the pixelization variant; appropriate when the source is well-described by a single Sersic.
- `delaunay`: Same wiring with a Delaunay-pixelized source — image-plane `Overlay` mesh ray-traced and triangulated in the source plane, with `ConstantSplit` regularization. More flexible than the rectangular mesh; the canonical follow-up once the rectangular fit converges on a sensible lens model.
- `likelihood_function`: Step-by-step JAX walkthrough of the per-channel log-evidence sum with an explicit eager-vs-JIT correctness check at `rtol=1e-4`. Useful for understanding what `af.FactorGraphModel` is doing under the hood.

Every modeling script loads `positions.json` and applies an `al.PositionsLH` penalty. **For pixelized fits this is essentially required** — without the penalty the search routinely converges on demagnified-source local maxima where the source pixels are reconstructed in low-magnification regions of the source plane.

# On-disk Layouts

The simulator writes two user-facing on-disk layouts side by side, plus an intermediate autolens-canonical form. Pick whichever matches the data you actually have:

- **CASA-like 4D cube** — `{visibilities,noise_map,uv_wavelengths}_4d_cube.fits`, each of shape `(n_pol, n_chan, n_vis, 2)`. Closest to what your CASA reduction will give you — same shape as Hannah's `(2, 34, 16984, 2)` example. Needs a polarisation-collapse step (average or concatenate) before loading; `data_preparation.py` walks through that.

- **autolens-native (3D cube or per-channel folders)** — both produced by the simulator from the same data, post-polarisation-collapse:
  - **3D-FITS cube** — `{visibilities,noise_map,uv_wavelengths}_cube.fits`, each of shape `(n_chan, n_vis, 2)`. Load with `dataset_list_from_3d_fits()` from `data_preparation.py`.
  - **Per-channel folders** — `channel_NNN/{data,noise_map,uv_wavelengths}.fits`, each of shape `(n_vis, 2)`. Used by every modeling script in this folder by default — the loop-over-folders pattern from Phase 1.

The two autolens-native forms hold identical data and are interchangeable; the 3D cube is more convenient if you've just collapsed polarisations from a 4D file, and the per-channel folders are the Phase 1 default the modeling scripts already understand. The CASA-like 4D form is the form you'll actually receive from a real reduction.

# Overview

A "datacube" here is a Python list of `Interferometer` objects, one per spectral channel. The lens galaxy is shared across channels, and each channel reconstructs its own pixelized source — capturing the source's emission-line morphology channel by channel. The likelihood is the sum of per-channel log-evidences, which `af.FactorGraphModel` does for you.

This Phase 1 prototype runs each channel's NUFFT and inversion independently — simple, but the per-channel inversion cost dominates runtime on CPU. The shared-`Lᵀ W̃ L` optimisation that exploits channel-invariant `uv_wavelengths` and `noise_map` is a follow-up.

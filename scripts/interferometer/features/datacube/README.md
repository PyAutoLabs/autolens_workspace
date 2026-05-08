The `datacube` folder contains example scripts showing how to model an ALMA-style spectral-line interferometer datacube as a list of per-channel `Interferometer` datasets sharing a single lens model.

# Files

The following example scripts illustrate datacube lens modeling where:

- `start_here`: A first walkthrough — load a 4-channel cube, build the FactorGraph, fit with Nautilus.
- `simulator`: Simulate a representative cube with a Gaussian emission line in the source. Writes both layouts: per-channel folders (`channel_NNN/{data,noise_map,uv_wavelengths}.fits`) and a single 3D-FITS cube (`{visibilities,noise_map,uv_wavelengths}_cube.fits`, all shape `(n_chan, n_vis, 2)`). Also writes `positions.json`.
- `data_preparation`: How to bridge from CASA's 4D `(n_pol, n_chan, n_vis, 2)` visibilities to the canonical `(n_chan, n_vis, 2)` layout — polarisation handling (average vs concatenate) and a self-contained `dataset_list_from_3d_fits()` loader function. Copy that into your own script if you have a 3D cube on disk.
- `modeling`: Focused FactorGraph + `RectangularAdaptDensity` pixelization fit, ready to point at your own cube.
- `modeling_parametric`: Same wiring with a parametric `Sersic` source — shared morphology, per-channel intensity. Faster than the pixelization variant; appropriate when the source is well-described by a single Sersic.
- `delaunay`: Same wiring with a Delaunay-pixelized source — image-plane `Overlay` mesh ray-traced and triangulated in the source plane, with `ConstantSplit` regularization. More flexible than the rectangular mesh; the canonical follow-up once the rectangular fit converges on a sensible lens model.
- `likelihood_function`: Step-by-step JAX walkthrough of the per-channel log-evidence sum with an explicit eager-vs-JIT correctness check at `rtol=1e-4`. Useful for understanding what `af.FactorGraphModel` is doing under the hood.

Every modeling script loads `positions.json` and applies an `al.PositionsLH` penalty. **For pixelized fits this is essentially required** — without the penalty the search routinely converges on demagnified-source local maxima where the source pixels are reconstructed in low-magnification regions of the source plane.

# On-disk Layouts

The simulator writes both supported layouts side by side. Pick whichever matches the data you already have on disk:

- **Per-channel folders** — `channel_NNN/{data,noise_map,uv_wavelengths}.fits`, each `(n_vis, 2)`. Used by every modeling script in this folder by default.
- **3D-FITS cube** — `{visibilities,noise_map,uv_wavelengths}_cube.fits`, each `(n_chan, n_vis, 2)`. Matches what CASA gives you (after polarisation collapse). Use the `dataset_list_from_3d_fits()` loader from `data_preparation.py` to load this layout directly.

# Overview

A "datacube" here is a Python list of `Interferometer` objects, one per spectral channel. The lens galaxy is shared across channels, and each channel reconstructs its own pixelized source — capturing the source's emission-line morphology channel by channel. The likelihood is the sum of per-channel log-evidences, which `af.FactorGraphModel` does for you.

This Phase 1 prototype runs each channel's NUFFT and inversion independently — simple, but the per-channel inversion cost dominates runtime on CPU. The shared-`Lᵀ W̃ L` optimisation that exploits channel-invariant `uv_wavelengths` and `noise_map` is a follow-up.

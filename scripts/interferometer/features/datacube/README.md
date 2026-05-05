The `datacube` folder contains example scripts showing how to model an ALMA-style spectral-line interferometer datacube as a list of per-channel `Interferometer` datasets sharing a single lens model.

# Files

The following example scripts illustrate datacube lens modeling where:

- `start_here`: A first walkthrough — load a 4-channel cube, build the FactorGraph, fit with Nautilus.
- `simulator`: Simulate a representative cube with a Gaussian emission line in the source.
- `modeling`: Focused FactorGraph + pixelization fit, ready to point at your own cube.

# Overview

A "datacube" here is a Python list of `Interferometer` objects, one per spectral channel. The lens galaxy is shared across channels, and each channel reconstructs its own pixelized source — capturing the source's emission-line morphology channel by channel. The likelihood is the sum of per-channel log-evidences, which `af.FactorGraphModel` does for you.

This Phase 1 prototype runs each channel's NUFFT and inversion independently — simple, but the per-channel inversion cost dominates runtime on CPU. The shared-`Lᵀ W̃ L` optimisation that exploits channel-invariant `uv_wavelengths` and `noise_map` is a follow-up.

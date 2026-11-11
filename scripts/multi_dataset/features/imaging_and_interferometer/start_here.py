"""
Features: Imaging And Interferometer
====================================

Imaging and interferometer observations can be fitted together using the parameters that their physical models
share. Each dataset still requires the forward calculation appropriate to its measurement.

__Two Measurements of a Shared System__

Imaging compares a PSF-blurred model with image pixels. Interferometry compares a Fourier-domain prediction
with visibilities. A displayed interferometer image is useful for inspection, but does not replace the
visibility likelihood used by these examples.

Shared geometry does not require identical emission. Different wavelengths may trace different components of a
galaxy, so decide which parameters should be linked and which should remain dataset-specific.

__Reading the Model__

Follow the construction of each dataset's analysis and then their combination. The example retains its actual
shared parameters, light components and instrumental settings. Inspect the fit to each dataset separately: an
acceptable joint score can conceal a poor description of one observation.

__In This Folder__

These examples connect several observations. Read which parameters are shared, which are dataset-specific and
whether the fits are simultaneous or sequential. The per-dataset likelihoods still retain their own response
and noise treatment.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

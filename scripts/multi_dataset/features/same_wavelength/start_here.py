"""
Features: Same Wavelength
=========================

Repeated observations at the same wavelength can constrain a common light model while retaining their
individual instrumental responses and noise properties.

__One Model, Several Forward Calculations__

The observations need not have the same PSF, sampling or signal-to-noise. Each analysis predicts what the
common model would look like through its own measurement process, rather than treating all images as identical
copies.

Whether additional quantities such as normalizations or offsets should be shared depends on the dataset and
the model. The example's construction shows its actual choices.

__Inspect Every Observation__

The combined likelihood brings the measurements together, but residuals should still be examined for each
observation. An instrumental mismatch in one exposure may otherwise be hidden by a strong fit to another.

Use the simulator to see the differences introduced between the observations, then follow how those
differences are handled in the modeling script.

__In This Folder__

These examples connect several observations. Read which parameters are shared, which are dataset-specific and
whether the fits are simultaneous or sequential. The per-dataset likelihoods still retain their own response
and noise treatment.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

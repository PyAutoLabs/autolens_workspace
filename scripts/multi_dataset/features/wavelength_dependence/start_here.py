"""
Features: Wavelength Dependence
===============================

A wavelength-dependent model describes how specified galaxy properties change across bands. It links
observations through a chosen relationship rather than treating every parameter in every band independently.

__A Relationship Is a Model Assumption__

A parameter can be shared across all wavelengths, fitted separately in each band, or expressed through a
function of wavelength. These are different models with different amounts of freedom.

A smooth relationship may reduce the number of free parameters and connect weakly constrained bands, but it
also restricts the changes the model can describe. The observations must still support the adopted form.

__Follow the Linked Parameters__

The example explains which properties vary, how their wavelength dependence is expressed and which
coefficients are fitted. Retain these details beside the code; “multi-wavelength” does not by itself specify a
unique linking scheme.

Inspect the predicted morphology and residuals in every band. Differences between bands may reflect the light
distribution, instrumental response or an inadequate relationship, and the fit should be interpreted
accordingly.

__In This Folder__

These examples connect several observations. Read which parameters are shared, which are dataset-specific and
whether the fits are simultaneous or sequential. The per-dataset likelihoods still retain their own response
and noise treatment.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

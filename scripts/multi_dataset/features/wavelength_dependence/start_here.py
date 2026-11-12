"""
Features: Wavelength Dependence
===============================

Measure how galaxy and source morphology changes with wavelength by linking selected parameters through a specified relation. This can stabilize weak bands while keeping the scientific assumption about colour-dependent structure explicit.

__Scientific Use Cases__

- Study wavelength-dependent sizes or other chosen morphology parameters.
- Connect multi-band measurements without requiring all bands to look identical.
- Compare a smooth wavelength relation against more independent band-by-band descriptions.

__This Dataset and Lens Regime__

The parametric example links lens and source effective radii to wavelength through fitted straight-line
relations. This is a chosen morphology model, not a law that galaxy sizes must obey. The source-mesh discussion
also distinguishes shared geometry from band-dependent brightness: using a common mesh does not require
identical source colours or identical intensities in every band.

![wavelength dependence](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

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


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

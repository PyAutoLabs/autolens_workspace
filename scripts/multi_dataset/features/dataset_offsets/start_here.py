"""
Features: Dataset Offsets
=========================

Fit relative astrometric shifts when several observations of the same lens are not perfectly registered. This helps prevent a small alignment mismatch from being interpreted as a change in source structure or lens geometry.

__Scientific Use Cases__

- Combine observations taken with different instruments or at different epochs.
- Test whether apparent inter-band source displacements persist after accounting for alignment uncertainty.
- Retain per-dataset coordinate corrections while fitting a shared physical model.

__This Dataset and Lens Regime__

Decide explicitly which parameters are shared, which vary by dataset and whether the fits are simultaneous
or sequential. Each dataset retains its own response and uncertainty treatment. Inspect individual residuals
alongside the combined result; sharing a lens does not require identical emission in every observation.

![dataset offsets](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

__Relative Alignment__

An offset describes how one dataset is placed relative to a reference. The reference choice is part of the
model; otherwise a common shift can be confused with moving the object itself.

Offsets should be distinguished from differences in PSF, pixel scale or morphology between observations.
Correcting one translation does not account for every possible discrepancy between datasets.

__Following the Example__

The simulator establishes a known offset and the modeling example shows which dataset provides the reference
and which offsets are fitted. Retain the coordinate order, units and priors alongside the code. After fitting,
inspect each dataset in its own coordinates as well as considering the joint result.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

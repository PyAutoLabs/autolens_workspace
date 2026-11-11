"""
Features: Dataset Offsets
=========================

Observations of the same object may have small relative coordinate offsets. Joint modeling can account for
those offsets while fitting the physical model shared by the datasets.

__Relative Alignment__

An offset describes how one dataset is placed relative to a reference. The reference choice is part of the
model; otherwise a common shift can be confused with moving the object itself.

Offsets should be distinguished from differences in PSF, pixel scale or morphology between observations.
Correcting one translation does not account for every possible discrepancy between datasets.

__Following the Example__

The simulator establishes a known offset and the modeling example shows which dataset provides the reference
and which offsets are fitted. Retain the coordinate order, units and priors alongside the code. After fitting,
inspect each dataset in its own coordinates as well as considering the joint result.

__In This Folder__

These examples connect several observations. Read which parameters are shared, which are dataset-specific and
whether the fits are simultaneous or sequential. The per-dataset likelihoods still retain their own response
and noise treatment.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

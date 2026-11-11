"""
Features: Slam
==============

SLaM organizes lens modeling into linked source, light and mass stages. Results from earlier stages inform the
setup of later stages, allowing the model to be developed in a controlled sequence.

__Several Datasets Add Another Choice__

Multi-dataset workflows must decide which information is shared across observations and which light
descriptions remain independent. Fitting observations independently using a transferred mass model is
different from fitting them simultaneously with common parameters.

The independent and simultaneous examples retain those decisions in their code. Read which results are passed
between stages, what is fixed and what is refitted before comparing the outputs.

For the general pipeline structure, first read the [canonical SLaM guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/modeling/slam_start_here.ipynb). This folder explains how
that workflow is adapted to multiple datasets rather than repeating every stage's implementation.

__In This Folder__

These examples connect several observations. Read which parameters are shared, which are dataset-specific and
whether the fits are simultaneous or sequential. The per-dataset likelihoods still retain their own response
and noise treatment.

__Choose An Example__

- [independent.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/slam/independent.ipynb) — Fit datasets with the independent pipeline.
- [simultaneous.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/slam/simultaneous.ipynb) — Fit datasets with the simultaneous pipeline.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

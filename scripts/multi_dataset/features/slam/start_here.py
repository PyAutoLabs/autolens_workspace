"""
Features: SLaM
==============

Organize multi-dataset lens modeling into source, foreground-light and mass stages. Use the workflow to build a tractable analysis while choosing whether datasets inform each other through a joint fit or through transferred earlier results.

__Scientific Use Cases__

- Construct staged models for multi-band source science.
- Compare independent and simultaneous treatments of the observations.
- Track which mass and light constraints are carried forward, fixed or refitted at each stage.

__This Dataset and Lens Regime__

Decide explicitly which parameters are shared, which vary by dataset and whether the fits are simultaneous
or sequential. Each dataset retains its own response and uncertainty treatment. Inspect individual residuals
alongside the combined result; sharing a lens does not require identical emission in every observation.

![slam](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

__Several Datasets Add Another Choice__

Multi-dataset workflows must decide which information is shared across observations and which light
descriptions remain independent. Fitting observations independently using a transferred mass model is
different from fitting them simultaneously with common parameters.

The independent and simultaneous examples retain those decisions in their code. Read which results are passed
between stages, what is fixed and what is refitted before comparing the outputs.

For the general pipeline structure, first read the [canonical SLaM guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/modeling/slam_start_here.ipynb). This folder explains how
that workflow is adapted to multiple datasets rather than repeating every stage's implementation.


__Choose An Example__

- [independent.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/slam/independent.ipynb) — Fit datasets with the independent pipeline.
- [simultaneous.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/slam/simultaneous.ipynb) — Fit datasets with the simultaneous pipeline.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

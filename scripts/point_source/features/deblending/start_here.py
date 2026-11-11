"""
Features: Deblending
====================

Compact lensed images can overlap the foreground galaxy or other extended light. Deblending fits their
contributions so the point-image positions and other relevant quantities can be measured.

__Image Fitting Comes Before the Point Dataset__

A point-source lens analysis uses measured observables; it does not automatically solve the earlier problem of
locating compact images in a blended image. The deblending model supplies that measurement step.

The compact-profile description, instrumental response and extended-light model all affect the separation. A
satisfactory combined image does not guarantee that the measured position of each compact component is
unbiased.

__Using the Examples__

The simulator makes the components visible in a controlled setting. The modeling example shows their joint fit
and the quantities passed to the subsequent point-source analysis. Keep the image labels and uncertainty
treatment consistent across that handoff.

__In This Folder__

These examples use point-source observables and their associated image identities. The relevant script states
which positions, fluxes or time delays enter the fit; do not assume every observable is used in every example.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/deblending/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/deblending/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

"""
Features: Multiple Sources
==========================

Several background point sources can be lensed by the same foreground system. Their multiple images must be
associated with the correct source, and sources at different redshifts require the corresponding lensing
geometry.

__Image Families__

Images belonging to one source form a family. The model predicts that family's positions from its source and
the foreground deflectors. Combining several families adds constraints, but only if the associations and
redshifts are represented correctly.

A galaxy on an intermediate plane can also contribute mass that deflects a more distant source. The example
retains the actual plane assignments and mass components beside the model construction.

__Follow the Simulation into the Fit__

Start by inspecting the simulated image families. Then trace how they become point datasets and how their
analyses share the foreground model. Keep family labels, coordinates and uncertainties explicit when replacing
the simulated inputs with measurements.

__In This Folder__

These examples use point-source observables and their associated image identities. The relevant script states
which positions, fluxes or time delays enter the fit; do not assume every observable is used in every example.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/multiple_sources/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/multiple_sources/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

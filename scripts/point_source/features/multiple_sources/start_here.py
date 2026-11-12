"""
Features: Multiple Sources
==========================

Fit several families of compact lensed images through a shared foreground mass model. Source identities and redshifts determine which images belong together and which deflecting planes their light crosses.

__Scientific Use Cases__

- Use multiple compact background sources to constrain the same foreground lens.
- Represent image families at different redshifts with the appropriate multi-plane geometry.
- Include deflection by a nearer source galaxy when modeling a more distant source.

__This Dataset and Lens Regime__

The example assigns a separate point dataset to each source family. Its foreground lens is at redshift 0.5;
the nearer source at redshift 1.0 also carries mass and deflects the farther source at redshift 2.0.
These values define the demonstration, not a general restriction on multiple-source models.

__Example Status__

The simulator and modeling scripts remain listed in `config/build/no_run.yaml`. Their comments cite
[PyAutoLens #480](https://github.com/PyAutoLabs/PyAutoLens/issues/480), which is now closed. That stale exclusion
is not proof that the current example fails, but this documentation review has not validated it end to end.
Read it as a worked model construction pending a separate execution check; do not infer runnable status from
the resolved issue alone.

![multiple sources](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

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


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/multiple_sources/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/multiple_sources/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

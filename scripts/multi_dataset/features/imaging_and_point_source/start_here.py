"""
Features: Imaging And Point Source
==================================

Combine extended lensed emission with compact-source measurements to constrain a common lens mass model. Arcs, point-image positions and available time delays supply different information about the same system.

__Scientific Use Cases__

- Analyse a lensed compact source together with the extended host emission.
- Test whether one mass model explains both resolved arcs and compact-image observables.
- Build joint models for time-delay science when delays are supplied, keeping the delay reference and cosmological assumptions explicit.

__This Dataset and Lens Regime__

The supplied example combines RXJ1131 imaging with a point dataset containing time delays. It shares the
mass model while constructing the extended-light and point-source descriptions separately. Inferences about
cosmology require explicit choices for distances, line-of-sight effects and other relevant systematics; fitting
this example alone is not a complete cosmographic analysis.

![imaging and point source](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

__Keep the Observables Distinct__

An arc image contains surface-brightness information. Point-source data can contain image positions and time
delays, with their own uncertainties and conventions. They should not be treated as interchangeable
measurements simply because they belong to the same lens.

The modeling example shows how the analyses share the lens mass and how their other components are specified.
Keep image identities, coordinate conventions and the time-delay reference explicit when adapting it.

Inspect both the extended-light fit and the point-source predictions. Agreement with one part of the data does
not guarantee that the shared model describes the other.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_point_source/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

"""
Features: Imaging And Point Source
==================================

Extended arcs and point-source measurements provide different constraints on a shared lens model. A joint
analysis connects them through the mass distribution while retaining the appropriate likelihood for each
observable.

__Keep the Observables Distinct__

An arc image contains surface-brightness information. Point-source data can contain image positions and time
delays, with their own uncertainties and conventions. They should not be treated as interchangeable
measurements simply because they belong to the same lens.

The modeling example shows how the analyses share the lens mass and how their other components are specified.
Keep image identities, coordinate conventions and the time-delay reference explicit when adapting it.

Inspect both the extended-light fit and the point-source predictions. Agreement with one part of the data does
not guarantee that the shared model describes the other.

__In This Folder__

These examples connect several observations. Read which parameters are shared, which are dataset-specific and
whether the fits are simultaneous or sequential. The per-dataset likelihoods still retain their own response
and noise treatment.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_point_source/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

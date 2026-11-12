"""
Features: Group Halo
====================

Test whether a group-scale lens requires a common mass component in addition to its member galaxies. The aim is to distinguish alternative allocations of the total lensing mass, not to add a halo simply because the system is called a group.

__Scientific Use Cases__

- Explain image configurations probing mass across the group rather than around one member.
- Compare member-only and member-plus-halo models using the same data and a justified model-comparison procedure.
- Assess degeneracy between the group-scale component and individual galaxy masses.

__This Dataset and Lens Regime__

The neighboring modeling script compares a member-only composition with a member-plus-group-halo
composition. Its simulated demonstration includes the common halo; that controlled result is not evidence
that every observed group requires one. Arcs probing different radii and additional observations can help
assess how well member and common-halo contributions are distinguished.

![group halo](../../../../images/features/group_halo.png)

Conceptual illustration. Dashed outlines distinguish member and common mass components; they are neither visible dark matter nor measured halo boundaries.

__Individual Galaxies and Shared Mass__

The galaxy components describe mass associated with particular members. A group halo adds a component
extending across the system. Both contribute to the total deflection, so their parameters may compensate for
one another when fitting the same arcs.

Adding a halo therefore changes more than the number of components: it changes how the model can allocate
mass. Inspect the inferred components and the constraints on them, not only the appearance of the total fit.

__Using This Example__

The simulator and modeling example show an explicit halo parameterization and its relationship to the member
galaxies. Their priors and fixed parameters are retained beside the code. Use the example to understand the
construction before deciding whether that mass description is appropriate for another group.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/group_halo/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/group_halo/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

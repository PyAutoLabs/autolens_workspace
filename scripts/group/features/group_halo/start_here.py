"""
Features: Group Halo
====================

A group-scale lens may be described by the mass of its individual galaxies, by those galaxies plus a shared
group halo, or by other suitable mass components. A shared halo is a modelling choice rather than a
requirement imposed by the name “group”.

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

__In This Folder__

The examples contain a group-scale lens with several galaxy components. A shared group halo, where present, is
an explicit model choice. Keep the individual deflectors, companion population and source components distinct
when following the recipe.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/group_halo/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/group_halo/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

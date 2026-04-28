No Lens Light (Group)
====================

These scripts show how to model a group-scale strong lens where none of the lens galaxies
have visible light emission. In the group context, this means **all** main lens galaxies
and **all** extra galaxies are modeled with mass profiles only — no light profiles.

This dramatically reduces the dimensionality of the model compared to a standard group fit.
For a group with one main lens and two extra galaxies, omitting the MGE light model removes
all light-related non-linear parameters, leaving only the mass and source parameters.

Files
-----

- ``simulator.py``: Simulate a group lens dataset with no lens galaxy light.
- ``modeling.py``: Fit a group lens model where all galaxies have mass only.
- ``slam.py``: SLaM pipeline for group lenses without lens light (skips light-fitting stages).

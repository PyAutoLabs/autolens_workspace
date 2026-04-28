Linear Light Profiles (Group)
============================

These scripts show how to model a group-scale strong lens using **linear light profiles**, where the
``intensity`` of each light profile is solved analytically via linear algebra rather than being a
free parameter in the non-linear search.

For a group-scale lens, this is especially beneficial: the group model may contain many galaxies
(main lenses and extra galaxies), and each galaxy's light profile would normally add an ``intensity``
parameter. By using linear light profiles, none of these contribute to the non-linear parameter space.

Files
-----

- ``modeling.py``: Fit a group lens model using linear ``Sersic`` light profiles for all galaxies.
- ``fit.py``: Demonstrate how to fit data and extract solved-for intensities.
- ``likelihood_function.py``: Step-by-step guide to the log likelihood function with linear profiles.
- ``slam.py``: Full SLaM pipeline for group lenses using linear Sersic profiles instead of MGE.

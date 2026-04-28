Operated Light Profiles (Group)
===============================

This folder contains scripts demonstrating operated (PSF-convolved) light profiles for group-scale strong lens
modeling. Operated light profiles represent light that has already been convolved with the PSF, so they are NOT
convolved again during model-fitting.

Files
-----

- ``modeling.py``: Fits a group-scale lens model where main lens galaxies include operated light profile
  components (e.g. for AGN emission) and extra galaxies can also use operated profiles.
- ``simulator.py``: Simulates a group-scale lens dataset using operated light profiles for the lens and
  extra galaxy light.

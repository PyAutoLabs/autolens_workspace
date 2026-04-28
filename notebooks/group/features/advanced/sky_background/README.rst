Sky Background (Group)
======================

This folder contains scripts demonstrating sky background modeling for group-scale strong lenses. The sky
background is modeled as an additional free parameter using a ``DatasetModel`` object.

Files
-----

- ``modeling.py``: Fits a group-scale lens model that includes a sky background as a free parameter,
  ensuring uncertainties on galaxy light profiles account for sky subtraction errors.
- ``fit.py``: Demonstrates how to include sky background subtraction in a fit for group data using the
  ``DatasetModel`` API.
- ``simulator.py``: Simulates a group-scale lens dataset with an elevated sky background that is not
  subtracted from the image.

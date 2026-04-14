Shapelets (Group)
=================

This folder contains scripts demonstrating shapelet basis function decomposition for group-scale strong lens
modeling. Shapelets capture complex morphological features of galaxies that standard light profiles cannot.

Files
-----

- ``modeling.py``: Fits a group-scale lens model where the source galaxy is decomposed into ~20 shapelet basis
  functions. Main lens and extra galaxies use MGE light profiles.
- ``fit.py``: Demonstrates how to create a fit using shapelet light profiles for the source galaxy in a group
  lens, without a non-linear search.

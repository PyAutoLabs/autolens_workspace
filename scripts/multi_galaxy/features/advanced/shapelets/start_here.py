"""
Features: Shapelets
===================

Shapelets describe light as a weighted combination of basis functions. Unlike a single smooth light profile,
the basis contains functions with different spatial structures, allowing their sum to describe more
complicated galaxy morphology.

__Basis Functions and Their Sum__

An individual basis function is a mathematical building block, not a separate physical component of the
galaxy. Its appearance should therefore not be interpreted as the appearance of a star-forming region, disk or
other structure by itself. The combined light distribution is the quantity to compare with the observations.

The basis order and scale determine the range of structures that can be represented. Increasing the number of
functions adds flexibility and computational work; it does not by itself establish that the additional
structure is constrained by the data.

__What the Examples Teach__

The direct-fit example introduces the basis and its coefficients. The modeling example shows how the basis is
embedded in the wider model and which parameters are varied. Keep the basis definition, coefficient
constraints and any regularization in view when interpreting the result.

As with other flexible light models, inspect the residuals and the reconstructed image together. A useful
basis should describe the signal without introducing structure solely to follow noise.

__In This Folder__

The examples contain multiple main lens galaxies, rather than a single dominant deflector with only small
companions. Follow how the feature is applied to each galaxy and which parameters are shared. A group halo is
not implicit in this setup.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/shapelets/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_galaxy/features/advanced/shapelets/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

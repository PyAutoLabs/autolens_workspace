"""
Features: Shapelets
===================

Describe structured source or galaxy light with a flexible basis of shapelets. This offers an alternative to a small set of smooth profiles when morphology matters to the lens fit and to the science inferred from the source.

__Scientific Use Cases__

- Represent departures from smooth, single-component light profiles.
- Compare source structure and lens inference across different light-model families.
- Explore how basis order and spatial scale affect the recovered morphology and residuals.

__This Dataset and Lens Regime__

This folder treats group-scale imaging with individual member galaxies. Distinguish the main deflectors,
additional members and any explicit common group halo. The light decomposition and the split of mass among
components can affect the inferred source; a good combined fit does not guarantee that each component is
independently constrained.

![shapelets](../../../../../images/features/shapelets_illustration.png)

Conceptual illustration. The basis represents a combined light distribution; individual functions need not correspond to separate physical components. The schematic illustrates the feature; retain all deflectors in this group example.

__Basis Functions and Their Sum__

An individual basis function is a mathematical building block, not a separate physical component of the
galaxy. Its appearance should therefore not be interpreted as the appearance of a star-forming region, disk or
other structure by itself. The combined light distribution is the quantity to compare with the observations.

The basis order and scale determine the range of structures that can be represented. Increasing the number of
functions adds flexibility and computational work; it does not by itself establish that the additional
structure is constrained by the data.

![Individual shapelet basis functions](../../../../../images/features/shapelets_basis.png)

The existing basis grid is retained as a teaching figure; the fitted light is the sum of the functions.

__What the Examples Teach__

The direct-fit example introduces the basis and its coefficients. The modeling example shows how the basis is
embedded in the wider model and which parameters are varied. Keep the basis definition, coefficient
constraints and any regularization in view when interpreting the result.

As with other flexible light models, inspect the residuals and the reconstructed image together. A useful
basis should describe the signal without introducing structure solely to follow noise.


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/shapelets/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/shapelets/modeling.ipynb) — Fit the model with a non-linear search.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

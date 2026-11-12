"""
Features: Deblending
====================

Measure the positions and fluxes of compact lensed images that overlap foreground galaxy light. This is an image-fitting step that prepares the observables for subsequent point-source lens modeling.

__Scientific Use Cases__

- Measure lensed quasar or supernova image positions with an explicit PSF and foreground-light model.
- Separate compact-image flux from extended galaxy emission.
- Assess how measurement uncertainties and blending affect the point-source lens analysis that follows.

__This Dataset and Lens Regime__

Although this feature lives under point_source, its measurement step fits an Imaging dataset. The compact
profiles and the extended foreground model jointly determine image centres and fluxes. A subsequent
PointDataset is a summary of those measurements; preserve their labels and uncertainty treatment through
that handoff. Estimating a centre more precisely than a pixel does not mean resolving intrinsic source structure.

![deblending](../../../../images/features/deblending.png)

Conceptual illustration. The markers indicate fitted image centres; the illustration does not claim to remove the PSF or improve angular resolution.

__Image Fitting Comes Before the Point Dataset__

A point-source lens analysis uses measured observables; it does not automatically solve the earlier problem of
locating compact images in a blended image. The deblending model supplies that measurement step.

The compact-profile description, instrumental response and extended-light model all affect the separation. A
satisfactory combined image does not guarantee that the measured position of each compact component is
unbiased.

__Using the Examples__

The simulator makes the components visible in a controlled setting. The modeling example shows their joint fit
and the quantities passed to the subsequent point-source analysis. Keep the image labels and uncertainty
treatment consistent across that handoff.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/deblending/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/point_source/features/deblending/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

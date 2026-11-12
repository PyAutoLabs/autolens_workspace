"""
Features: Pixelization
======================

Reconstruct the background galaxy on a source-plane mesh when simple light profiles do not capture its morphology. Pixelization is a key source-science feature: it supports measurements of irregular emission while the lens model maps that emission into the observed data.

__Scientific Use Cases__

- Reconstruct extended or clumpy source morphology without specifying a small set of analytic components.
- Measure source brightness and spatial structure while accounting for lensing and the instrumental response.
- Test lens mass models using the detailed structure of the lensed emission, with mesh and regularization choices kept explicit.

For source-science measurements, continue to [source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/source_science.ipynb).
Interpret the combined reconstructed light together with the lens and reconstruction assumptions.

__This Dataset and Lens Regime__

A shared source mesh supplies comparable spatial coordinates across datasets; its brightness reconstruction
can still differ between them. This is useful for source morphology and colour comparisons, provided the
shared lens geometry, relative alignment and different instrumental responses are modeled consistently.
Avoid interpreting different apparent detail in two reconstructions as a physical difference before checking
their resolution and regularization.

![pixelization](../../../../images/features/pixelization_illustration.png)

Conceptual illustration. The mesh and source are illustrative; recoverable structure depends on the lens model, response, sampling and regularization.

__From a Mesh to the Data__

The reconstructed light must be mapped into the observable being fitted. Imaging examples include the PSF;
interferometer examples compare the prediction with visibilities through the Fourier response. In lensing, ray
tracing also maps the source mesh into the image plane.

A reconstructed pixel is therefore not simply copied from a data pixel. Its value is inferred through the
forward model, and neighbouring reconstructed pixels may be constrained by overlapping parts of the data.

__Why Regularization Is Needed__

A flexible mesh can describe structure, but it can also follow noise. Regularization expresses a preference
for relationships between reconstructed pixels, such as a smoother distribution of light. The regularization
strength controls the balance between flexibility and that preference.

A reconstruction with a visually smooth appearance is not automatically the right one. Too much regularization
can suppress real structure; too little can introduce features driven by noise. The mesh, regularization and
residuals should be considered together.

__Choosing the Mesh__

The number and placement of mesh pixels affect both the structures the model can describe and the
computational cost. Adaptive meshes concentrate resolution where it is useful, while different mesh
constructions imply different interpolation and regularization choices.

The examples retain these choices alongside the code. Read the fixed-model fit first, then the modeling
example; specialist examples explore other meshes, adaptive calculations or measurements from the
reconstruction.

__Interpreting the Reconstruction__

Check the fit in the space where the data were measured as well as inspecting the reconstructed light. A
plausible reconstruction can still accompany an inadequate PSF, an unsuitable mask or an incomplete physical
model. Fine structure should be interpreted in light of the resolution and assumptions of the full analysis.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/pixelization/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/pixelization/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

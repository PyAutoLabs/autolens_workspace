"""
Features: Pixelization
======================

Reconstruct the background galaxy on a source-plane mesh when simple light profiles do not capture its morphology. Pixelization is a key source-science feature: it supports measurements of irregular emission while the lens model maps that emission into the observed data.

__Scientific Use Cases__

- Reconstruct extended or clumpy source morphology without specifying a small set of analytic components.
- Measure source brightness and spatial structure while accounting for lensing and the instrumental response.
- Test lens mass models using the detailed structure of the lensed emission, with mesh and regularization choices kept explicit.

For source-science measurements, continue to [source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/source_science.ipynb).
Interpret the combined reconstructed light together with the lens and reconstruction assumptions.

__This Dataset and Lens Regime__

These examples fit complex visibilities: the lensed sky prediction is Fourier transformed and compared at
the measured baseline coordinates with the supplied noise. A dirty or reconstructed image is useful for
inspection, but does not replace this likelihood. Interpret source structure in the context of visibility
coverage and the assumptions used to model the sky.

![pixelization](../../../../images/features/pixelization_illustration.png)

Conceptual illustration. The mesh and source are illustrative; recoverable structure depends on the lens model, response, sampling and regularization. The scene illustrates sky/source structure; the fit here is to visibilities, not the displayed image.

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


__Disadvantages__

Lens modeling with pixelizations is conceptually more complex. There are additional failure modes, such as
solutions where the source is reconstructed in a highly demagnified configuration due to an unphysical lens mass
model (e.g. too little or too much mass). These issues are discussed in detail later in the workspace.

As a result, learning to successfully fit lens models with pixelizations typically requires more time and experience
than the simpler modeling approaches introduced elsewhere in the workspace.

__Positive Only Solver__

The supplied fit and modeling examples explicitly disable the positive-only source solver. This is a solver
choice, not a consequence of negative visibility components: real and imaginary Fourier components can have
either sign even when the sky brightness is everywhere non-negative.

An unconstrained inversion can return negative source pixels. Inspect their significance and the effects of
noise, regularization and incomplete sampling. Where a non-negative solve is appropriate and supported by the
chosen calculation, compare its behavior; a positivity constraint alone does not guarantee a correct lens or source model.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/likelihood_function.ipynb) — Follow the likelihood calculation.
- [delaunay.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/delaunay.ipynb) — Use a Delaunay mesh.
- [many_visibilities_preparation.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/many_visibilities_preparation.ipynb) — Prepare a large visibility dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/slam.ipynb) — Follow the staged SLaM fit.
- [source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/pixelization/source_science.ipynb) — Measure quantities from the reconstructed source.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

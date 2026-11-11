"""
Features: Pixelization
======================

A pixelization represents light using a mesh whose brightness values are solved during the fit. It provides a
flexible alternative to describing all of the emission with a small collection of analytic profiles.

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

__In This Folder__

These examples fit visibilities. The image-plane light prediction is transformed into the measured
Fourier-domain quantities; a dirty image is an inspection product, not a substitute for that likelihood. Keep
the visibility sampling, noise and model-specific response choices in view when adapting an imaging example.

__Disadvantages__

Lens modeling with pixelizations is conceptually more complex. There are additional failure modes, such as
solutions where the source is reconstructed in a highly demagnified configuration due to an unphysical lens mass
model (e.g. too little or too much mass). These issues are discussed in detail later in the workspace.

As a result, learning to successfully fit lens models with pixelizations typically requires more time and experience
than the simpler modeling approaches introduced elsewhere in the workspace.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This could be problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelizaiton, this often produces negative source pixels which over-fit
the data, producing unphysical solutions.

For CCD imaging datsets pixelized source reconstructions use a positive-only solver, meaning that every source-pixel
is only allowed to reconstruct positive flux values. This ensures that the source reconstruction is physical and
that we don't reconstruct negative flux values that don't exist in the real source galaxy (a common systematic
solution in lens analysis).

However, for interferometer datasets this positive-only solver is often disabled, because negative pixel values
can be observed from the measurement process. All interferometer examples therefore disable the positive only solver,
but you may want to consider if using the positive-only solver is appropriate for your dataset.

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

"""
Features: Pixelization
======================

Strong gravitational lensing stretches a background galaxy into arcs and multiple images. A pixelized source
reconstruction uses those observations, together with a lens mass model, to infer what the galaxy looks like
before lensing. It is particularly useful when the source has spiral arms, star-forming clumps or irregular
structure that a simple smooth light profile struggles to describe.

**Study the background galaxy.** Pixelized reconstructions are a key tool for source science: investigating
galaxy structure and measuring properties such as intrinsic flux, size and lensing magnification. Start with
[source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/source_science.ipynb)
to see how to analyse the reconstructed source, calculate flux and magnification, propagate uncertainties and
export the reconstruction.

Use a pixelization when you want to:

- **Reconstruct the structure of a distant galaxy.** Describe extended, asymmetric or clumpy emission without
  prescribing a Sersic profile or a common-centre Gaussian model. A suitable mesh can also describe spatially
  separated components; sources at different redshifts require the appropriate separate source planes.
- **Fit the lens mass with a flexible source model.** Reconstruct the source while testing lens mass models,
  so an unnecessarily restrictive source shape does not have to account for the observed arcs. Flexibility
  does not remove lens-model degeneracies or guarantee that the inferred mass is correct.
- **Measure the source's properties.** Use the reconstructed brightness to estimate source flux, size and
  magnification. The `source_science.py` example explains flux and magnification calculations, uncertainties
  and exporting the reconstruction for further analysis.

![Pixelization illustration: recover the galaxy behind the arcs](../../../../images/features/pixelization_illustration.png)

Conceptual illustration, not observational data or a fitted reconstruction. The mesh and the galaxy on the
right represent the same source brightness distribution; removing the mesh overlay does not add information.
The spiral is one possible source morphology, and the detail recoverable from real data depends on the lensing
geometry, instrumental resolution, noise and modelling assumptions.

__A Scientific Example__

[Nightingale, Dye & Massey (2018), AutoLens: automated modeling of a strong lens's light, mass, and
source](https://doi.org/10.1093/mnras/sty1264) describes simultaneous lens-light and mass modelling with an
adaptive pixelized source reconstruction, tested on simulated strong-lens images. It provides a scientific
starting point for understanding how the source reconstruction and lens model are inferred together.

__How the Source Is Reconstructed__

A pixelization represents light using a mesh whose brightness values are solved during the fit. It provides a
flexible alternative to describing all of the emission with a small collection of analytic profiles.

For a trial lens mass model, the source mesh is mapped through gravitational lensing and the instrumental
response to predict the data. A regularized linear inversion solves for the source brightness values. The
non-linear search explores the model's remaining parameters, including the lens mass and, where free, the
regularization parameters. The source is therefore reconstructed as part of lens modelling, rather than by
applying a fixed image-unwarping filter after the fit.

A smooth source may be adequately described by a Sersic profile or an MGE. Pixelization offers greater freedom
for complex morphology, at the cost of choosing a mesh and regularization and solving a larger reconstruction
problem. The following sections explain those choices and how to judge the result.

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

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
foreground light, deflecting mass and background source play different roles; retain the particular component
choices stated in each recipe.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/likelihood_function.ipynb) — Follow the likelihood calculation.
- [adaptive.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/adaptive.ipynb) — Explore adaptive reconstruction choices.
- [cpu_fast_modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/cpu_fast_modeling.ipynb) — Configure the CPU calculation.
- [delaunay.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/delaunay.ipynb) — Use a Delaunay mesh.
- [plot.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/plot.ipynb) — Inspect the available visualizations.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/slam.ipynb) — Follow the staged SLaM fit.
- [source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/source_science.ipynb) — Measure quantities from the reconstructed source.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

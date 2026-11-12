"""
Features: No Lens Light
=======================

Model the lens mass and reconstruct the background source when foreground lens light is negligible in your
image, or has already been subtracted. This feature keeps the gravitational lens model while omitting its
foreground light component, so the fit concentrates on the lensed source emission.

__Scientific Use Cases__

- **Foreground-faint observations:** fit arcs in a band where the foreground galaxy contributes little light
  compared with the source and the measurement uncertainties.
- **Lens-light-subtracted imaging:** infer the lens mass and source from an image prepared using a separate
  foreground-light model, while assessing how the subtraction affects the inference.
- **Source science:** reconstruct source morphology and brightness when foreground emission can be omitted.
  The [source science guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/source_science.ipynb)
  explains how fitted lens models connect to measurements of the background galaxy.

![Lens-light-free imaging with the deflecting mass retained](../../../../images/features/no_lens_light.png)

Conceptual illustration: both panels contain lensed source light. The dashed contours on the right represent
an assumed mass model, not visible emission or a measured mass map. Omitting lens light does not remove gravity.

__Light and Mass Are Different Parts of the Model__

The foreground galaxy's light describes photons it emits; its mass describes how it deflects photons from
background sources. These are separate model components. A foreground galaxy can be faint in the observed
band and still produce prominent arcs or an Einstein ring.

The modeling example retains an isothermal lens mass and external shear, and describes the source with a
multi-Gaussian expansion. The foreground lens has no light profile. The data are still imaging data: the
model must account for the PSF, pixel sampling, mask and noise when predicting the observed arcs.

__Absent Light and Subtracted Light Are Different Cases__

The simulator generates a lens without foreground emission. This isolates the mass and source modeling
problem; it does not simulate the errors introduced by subtracting a bright foreground galaxy from real data.

In observed images, foreground and source light can overlap, particularly after PSF convolution. A separate
foreground fit can remove some source flux or leave foreground residuals. The subsequent source reconstruction
may absorb that structure, changing the inferred source morphology, brightness or lens mass.

Inspect the subtraction around the arcs and test plausible alternative foreground models. Where the two
components are strongly blended, simultaneous lens-light and source fitting lets their allocations adjust
together. Supplying a subtracted image alone does not automatically propagate uncertainty in the subtraction.

__A Smaller Model, With a Scientific Assumption__

Omitting foreground light removes its parameters from the fit. The reduction depends on the light model that
would otherwise have been used; it is not a fixed number of parameters or a guaranteed speed-up.

The important assumption is that omitted foreground emission is sufficiently small, or sufficiently well
accounted for, in the fitted pixels. A visually faint residual can still matter near a high-signal arc.
Use residual maps and comparisons of the inferred lens and source properties to assess that assumption.

__Following the Examples__

Start with the simulator to see what emission and deflectors are included. The modeling example fits the
mass and source together; the SLaM example carries the absence of foreground light through a staged analysis.
Keep the distinction between the simulated demonstration and your own data preparation explicit.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/modeling.ipynb) — Fit the lens mass and source without a foreground light profile.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/simulator.ipynb) — Generate imaging with lens deflection but no foreground lens emission.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/no_lens_light/slam.ipynb) — Follow the staged SLaM fit without lens light.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

"""
Features: Double Source-Plane Lenses
===================================

A single foreground galaxy can lens two background galaxies at different redshifts. A double source-plane
lens lets us study the same foreground mass through two different lensing geometries, while reconstructing
the light of both background sources.

Use this feature to investigate:

- **Cosmic geometry:** the relative lensing strengths for the two sources depend on cosmological distance
  ratios, providing information about cosmological parameters when combined with an appropriate mass model
  and redshift measurements.
- **Galaxy mass structure:** the two sets of arcs can probe different regions of the foreground mass
  distribution, while the farther source can also constrain the mass of the nearer source galaxy.
- **Source science:** reconstruct two background galaxies at different cosmic epochs, accounting for the
  additional deflection of the farther source by the nearer one.

[Collett & Auger (2014)](https://arxiv.org/abs/1403.5278) demonstrate the cosmological application using
SDSS J0946+1006, the Jackpot lens. Their analysis illustrates the scientific opportunity; the basic modeling.py
tutorial in this folder uses a fixed cosmology and does not itself measure cosmological parameters.

![Two sets of lensed arcs and the three redshift planes that produce them](../../../../../images/features/double_source_plane_illustration.png)

The image is a conceptual illustration, not a ray-traced prediction. Cyan and magenta identify the two
sources; they do not represent measured colours or redshifts. The simulated examples produce a double ring,
but double source-plane lenses can also appear as incomplete arcs or multiple images.

__Keep the Planes Distinct__

The geometry contains a foreground lens, a nearer source and a farther source, ordered by redshift. Each
source's light is traced through the relevant preceding planes to predict the observed image.

The nearer source has two roles: its light is lensed by the foreground galaxy, and its mass can deflect the
farther source's light. A joint model therefore distinguishes the nearer galaxy's light profile from its
mass profile. The farther source is lensed by both preceding mass distributions.

The two sources do not share the same lensing geometry merely because their arcs appear in the same image.
Their redshifts and the adopted cosmology determine the angular-diameter distance ratios used in ray tracing.
Those ratios set how strongly each deflector acts on each source.

The model must identify each galaxy's plane and distinguish its light from its mass contribution. The
examples retain their particular redshifts and component choices beside the construction of the model.
Multiple deflectors in group or multi-galaxy examples add components within this geometry; preserve those
distinctions when adapting a single-deflector example.

__What Can the Data Tell Us?__

Two source planes provide information beyond a single source plane, but do not remove every degeneracy
between mass and geometry. Changes to the assumed mass distributions can mimic changes in cosmological
distance ratios. Reliable cosmological inference therefore requires attention to the mass model, source
redshifts and matter along the line of sight, with additional information such as stellar kinematics useful
for constraining the lens mass.

Treat the reconstruction of the two sources, the inference of the deflectors' masses, and cosmological
inference as related but distinct goals. A good image fit with fixed cosmology demonstrates a lens model;
it is not by itself a cosmological measurement.

__Follow the Geometry Before Fitting__

Start with the simulator or direct fit to see which arcs belong to which source and how the model assigns
light and mass to each plane. Then read modeling.py to follow the model construction and likelihood.

The modeling.py tutorial initializes priors close to the simulated truth. Those values are unavailable for
an observed lens, so use it to learn the composition of the model before moving to the staged examples.

The chaining.py example first fits the foreground lens and nearer source using a smaller mask that excludes
the farther source's light. It then expands the mask, adds the farther source, and allows the nearer source's
mass to contribute to the fit. This uses the fact that the nearer source's lensing does not depend on the
farther source. Check the observed image before adopting this masking strategy: the two sources may overlap.

The slam.py example develops staged fitting further, including pixelized reconstructions of the sources.
Read its stage-specific choices carefully: some components are fixed or omitted at intermediate stages.
Chaining and SLaM carry information between fits; they do not replace checking the plane assignments or
whether the chosen model can describe the data.

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF.
Foreground lens light, deflecting mass and the two background sources play different roles. Some recipes
omit foreground lens light in their simulations; retain the particular component choices stated in each
recipe and include foreground emission when the data require it.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/modeling.ipynb) — Learn model composition with priors near the simulated truth and fixed cosmology.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/simulator.ipynb) — Generate the example dataset.
- [chaining.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/chaining.ipynb) — Initialize the nearer-source fit before adding the farther source.
- [plot.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/plot.ipynb) — Inspect the available visualizations.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/double_source_plane_lens/slam.ipynb) — Follow staged fits including pixelized source reconstructions.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

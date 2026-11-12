"""
Features: Stellar + Dark Matter Mass Models
==========================================

How much of a lens galaxy's mass is in stars, and how much is dark matter? A decomposed mass model combines
the galaxy's observed light with the gravitational lensing of a background source to investigate that split.

Use this feature to infer the stellar mass-to-light ratio, estimate the stellar and dark matter masses within
a chosen aperture, and study how the two components are distributed. These measurements connect strong
lensing to questions about the structure and formation of massive galaxies.

[Nightingale et al. (2019)](https://arxiv.org/abs/1901.07801) provide an example: they used PyAutoLens to
combine light-profile decomposition with strong lensing, investigating the stellar structures and inner dark
matter halos of massive elliptical galaxies.

![Observed lens light and inferred stellar plus dark matter components](../../../../../images/features/stellar_dark_illustration.png)

The illustration shows the idea, not a measured decomposition. Lensing constrains the combined gravity of
the stars and dark matter; separating them requires assumptions about how each component is distributed.

__From Total Mass to Stars and Dark Matter__

A total-mass profile, such as an isothermal or power-law profile, describes the lens galaxy's combined mass.
A decomposed model instead adds a stellar mass component tied to the galaxy's light and a dark matter halo,
often described by an NFW profile. Both contribute to the deflection of the background source's light.

This brings information from the foreground galaxy into the mass model. The lens light is more than
emission to subtract: its spatial distribution provides a template for the stellar mass. Depending on the
chosen profiles, the stellar and dark components can have different shapes or orientations.

__From Light to Stellar Mass__

A stellar mass component uses a light distribution together with a mass-to-light ratio or related
normalization. For a constant mass-to-light ratio, increasing this ratio increases the stellar mass assigned
to the same light distribution.

The light model therefore becomes part of the mass-model assumptions: an inadequate light description can
affect the inferred stellar contribution. A single constant ratio, different ratios for separate light
components, and a ratio that varies with radius represent different physical assumptions.

The examples state how the stellar mass follows the light, which mass-to-light ratios are shared or free,
and how the halo is parameterized. Multi-galaxy and group examples make these decisions for more than one
deflector; preserve the component choices stated in the recipe you are adapting.

__What Can the Data Tell Us?__

Lensing responds to the combined deflection, so a satisfactory total model does not necessarily imply that
the stellar and dark contributions are individually well determined. A larger stellar contribution can
sometimes be compensated by a smaller halo contribution, or by changes to the halo's shape.

Interpret the decomposition together with its priors and assumptions. Compare plausible light models,
mass-to-light prescriptions and halo models before drawing conclusions about either component. State the
aperture when quoting masses or dark matter fractions: constraints within the region probed by the arcs do
not automatically determine the mass of the entire halo.

These models can also be more computationally expensive than a simple total-mass model. Establishing a
robust total-mass fit first provides a useful starting point and a comparison for the decomposition.

__From the Tutorial to a Staged Fit__

Start with the simulator or direct fit to inspect the dataset and understand the model components. Then
read modeling.py to see how the stellar and dark profiles are composed and fitted.

The modeling.py example uses priors initialized close to the simulated truth. This makes it useful for
learning the model construction, but those true values are unavailable for an observed lens.

For a staged approach, follow chaining.py: it first fits the lens light, then uses that information to
initialize the stellar component in a lensing fit with a dark halo and background source. Check whether
the foreground and source light can be separated adequately before adopting a light-first fit.

The slam.py example develops this approach through the Source, Light and Mass pipelines, ending with
MASS LIGHT DARK and a pixelized source reconstruction. The detailed choices of profiles, priors and
source representation remain beside the model construction in each script.

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF.
The foreground light, deflecting mass and background source play different roles: the observed arcs belong
to the background source, while this feature investigates the mass of the foreground lens galaxy.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/modeling.ipynb) — Learn model composition with priors close to the simulated truth.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/simulator.ipynb) — Generate the example dataset.
- [chaining.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/chaining.ipynb) — Initialize the stellar mass model from an earlier lens-light fit.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/mass_stellar_dark/slam.ipynb) — Follow the staged SLaM fit through to stellar + dark matter modelling.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

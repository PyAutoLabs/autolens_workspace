Features (Group)
================

The scripts in this folder illustrate lens modeling features for **group-scale** strong lenses, where
there are multiple main lens galaxies and extra companion galaxies whose combined mass produces the
lensing effect.

Each feature folder contains scripts adapted from the corresponding ``imaging/features`` examples,
with all models, descriptions, and API usage specific to the group-scale case.

Features
--------

Core features:

- ``linear_light_profiles``: All galaxy light profiles use linear algebra to solve for intensity.
- ``multi_gaussian_expansion``: Galaxy light modeled as ~10-30 Gaussian basis functions (MGE).
- ``no_lens_light``: All group galaxies have no visible light — mass-only models.
- ``pixelization``: Source galaxy reconstructed using an adaptive pixel mesh.

Advanced features:

- ``advanced/operated_light_profile``: PSF-convolved (operated) light profiles for group galaxies.
- ``advanced/shapelets``: Shapelet basis functions for group galaxy light.
- ``advanced/sky_background``: Modeling a uniform sky background alongside the group.
- ``advanced/subhalo``: Dark matter subhalo detection in group-scale lenses.

Group-Scale Considerations
--------------------------

Group-scale lenses differ from galaxy-scale lenses in several important ways that affect feature usage:

1. **More galaxies**: Multiple main lenses and extra galaxies all contribute light and mass.
2. **Higher dimensionality**: Features that reduce model complexity (MGE, linear profiles) are
   especially valuable for groups.
3. **Larger masks**: Group lenses use ~7.5" masks vs ~3.0" for galaxy-scale, meaning more pixels
   and higher computational cost.
4. **Multi-galaxy over-sampling**: Adaptive over-sampling must be applied at the centre of every
   galaxy in the group, not just one.

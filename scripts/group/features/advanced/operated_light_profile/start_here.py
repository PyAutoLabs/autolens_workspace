"""
Features: Operated Light Profile
================================

Separate compact emission from the extended light around it using a profile whose instrumental meaning is explicit. For imaging, an operated profile describes a component that already includes PSF blurring; this is separate from whether its intensity is solved linearly.

__Scientific Use Cases__

- Model a compact nuclear component alongside extended galaxy emission.
- Assess how allocating central light changes the foreground decomposition and inferred source.
- Keep intrinsic light, observed profile shape and instrumental response distinct when constructing a model.

__This Dataset and Lens Regime__

This folder treats group-scale imaging with individual member galaxies. Distinguish the main deflectors,
additional members and any explicit common group halo. The light decomposition and the split of mass among
components can affect the inferred source; a good combined fit does not guarantee that each component is
independently constrained.

![operated light profile](../../../../../images/features/operated_light_profile.png)

Conceptual illustration. The PSF illustration refers to imaging: an already blurred component is not convolved a second time. The schematic illustrates the feature; retain all deflectors in this group example.

__Avoid Applying the Same Response Twice__

Ordinary intrinsic light profiles are passed through the instrumental response before comparison with the
data. An operated profile is already defined at that stage for the relevant operation. Applying the same PSF
convolution again would broaden it a second time.

This distinction is about the meaning of the profile, not simply whether a source is small. Decide whether
your parameters describe intrinsic emission or its observed appearance before choosing the profile class.

__Compact Emission and Extended Light__

Compact nuclear emission can overlap an extended host galaxy. The examples show how those components are
combined and which instrumental response is already represented. Their decomposition depends on the adopted
compact-profile shape as well as the host model.

Read the simulator and modeling example together. Imaging and interferometer responses are different, so
retain the data-type-specific explanation in the recipe rather than applying an imaging PSF assumption to
every dataset.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/operated_light_profile/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/advanced/operated_light_profile/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

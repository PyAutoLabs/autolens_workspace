"""
Features: Operated Light Profile
================================

Model compact emission alongside extended light in an interferometer fit. The operated profile still contributes a sky brightness distribution that must be transformed into visibilities; the imaging meaning of an already PSF-blurred profile should not be transferred to a synthesized beam.

__Scientific Use Cases__

- Describe a compact emitting component without forcing it into the extended host profile.
- Assess how separating compact and extended emission changes the inferred lens and source.
- Use compatible light-model components across imaging and visibility analyses while preserving their different instrumental responses.

__This Dataset and Lens Regime__

For this visibility fit, the compact component is an image-plane emission profile that is Fourier
transformed like the other light profiles. The synthesized beam describes an image formed from visibilities;
it is not a second convolution to apply before fitting them. The operated class does not remove the need to
predict the measured visibilities, and its width should not be identified with an imaging PSF by default.

![operated light profile](../../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

__Avoid Applying the Same Response Twice__

In imaging, an operated profile bypasses a PSF convolution because its shape already describes that operation.
In this interferometer example there is no image-plane PSF convolution in the visibility prediction to bypass:
the component image is Fourier transformed at the measured baseline coordinates.

This distinction concerns the response represented by the model, not simply how small the component is.

__Compact Emission and Extended Light__

Compact nuclear emission can overlap an extended host galaxy. The examples show how those components are
combined and which instrumental response is already represented. Their decomposition depends on the adopted
compact-profile shape as well as the host model.

Read the simulator and modeling example together. Imaging and interferometer responses are different, so
retain the data-type-specific explanation in the recipe rather than applying an imaging PSF assumption to
every dataset.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/operated_light_profile/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/advanced/operated_light_profile/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

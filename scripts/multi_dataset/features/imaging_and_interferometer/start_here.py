"""
Features: Imaging And Interferometer
====================================

Combine resolved imaging with interferometer visibilities to constrain one lens while allowing the observed emission to differ. Optical or near-infrared light and radio or submillimetre emission can reveal complementary source structure.

__Scientific Use Cases__

- Compare the stellar-light and dust or gas-emission morphology of a lensed galaxy.
- Constrain shared lens mass with observations that probe different parts of the source.
- Use the relevant measurement process for each dataset instead of treating a reconstructed interferometer image as another CCD exposure.

__This Dataset and Lens Regime__

Decide explicitly which parameters are shared, which vary by dataset and whether the fits are simultaneous
or sequential. Each dataset retains its own response and uncertainty treatment. Inspect individual residuals
alongside the combined result; sharing a lens does not require identical emission in every observation.

![imaging and interferometer](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

__Two Measurements of a Shared System__

Imaging compares a PSF-blurred model with image pixels. Interferometry compares a Fourier-domain prediction
with visibilities. A displayed interferometer image is useful for inspection, but does not replace the
visibility likelihood used by these examples.

Shared geometry does not require identical emission. Different wavelengths may trace different components of a
galaxy, so decide which parameters should be linked and which should remain dataset-specific.

__Reading the Model__

Follow the construction of each dataset's analysis and then their combination. The example retains its actual
shared parameters, light components and instrumental settings. Inspect the fit to each dataset separately: an
acceptable joint score can conceal a poor description of one observation.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

"""
Features: Same Wavelength
=========================

Constrain a common lens and light distribution with repeated observations in the same band, retaining each exposure's response and noise. Joint fitting can use the information from different resolutions or signal-to-noise levels without first forcing every image onto one measurement model.

__Scientific Use Cases__

- Combine repeat exposures or observations with different PSFs and sampling.
- Infer shared physical parameters while forward-modeling each observation separately.
- Check consistency between observations through their individual residuals.

__This Dataset and Lens Regime__

Decide explicitly which parameters are shared, which vary by dataset and whether the fits are simultaneous
or sequential. Each dataset retains its own response and uncertainty treatment. Inspect individual residuals
alongside the combined result; sharing a lens does not require identical emission in every observation.

![same wavelength](../../../../images/new_user_guide/dataset_types.png)

Reused new-user-guide schematic. The interferometer panel shows sampling locations, not visibility amplitudes or phases; cube slices are spectral channels. These are conceptual dataset representations, not measurements.

__One Model, Several Forward Calculations__

The observations need not have the same PSF, sampling or signal-to-noise. Each analysis predicts what the
common model would look like through its own measurement process, rather than treating all images as identical
copies.

Whether additional quantities such as normalizations or offsets should be shared depends on the dataset and
the model. The example's construction shows its actual choices.

__Inspect Every Observation__

The combined likelihood brings the measurements together, but residuals should still be examined for each
observation. An instrumental mismatch in one exposure may otherwise be hidden by a strong fit to another.

Use the simulator to see the differences introduced between the observations, then follow how those
differences are handled in the modeling script.


__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

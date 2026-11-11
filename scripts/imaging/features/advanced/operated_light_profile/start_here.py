"""
Features: Operated Light Profile
================================

An operated light profile describes emission after an instrumental operation has already been included in the
profile. For imaging, the common example is a profile that already describes the PSF-blurred appearance of
compact emission.

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

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
foreground light, deflecting mass and background source play different roles; retain the particular component
choices stated in each recipe.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/operated_light_profile/modeling.ipynb) — Fit the model with a non-linear search.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/operated_light_profile/simulator.ipynb) — Generate the example dataset.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

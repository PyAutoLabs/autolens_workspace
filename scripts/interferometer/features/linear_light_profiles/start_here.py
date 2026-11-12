"""
Features: Linear Light Profiles
===============================

Solve component brightnesses during the fit instead of asking the non-linear search to sample a separate intensity for every light profile. Brightness is recalculated at every trial lens and light geometry: solved does not mean fixed.

__Scientific Use Cases__

- Simplify a model with many foreground or source light components by reducing its sampled dimensions.
- Let component brightnesses respond together as the lens mass and light-profile shapes change.
- Build flexible light descriptions, including MGEs, without sampling every basis coefficient.

__This Dataset and Lens Regime__

These examples fit complex visibilities: the lensed sky prediction is Fourier transformed and compared at
the measured baseline coordinates with the supplied noise. A dirty or reconstructed image is useful for
inspection, but does not replace this likelihood. Interpret source structure in the context of visibility
coverage and the assumptions used to model the sky.

![linear light profiles](../../../../images/features/linear_light_profiles.png)

Conceptual illustration. The search diagrams are conceptual, not measured likelihood surfaces or a timing benchmark. The scene illustrates sky/source structure; the fit here is to visibilities, not the displayed image.

__What Is Linear About It?__

Once a profile's centre, shape and size are specified, changing its intensity simply scales its image. The
image of several profiles is a weighted sum of their individual images. This allows their intensities to be
determined together by linear algebra, accounting for the data and noise.

The other model parameters still need to be fitted. At each proposed geometry, the profile images are
recalculated and their intensities are solved again. Solving brightness therefore does not mean fixing it, and
the best intensities depend on the rest of the model.

__Why This Changes the Search__

With ordinary profiles, the search explores brightness as well as shape. With linear profiles, brightness is
determined conditionally at every step, reducing the number of sampled parameters. This is especially useful
when a model contains many light components.

The calculation still has a cost, and the total run time depends on both the likelihood calculation and the
search. The examples show the construction and solved intensities so that you can inspect what changed, rather
than treating the linear solve as a black box.

__Reading the Examples__

Start with the direct fit to see how the solved intensities are retrieved. Then read the modeling example to
see which parameters remain in the non-linear model. The likelihood example explains the underlying
calculation in more detail.


__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/likelihood_function.ipynb) — Follow the likelihood calculation.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

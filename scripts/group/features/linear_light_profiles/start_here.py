"""
Features: Linear Light Profiles
===============================

A linear light profile has the same spatial form as an ordinary light profile, but its intensity is solved
during each likelihood evaluation instead of being sampled by the non-linear search.

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

__In This Folder__

The examples contain a group-scale lens with several galaxy components. A shared group halo, where present, is
an explicit model choice. Keep the individual deflectors, companion population and source components distinct
when following the recipe.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/linear_light_profiles/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/linear_light_profiles/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/linear_light_profiles/likelihood_function.ipynb) — Follow the likelihood calculation.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/linear_light_profiles/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

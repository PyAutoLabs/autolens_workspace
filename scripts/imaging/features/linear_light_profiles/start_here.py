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

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
foreground light, deflecting mass and background source play different roles; retain the particular component
choices stated in each recipe.

__Advantages__

Each light profile's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in this example by 2 dimensions).

This also removes the degeneracies that occur between the `intensity` and other light profile parameters
(e.g. `effective_radius`, `sersic_index`), which are difficult degeneracies for the non-linear search to map out
accurately. This produces more reliable lens model results and the fit converges in fewer iterations, speeding up the
overall analysis.

The inversion has a relatively small computational cost, thus we reduce the model complexity without much slow-down and
can therefore fit models more reliably and faster!

__Disadvantages__

Although the computation time of the inversion is small, it is not non-negligible. It is approximately 3-4x slower
than using a standard light profile.

The gains in run times due to the simpler non-linear parameter space therefore are somewhat balanced by the slower
likelihood calculation.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algebra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/likelihood_function.ipynb) — Follow the likelihood calculation.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

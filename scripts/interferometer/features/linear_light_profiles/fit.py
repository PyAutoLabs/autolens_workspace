"""
Modeling Features: Linear Light Profiles Fit (Interferometer)
==============================================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood) given
the light profile's other parameters.

This script illustrates how to perform a single `FitInterferometer` of a linear light profile model — that is,
not the full Nautilus model-fit, but a single likelihood evaluation given known light/mass profile parameters.
This is useful for understanding how the inversion produces the solved-for `intensity`, and how to extract that
value from the resulting fit.

For an explanation of why linear light profile fits are now practical against visibility data thanks to the
JAX-native NUFFT `nufftax` (https://github.com/GragasLab/nufftax), see the companion `modeling.py` example.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of linear light profiles for interferometer data.
- **Positive Only Solver:** Ensuring positive-only solutions for linear light profile intensities.
- **Model:** The lens model whose `intensity` we solve for via inversion.
- **Mask:** Define the `real_space_mask` which sets the grid the strong lens is evaluated on.
- **Dataset:** Load the strong lens `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Fit:** Perform a single `FitInterferometer` using the model and inspect the inversion.
- **Intensities:** Extract the solved-for `intensity` via `fit.linear_light_profile_intensity_dict`.
- **Visualization:** Build the helper tracer where linear light profiles are replaced with ordinary light
  profiles carrying their solved-for `intensity`, then plot.
- **Wrap Up:** Summary of the script and next steps.

__Advantages__

The source galaxy's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by one. The lens light is already omitted for interferometer data,
so the saving is smaller than the imaging case (where lens and source both contribute) — but the inversion still
removes the degeneracies between `intensity` and the source's shape parameters (e.g. `effective_radius`,
`sersic_index`), which are difficult degeneracies for the non-linear search to map out accurately.

The inversion has a relatively small computational cost on top of the NUFFT, so we reduce the model complexity
without much slow-down.

__Disadvantages__

Although the computation time of the inversion is small, it is not non-negligible. It is approximately 3-4x
slower per inversion-only term than using a standard light profile with a fixed `intensity`. The NUFFT typically
dominates the total per-likelihood cost on interferometer data, so the overall slow-down is usually smaller.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algebra solver which allows for positive and
negative values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a
galaxy's light, which is clearly unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is
as fast as positive-negative solvers. This ensures that all light profile intensities are positive and
therefore physical.

__Model__

This script fits an `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data). This is the standard
   convention for interferometer modeling.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Mask__

We define the `real_space_mask` which defines the grid the image of the strong lens is evaluated on.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files, using `TransformerNUFFT`
backed by `nufftax`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Fit__

We now illustrate how to perform a fit to the dataset using a linear light profile, with the lens mass and the
source's shape parameters fixed to known values.

The API follows closely the standard use of a `FitInterferometer` object, but simply uses a linear light
profile (via the `lp_linear` module) instead of a standard light profile.

Note that the linear light profile below does not have an `intensity` parameter input — we let the inversion
solve for it.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

"""
The fit's `subplot_fit_interferometer` shows the visibility-plane fit and dirty-image residuals. Because the
source bulge is a linear light profile, the inversion has solved for its `intensity` to maximize the fit to
the observed visibilities.
"""
aplt.subplot_fit_interferometer(fit=fit)

"""
The `subplot_fit_dirty_images` provides a real-space view of the data, model and residuals via inverse-NUFFT
of the visibility-plane quantities. This is generally more interpretable to the human eye than the uv-plane
plots above.
"""
aplt.subplot_fit_dirty_images(fit=fit)

"""
__Intensities__

The fit contains the solved-for `intensity` value.

This is computed using a fit's `linear_light_profile_intensity_dict`, which maps each linear light profile in
the model parameterization above to its `intensity`.

The code below shows how to use this dictionary, as an alternative to using the `max_log_likelihood` quantities
covered in `modeling.py`.
"""
source_bulge = tracer.galaxies[-1].bulge

print(fit.linear_light_profile_intensity_dict)

print(
    f"\n Intensity of source bulge (lp_linear.SersicCore) = "
    f"{fit.linear_light_profile_intensity_dict[source_bulge]}"
)

"""
A `Tracer` where all linear light profile objects are replaced with ordinary light profiles using the
solved-for `intensity` values is also accessible from a fit.

For example, the linear `SersicCore` of the source `bulge` component above has a solved-for `intensity` of
~0.3.

The `tracer` created below instead has an ordinary `SersicCore` light profile with `intensity` ~0.3. The
benefit of this tracer is that it can be visualised (linear light profiles cannot be plotted by default
because they do not have `intensity` values).
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

print(tracer.galaxies[-1].bulge.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies, a tracer) cannot be plotted because they
do not have an `intensity` value.

Therefore, the helper-tracer created above (with all linear light profiles replaced by ordinary light profiles
carrying their solved-for `intensity`) must be used for visualization:
"""
aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Tracer Image")


"""
__Wrap Up__

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""

"""
Modeling Features: Shapelets Fit (Interferometer)
==================================================

A shapelet is a basis function appropriate for capturing the exponential / disk-like features of a galaxy.
The `intensity` of every shapelet is solved for via linear algebra ("inversion") rather than being a free
parameter of the non-linear search.

This script illustrates how to perform a single `FitInterferometer` of a shapelet source model — that is,
not the full Nautilus model-fit, but a single likelihood evaluation given known basis parameters. This is
useful for understanding how the inversion produces the per-shapelet solved-for `intensity` values, and
how to extract them from the resulting fit.

For an explanation of why shapelet fits to visibility data are now practical thanks to the JAX-native
NUFFT `nufftax` (https://github.com/GragasLab/nufftax), and why shapelets require the positive-negative
solver, see the companion `modeling.py` example.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of a shapelet source for interferometer data.
- **Positive Negative Solver:** Why shapelets require the positive-negative solver (unlike MGE or linear
  Sersic).
- **Model:** The lens model whose `intensity` values we solve for via inversion.
- **Mask:** Define the `real_space_mask` which sets the grid the strong lens is evaluated on.
- **Dataset:** Load the strong lens `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Basis:** Build the linear polar shapelet basis used as the source bulge.
- **Fit:** Perform a single `FitInterferometer` and inspect the inversion.
- **Intensities:** Extract the per-shapelet solved-for `intensity` values.
- **Visualization:** Build the helper tracer where linear light profiles are replaced with ordinary light
  profiles carrying their solved-for `intensity`, then plot.
- **Wrap Up:** Summary of the script and next steps.

__Model__

This script fits an `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data). Interferometer
   convention.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's bulge is a superposition of linear `ShapeletPolar` profiles with shared centre,
   ell_comps, and beta.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

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
simulator script.
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
__Basis__

We build a `Basis` of linear polar shapelets for the source bulge. All shapelets share centre, ell_comps,
and a single `beta` size scale; the (n, m) quantum numbers are assigned procedurally.

We use `lp_linear.ShapeletPolar`, which solves for each shapelet's `intensity` analytically via the
inversion. Linear light profiles are described in detail in the `linear_light_profiles.py` example.
"""
total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = []
n_count = 1
m_count = -1

for i in range(total_n + total_m + 1):
    if i == 0:
        n, m = 0, 0
    else:
        n, m = n_count, m_count
        m_count += 2
        if m_count > n_count:
            n_count += 1
            m_count = -n_count

    shapelet = al.lp_linear.ShapeletPolar(
        n=n,
        m=m,
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        beta=0.1,
    )
    shapelets_bulge_list.append(shapelet)

source_bulge = al.lp_basis.Basis(profile_list=shapelets_bulge_list)

"""
__Fit__

We now illustrate the API for performing a single shapelet fit using standard `Galaxy`, `Tracer` and
`FitInterferometer` objects. Once we have a `Basis`, we can treat it like any other light profile.

Note `Settings(use_positive_only_solver=False)` is passed to the fit — shapelets require the
positive-negative solver to function.
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

source = al.Galaxy(redshift=1.0, bulge=source_bulge)

tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitInterferometer(
    dataset=dataset,
    tracer=tracer,
    settings=al.Settings(use_positive_only_solver=False),
)

"""
The fit's `subplot_fit_interferometer` shows the visibility-plane fit and dirty-image residuals. Because
the source bulge is a Basis of linear light profiles, the inversion has solved for each shapelet's
`intensity` to maximize the fit to the observed visibilities.
"""
aplt.subplot_fit_interferometer(fit=fit)

"""
The `subplot_fit_dirty_images` provides a real-space view of the data, model and residuals via
inverse-NUFFT of the visibility-plane quantities.
"""
aplt.subplot_fit_dirty_images(fit=fit)

"""
__Intensities__

The fit contains the solved-for `intensity` value of every shapelet in the basis.

These are computed via `fit.linear_light_profile_intensity_dict`, which maps each linear light profile in
the model to its inferred `intensity`. Print the first few entries for brevity.
"""
print(fit.linear_light_profile_intensity_dict)

for shapelet in shapelets_bulge_list[:5]:
    intensity = fit.linear_light_profile_intensity_dict[shapelet]
    print(f"  n={shapelet.n}  m={shapelet.m}  intensity = {intensity:+.6e}")

print(
    f"\n  number of negative-intensity shapelets: "
    f"{sum(1 for s in shapelets_bulge_list if fit.linear_light_profile_intensity_dict[s] < 0)} "
    f"/ {len(shapelets_bulge_list)}"
)

"""
A `Tracer` where all linear light profile objects are replaced with ordinary light profiles using the
solved-for `intensity` values is also accessible from a fit.

The benefit of this helper-tracer is that it can be visualised (linear light profiles cannot be plotted
by default because they do not have `intensity` values).
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

"""
__Visualization__

The helper-tracer created above replaces every linear `ShapeletPolar` with an ordinary `ShapeletPolar`
carrying its solved-for `intensity` — that tracer can be plotted directly.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Tracer Image (shapelet source)")


"""
__Wrap Up__

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""

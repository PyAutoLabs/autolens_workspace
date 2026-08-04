"""
__Log Likelihood Function: Group DSPL__

This script describes the additional steps required to compute the `log_likelihood` for a group-scale double
source-plane lens (DSPL) — two source galaxies at different redshifts lensed by multiple main lens galaxies at
the lens-plane redshift.

This script does NOT repeat the steps shared with single-plane or single-lens-galaxy lensing (mask, image-plane
grid, convolution, chi-squared, noise normalization, MGE linear algebra). It documents only what is new for the
combined group + DSPL case.

__Prerequisites__

The likelihood function below builds on standard imaging, MGE and single-lens DSPL likelihoods.
Read these first:

 - `autolens_workspace/scripts/imaging/likelihood_function.py` — single-plane imaging log likelihood, covering
   chi-squared and noise normalization.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py` — the MGE
   `Basis` of linear Gaussians and the associated linear-algebra terms.
 - `autolens_workspace/scripts/imaging/features/advanced/double_source_plane_lens/likelihood_function.py` — the
   multi-plane deflection chain for a single-lens-galaxy DSPL.

This script focuses entirely on what differs for the group-scale case.

__Contents__

- **Prerequisites:** Reading order before this script (see above).
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Main Lens Centres:** Load the two main lens galaxy centres from JSON.
- **Galaxies:** Multiple main lens galaxies at z=0.5, source_0 at z=1.0 (light+mass), source_1 at z=2.0 (light).
- **Multi-Plane Ray-Tracing:** The deflection chain to `source_1`'s plane accumulates contributions from EVERY
  main lens galaxy at z=0.5, plus from `source_0`'s mass.
- **Source-Plane Images:** Both source galaxies are evaluated at their respective ray-traced grids.
- **Likelihood:** Reference up to canonical scripts for chi-squared / noise / linear algebra.
- **Fit Check:** Confirm the manual reconstruction matches `FitImaging.log_likelihood`.
- **Wrap Up.**

__What Changes Relative To The Single-Lens DSPL__

In the single-lens DSPL likelihood function, the deflection map applied to image-plane
coordinates was simply `alpha_lens(theta)` — the deflection field of the lone foreground lens galaxy.

In the group case, the lens-plane contains MULTIPLE main lens galaxies (indexed `lens_0`, `lens_1`, ...), each
with its own mass distribution. The lens-plane deflection field is the SUM of every main lens galaxy's
deflection:

  alpha_total(theta) = sum_i alpha_lens_i(theta)

The remainder of the multi-plane chain is unchanged from the single-lens DSPL case:

  Plane 0 (image-plane)        : theta
  Plane 1 (source_0 at z=1.0)  : theta - alpha_total(theta)
  Plane 2 (source_1 at z=2.0)  : theta - alpha_total(theta) - beta_01 * alpha_source_0(plane_1_grid)

PyAutoLens handles this summation automatically inside `Tracer.traced_grid_2d_list_from`, so no special code is
required — but it is the conceptual difference relative to a single-lens-galaxy DSPL.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the group DSPL dataset.
"""
dataset_name = "double_source_plane_lens"
dataset_path = Path("dataset") / "group" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/double_source_plane_lens/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 4.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Galaxies__

The galaxies that participate in the multi-plane ray-tracing:

 - Two main lens galaxies at z=0.5, each with an `IsothermalSph` mass and a simple MGE bulge. Their mass
   profiles together set the deflection field for the lens-plane.
 - `source_0` at z=1.0 with an MGE bulge AND an `IsothermalSph` mass — this source deflects light from
   `source_1`.
 - `source_1` at z=2.0 with an MGE bulge only.

Each galaxy's parameters are set to the simulator's true values so the manual likelihood computation produces a
sensible model image.
"""
total_gaussians = 10
log10_sigma_list_lens = np.linspace(-2, np.log10(2.0), total_gaussians)
log10_sigma_list_source = np.linspace(-2, np.log10(0.5), total_gaussians)


def build_basis(centre, log10_sigma_list):
    gaussian_list = [
        al.lp_linear.Gaussian(
            centre=centre,
            ell_comps=(0.0, 0.0),
            sigma=10 ** log10_sigma_list[i],
        )
        for i in range(total_gaussians)
    ]
    return al.lp_basis.Basis(profile_list=gaussian_list)


lens_galaxies = []
for centre in main_lens_centres:
    lens_galaxies.append(
        al.Galaxy(
            redshift=0.5,
            bulge=build_basis(
                centre=(centre[0], centre[1]), log10_sigma_list=log10_sigma_list_lens
            ),
            mass=al.mp.IsothermalSph(
                centre=(centre[0], centre[1]), einstein_radius=1.2
            ),
        )
    )

source_0 = al.Galaxy(
    redshift=1.0,
    bulge=build_basis(centre=(0.0, 0.0), log10_sigma_list=log10_sigma_list_source),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.25),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=build_basis(centre=(-0.3, 0.3), log10_sigma_list=log10_sigma_list_source),
)

tracer = al.Tracer(galaxies=lens_galaxies + [source_0, source_1])

"""
__Multi-Plane Ray-Tracing__

`traced_grid_2d_list_from` returns one grid per plane, in redshift order. For the group DSPL case there are
still THREE planes (image, source_0, source_1) — the multi-plane chain does not grow with the number of main
lens galaxies, because all main lens galaxies share the same redshift z=0.5.

What changes is the deflection field applied at the lens-plane boundary: it is the SUM over all main lens
galaxies' deflection contributions, computed automatically by `Tracer`.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=dataset.grid)

grid_image_plane = traced_grid_list[0]
grid_source_0 = traced_grid_list[1]
grid_source_1 = traced_grid_list[2]

print(f"Number of planes traced: {len(traced_grid_list)}")
print(f"Number of main lens galaxies at z=0.5: {len(lens_galaxies)}")
print(f"Plane 1 (source_0) first coord: {grid_source_0[0]}")
print(f"Plane 2 (source_1) first coord: {grid_source_1[0]}")

aplt.plot_grid(grid=grid_source_0, title="Ray-traced grid at source_0 plane (z=1.0)")
aplt.plot_grid(grid=grid_source_1, title="Ray-traced grid at source_1 plane (z=2.0)")

"""
__Source-Plane Images__

`tracer.image_2d_from` evaluates every galaxy's light at the correct plane and sums the contributions into a
single model image. Internally it performs:

  1. Ray-traces the image-plane grid through the lens-plane (summing the deflection fields of all main lens
     galaxies) to obtain `grid_source_0`.
  2. Continues to ray-trace through `source_0`'s mass to obtain `grid_source_1`.
  3. Evaluates `source_0`'s MGE basis at `grid_source_0` and `source_1`'s MGE basis at `grid_source_1`.
  4. Sums all source contributions, returning the model image.

The only group-specific step is (1). Every other step is identical to the imaging DSPL case.
"""
model_image_unconvolved = tracer.image_2d_from(grid=dataset.grid)

aplt.plot_array(
    array=model_image_unconvolved, title="Model image before PSF convolution"
)

"""
__Likelihood__

PSF convolution, chi-squared, noise normalization, and the MGE linear-algebra terms are unchanged. We delegate
to `FitImaging`, which handles the linear solve that recovers each Gaussian's `intensity` and assembles the
full `log_likelihood`.

For the form of these terms, refer to:

 - `imaging/likelihood_function.py` — chi-squared and noise normalization.
 - `imaging/features/multi_gaussian_expansion/likelihood_function.py` — linear-algebra terms.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"\nLog likelihood of the manual group DSPL fit: {fit.log_likelihood}")

"""
__Wrap Up__

The group DSPL `log_likelihood` differs from the single-lens DSPL case in exactly one place:
the lens-plane deflection field is the sum of contributions from EVERY main lens galaxy at z=0.5, rather than
from a lone lens galaxy. `Tracer` handles the summation automatically — no new code is required.

The deflection scaling factor `beta_01` between source_0 and source_1 (and therefore the cosmological
sensitivity of the system) is unchanged from the imaging case. Group DSPLs are valuable for cosmology because
they tend to have larger Einstein radii, which improves astrometric precision and therefore the angular
diameter distance ratios that `beta_01` depends on.
"""

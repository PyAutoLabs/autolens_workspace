"""
__Log Likelihood Function: Stellar and Dark Mass (Multi Galaxy)__

This script walks through the log likelihood function of a multi-galaxy lens whose deflectors have **decomposed
stellar and dark mass**, step by step.

__Prerequisites__

Read these first; they are not repeated here:

 - `multi_galaxy/likelihood_function.py` — the same walkthrough with one total mass profile per deflector. It
   covers the summed deflection field and the multi-deflector light subtraction in full.
 - `imaging/features/advanced/mass_stellar_dark/likelihood_function.py` — the single-deflector decomposition.

__Contents__

- **What Changes:** Which steps a decomposition touches.
- **Dataset, Mask & Over Sampling:** Set up.
- **Galaxies:** The two deflectors, each with a stellar component and a dark halo.
- **Lens Light:** The light, which now comes from the same profile as the stellar mass.
- **Decomposed Deflection:** Four mass profiles summed into one field.
- **Ray Tracing:** Trace the grid with that field.
- **Source-Plane Image:** Evaluate the source on the traced grid.
- **Likelihood:** Chi-squared and the noise normalization.
- **Fit:** Confirm the walkthrough matches `FitImaging`.
- **Wrap Up:** Where to go next.

__What Changes__

Exactly one step, and it is not the one people expect.

`multi_galaxy/likelihood_function.py` computes a deflection field by summing one mass profile per deflector. This
script sums **two per deflector** — a stellar component and a dark halo. That is the whole structural difference:
the sum has four terms instead of two, plus the shear.

Everything downstream — ray tracing, evaluating the source, convolving, chi-squared — is identical, because it
only ever sees the summed field.

The step that does *not* change but is worth noticing: the lens light. With a total mass profile, the light and
the mass are separate objects that happen to sit at the same place. With an `lmp.Sersic` they are **one object**
— the same `intensity`, `effective_radius` and `sersic_index` produce both the light you subtract and the
stellar mass you deflect with. Changing the light model here changes the deflection field, which is not true in
any other script in this package.

__Simplifications__

The source uses an ordinary light profile with an input intensity, so there is no inversion and the likelihood
is a plain chi-squared.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `mass_stellar_dark` multi-galaxy dataset.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/mass_stellar_dark/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__
"""
mask_radius = 3.0

masked_dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

"""
__Over Sampling__

Disabled for this step-by-step guide, so each image pixel is one coordinate.
"""
masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Galaxies__

The two deflectors at their simulated values, each with a stellar component and a dark halo.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
        mass_to_light_ratio=0.6,
    ),
    dark=al.mp.NFWSph(centre=(0.35, 0.25), kappa_s=0.08, scale_radius=15.0),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
        mass_to_light_ratio=0.4,
    ),
    dark=al.mp.NFWSph(centre=(-0.35, -0.25), kappa_s=0.06, scale_radius=15.0),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

"""
__Lens Light__

The light subtracted from the data is the sum of both deflectors' `bulge` images — the same profiles that supply
the stellar mass below.
"""
lens_0_image = lens_0.bulge.image_2d_from(grid=masked_dataset.grid)
lens_1_image = lens_1.bulge.image_2d_from(grid=masked_dataset.grid)

total_lens_image = lens_0_image + lens_1_image

aplt.plot_array(array=total_lens_image, title="Total Lens Light (Both Deflectors)")

"""
__Decomposed Deflection__

Four mass profiles, plus the shear. Each is evaluated on the same grid and the results are added.
"""
deflections_0_stellar = np.asarray(lens_0.bulge.deflections_yx_2d_from(grid=masked_dataset.grid))
deflections_0_dark = np.asarray(lens_0.dark.deflections_yx_2d_from(grid=masked_dataset.grid))
deflections_1_stellar = np.asarray(lens_1.bulge.deflections_yx_2d_from(grid=masked_dataset.grid))
deflections_1_dark = np.asarray(lens_1.dark.deflections_yx_2d_from(grid=masked_dataset.grid))
deflections_shear = np.asarray(shear_galaxy.shear.deflections_yx_2d_from(grid=masked_dataset.grid))

deflections_total = (
    deflections_0_stellar
    + deflections_0_dark
    + deflections_1_stellar
    + deflections_1_dark
    + deflections_shear
)

print(f"mean |deflection|, lens_0 stellar = {np.mean(np.abs(deflections_0_stellar)):.6f}")
print(f"mean |deflection|, lens_0 dark    = {np.mean(np.abs(deflections_0_dark)):.6f}")
print(f"mean |deflection|, lens_1 stellar = {np.mean(np.abs(deflections_1_stellar)):.6f}")
print(f"mean |deflection|, lens_1 dark    = {np.mean(np.abs(deflections_1_dark)):.6f}")
print(f"mean |deflection|, total          = {np.mean(np.abs(deflections_total)):.6f}")

"""
The data constrains that total. Every statement about an individual component is inferred from a sum, which is
why decompositions are harder than total mass models — and why, with two galaxies contributing two components
each, `modeling.py` argues for tying the two galaxies' mass-to-light ratios together.

__Ray Tracing__

Trace the image-plane grid to the source plane by subtracting the total deflection.
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

"""
Confirm the tracer's field is the sum computed by hand above.
"""
tracer_deflections = np.asarray(
    tracer.deflections_yx_2d_from(grid=masked_dataset.grid)
)

print(
    f"\nMax difference, summed components vs tracer: "
    f"{np.max(np.abs(tracer_deflections - deflections_total)):.3e}"
)

aplt.plot_grid(grid=traced_grid, title="Source Plane Grid (Traced)")

"""
__Source-Plane Image__

The source is evaluated on the traced grid. It sees only that grid, so it cannot tell whether the deflection
that produced it came from two mass profiles or four.
"""
source_image = source_galaxy.bulge.image_2d_from(grid=traced_grid)

print(f"Total source flux in the image plane: {np.sum(source_image):.4f}")

"""
__Likelihood__

With no linear objects in the model, the log likelihood is the standard chi-squared expression:

 -2 ln L = chi^2 + N ln(2 pi sigma^2)

A decomposition adds nothing to it. What it changes is how the model image was produced.

__Fit__

`FitImaging` performs every step above.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

print(f"Log likelihood: {fit.log_likelihood}")
print(f"Chi-squared:    {fit.chi_squared}")

aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/mass_stellar_dark/fit.py` — the same decomposition with the component
   fractions computed as a function of radius.
 - `multi_galaxy/features/advanced/mass_stellar_dark/modeling.py` — fitting it, and the tying choice.
 - `multi_galaxy/likelihood_function.py` — the total-mass walkthrough this one extends.
 - `imaging/features/advanced/mass_stellar_dark/likelihood_function.py` — the single-deflector version.
"""

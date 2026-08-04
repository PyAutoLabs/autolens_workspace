"""
__Log Likelihood Function: DSPL (Multi Galaxy)__

This script walks through the log likelihood function of a multi-galaxy double source-plane lens, step by step,
so the calculation can be followed without reading the source code.

__Prerequisites__

Read these first; they are not repeated here:

 - `multi_galaxy/likelihood_function.py` — the two-plane multi-galaxy walkthrough. It covers the summed
   deflection field and the multi-deflector light subtraction in full.
 - `imaging/features/advanced/double_source_plane_lens/likelihood_function.py` — the single-deflector DSPL
   derivation.

__Contents__

- **What Changes:** Which steps a second source plane touches.
- **Dataset, Mask & Over Sampling:** Set up.
- **Galaxies:** The three planes, at their simulated values.
- **Multi-Plane Ray-Tracing:** Trace the grid plane by plane, by hand.
- **Source-Plane Images:** Evaluate each source on the grid that reaches it.
- **Image Assembly:** Sum the planes and convolve with the PSF.
- **Likelihood:** Chi-squared and the noise normalization.
- **Fit:** Confirm the walkthrough matches `FitImaging`.
- **Wrap Up:** Where to go next.

__What Changes__

`multi_galaxy/likelihood_function.py` traces the image-plane grid once, through a deflection field that is the
sum of both deflectors' contributions.

A DSPL traces it **twice**, and the second trace is not a repeat of the first:

 1. Image plane to z=1.0 — subtract the summed deflection of both deflectors and the shear, scaled for the
    z=0.5 to z=1.0 geometry. This is exactly the trace the two-plane script performs.

 2. z=1.0 to z=2.0 — subtract the deflection of everything at or before z=1.0, which now includes `source_0`'s
    own mass, scaled for the z=0.5/z=1.0 to z=2.0 geometry.

Step 2 is what makes this multi-plane. If `source_0` were massless, the grid reaching z=2.0 would be the grid
reaching z=1.0 rescaled by a single number, and nothing structurally new would happen.

Everything after the tracing — evaluating light profiles, convolving, chi-squared — is unchanged from the
two-plane case, applied to one more plane.

__Simplifications__

The galaxies below use ordinary light profiles with input intensities rather than linear ones, so there is no
inversion in this script and the likelihood is a plain chi-squared. `multi_galaxy/likelihood_function.py`
covers the linear-algebra case.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `dspl` multi-galaxy dataset.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "dspl"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/double_source_plane_lens/simulator.py",
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

The standard 3.0" circular mask, sized to contain both Einstein rings.
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

Over sampling is disabled for this step-by-step guide, so each image pixel is one coordinate and the arrays
below are easy to follow.
"""
masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Galaxies__

The three planes, at the values `simulator.py` used.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.03), einstein_radius=0.15),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.25, 0.28),
        intensity=1.5,
        effective_radius=0.08,
    ),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_0, source_1])

"""
__Multi-Plane Ray-Tracing__

`traced_grid_2d_list_from` returns one grid per plane. Index 0 is the image plane itself, index 1 is the grid
as it arrives at z=1.0, index 2 as it arrives at z=2.0.
"""
traced_grids = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)

print(f"Number of planes: {len(traced_grids)}")

"""
The deflection applied between the image plane and z=1.0 is the sum of both deflectors' and the shear's, which
is the same quantity `multi_galaxy/likelihood_function.py` computes.
"""
deflections_to_plane_1 = np.asarray(traced_grids[0]) - np.asarray(traced_grids[1])

print(
    f"Mean |deflection| image plane -> z=1.0: {np.mean(np.abs(deflections_to_plane_1)):.6f}"
)

"""
The deflection applied between z=1.0 and z=2.0 includes `source_0`'s mass on top of the rescaled lens-plane
contribution. It is not a constant multiple of the first, and the spread below is one way to see that.
"""
deflections_1_to_2 = np.asarray(traced_grids[1]) - np.asarray(traced_grids[2])

ratio = np.abs(deflections_1_to_2) / (np.abs(deflections_to_plane_1) + 1e-12)

print(
    f"Mean |deflection| z=1.0 -> z=2.0:       {np.mean(np.abs(deflections_1_to_2)):.6f}"
)
print(
    f"Ratio of the two, spread across the grid: min {ratio.min():.4f}, max {ratio.max():.4f}"
)

"""
__Source-Plane Images__

Each source is evaluated on the grid that reaches its own plane. `source_0` sees `traced_grids[1]`, `source_1`
sees `traced_grids[2]`.
"""
lens_0_image = lens_0.image_2d_from(grid=masked_dataset.grid)
lens_1_image = lens_1.image_2d_from(grid=masked_dataset.grid)

source_0_image = source_0.bulge.image_2d_from(grid=traced_grids[1])
source_1_image = source_1.bulge.image_2d_from(grid=traced_grids[2])

print(f"Total flux, deflectors: {np.sum(lens_0_image) + np.sum(lens_1_image):.4f}")
print(f"Total flux, source_0:   {np.sum(source_0_image):.4f}")
print(f"Total flux, source_1:   {np.sum(source_1_image):.4f}")

"""
__Image Assembly__

The model image is the sum of every plane's contribution, convolved with the PSF. The tracer does this in one
call, and the result should match the four arrays above summed and blurred.
"""
model_image = tracer.image_2d_from(grid=masked_dataset.grid)

print(f"Total flux, all planes: {np.sum(model_image):.4f}")

"""
__Likelihood__

With no linear objects in the model, the log likelihood is the standard chi-squared expression:

 -2 ln L = chi^2 + N ln(2 pi sigma^2)

where `chi^2 = sum((d - model)^2 / sigma^2)` over the masked pixels.

A DSPL adds nothing to this expression. What it changes is the model image the chi-squared is computed against —
which now carries two source planes, produced by two traces rather than one.

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

 - `multi_galaxy/features/advanced/double_source_plane_lens/fit.py` — the same fit with MGE bases and a linear
   intensity solve.
 - `multi_galaxy/likelihood_function.py` — the two-plane multi-galaxy walkthrough this one extends.
 - `imaging/features/advanced/double_source_plane_lens/likelihood_function.py` — the single-deflector DSPL
   derivation.
"""

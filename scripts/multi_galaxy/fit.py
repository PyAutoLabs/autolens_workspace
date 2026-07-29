"""
Fits: Multi Galaxy
==================

An anatomy of how a multi-galaxy lens fit works: compose the two co-dominant galaxies and the source by
hand, build the `Tracer`, and fit the imaging data with `FitImaging` — inspecting how each deflector
contributes to the lensing.

This is the multi-galaxy counterpart of `imaging/fit.py`, which walks through every step of the fitting
API in detail (model image, residual map, chi-squared, likelihood). The API is identical here — a fit
neither knows nor cares how many galaxies deflect the light — so this script focuses on the one thing
that IS different: **the deflection field is the sum of every deflector's field**, and each galaxy's
share of it can be inspected separately.

__Contents__

- **Dataset & Mask:** Load the multi-galaxy dataset (auto-simulating if absent).
- **Galaxies:** Compose the true pair + source by hand.
- **Per-Galaxy Deflections:** Each deflector's contribution to the summed deflection field.
- **Tracer + Fit:** Build the tracer and fit the data.
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset & Mask__
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxies__

Compose the truth: the values the simulator used (see `multi_galaxy/simulator.py`), so the fit below
sits at the likelihood's maximum. Note the small light/mass centre offsets on each galaxy — the
J1011+0143-style science this regime measures.
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

source = al.Galaxy(
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
__Per-Galaxy Deflections__

Because both deflectors are at one redshift, the total deflection field is simply the sum of each
galaxy's field. Evaluate each on the masked grid and compare their magnitudes at a point between the two
galaxies — co-dominance in numbers: neither field is negligible anywhere near the ring.
"""
grid = dataset.grids.lp

deflections_0 = lens_0.deflections_yx_2d_from(grid=grid)
deflections_1 = lens_1.deflections_yx_2d_from(grid=grid)

magnitude_0 = float(np.mean(np.linalg.norm(np.asarray(deflections_0.array), axis=-1)))
magnitude_1 = float(np.mean(np.linalg.norm(np.asarray(deflections_1.array), axis=-1)))

print(f'mean |deflection| lens_0 = {magnitude_0:.3f}"  lens_1 = {magnitude_1:.3f}"')
print(
    f"ratio = {magnitude_1 / magnitude_0:.2f}  (co-dominant: neither is a minor perturber)"
)

"""
__Tracer + Fit__

The tracer sums the deflection fields internally; `FitImaging` then does exactly what it does for a
single-galaxy lens — see `imaging/fit.py` for the full step-by-step anatomy (model image, residuals,
chi-squared, likelihood).
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

print(f"log likelihood at the truth = {float(fit.log_likelihood):.2f}")

aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

- `imaging/fit.py` — the complete fitting-API anatomy, identical machinery.
- `multi_galaxy/modeling.py` — fitting a model (rather than the truth) to this dataset.
- `multi_galaxy/features/scaling_galaxies` — adding a far-out scaling tier to the deflection sum.
"""

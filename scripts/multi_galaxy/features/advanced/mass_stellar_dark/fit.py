"""
Fits: Stellar and Dark Mass (Multi Galaxy)
==========================================

This script fits a multi-galaxy strong lens with decomposed stellar and dark mass, without a non-linear search,
so the individual mass components and the deflection field each contributes can be inspected directly.

__Contents__

- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Mask:** Standard set up of the mask that is fitted.
- **Centres:** The centres of the co-dominant deflectors.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **Galaxies:** The two deflectors, each with a stellar component and a dark halo.
- **Tracer:** Build the tracer.
- **Fit:** Fit the dataset.
- **Decomposed Deflection:** Pull each component's deflection field out separately.
- **Component Fractions:** How much of the total deflection each component supplies, with radius.
- **Wrap Up:** Where to go next.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light and stellar mass is an `lmp.Sersic`, at its simulated values.
 - Each co-dominant deflector's dark matter is an `NFWSph`, at its simulated values.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source's light is a `SersicCore`.

__What This Script Is For__

`modeling.py` argues that the two galaxies' mass-to-light ratios are near-degenerate with each other, and that
tying them is what makes the decomposition tractable. This script is where you can see the ingredients of that
argument directly: four mass profiles, two per galaxy, all summing into one deflection field that the data
constrains as a whole.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/fit.py` for the multi-galaxy fit anatomy and
`imaging/features/advanced/mass_stellar_dark/fit.py` for the single-deflector decomposition.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
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

"""
__Mask__

The standard 3.0" circular mask.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Over Sampling__
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxies__

The two deflectors at the values `simulator.py` used, each with a stellar component and a dark halo.

Note the two `mass_to_light_ratio` values differ. That is deliberate in the simulator, so `modeling.py`'s tied
model has something to be wrong about.
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
__Tracer__
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

"""
__Fit__
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log likelihood = {fit.log_likelihood}")

"""
__Decomposed Deflection__

Each mass profile has its own `deflections_yx_2d_from`, so the total field can be taken apart.

There are four contributions here, plus the shear: a stellar and a dark component for each of the two
deflectors. `multi_galaxy/fit.py` has two.
"""
grid = dataset.grid

deflections = {
    "lens_0 stellar": lens_0.bulge.deflections_yx_2d_from(grid=grid),
    "lens_0 dark": lens_0.dark.deflections_yx_2d_from(grid=grid),
    "lens_1 stellar": lens_1.bulge.deflections_yx_2d_from(grid=grid),
    "lens_1 dark": lens_1.dark.deflections_yx_2d_from(grid=grid),
    "shear": shear_galaxy.shear.deflections_yx_2d_from(grid=grid),
}

total = sum(np.asarray(d) for d in deflections.values())

for name, d in deflections.items():
    print(f"  mean |deflection|, {name:16s} = {np.mean(np.abs(np.asarray(d))):.6f}")

print(f"  mean |deflection|, {'total':16s} = {np.mean(np.abs(total)):.6f}")

"""
Confirm the tracer's own deflection field is that sum, so nothing above is a reconstruction of something the
tracer does differently.
"""
tracer_deflections = np.asarray(tracer.deflections_yx_2d_from(grid=grid))

print(
    f"\nMax difference, summed components vs tracer: "
    f"{np.max(np.abs(tracer_deflections - total)):.3e}"
)

"""
__Component Fractions__

The fraction of the total deflection each component supplies, as a function of distance from each galaxy.

This is the calculation behind `modeling.py`'s point about the mask. Near a galaxy's centre both of its
components are steep and hard to distinguish; further out their profiles diverge, and that is where the
decomposition is actually constrained. A mask cut tight removes those pixels.
"""
radii = np.array([0.25, 0.5, 1.0, 2.0, 3.0])

centre_0 = np.asarray(main_lens_centres[0])

for r in radii:
    ring = al.Grid2DIrregular(
        [
            (centre_0[0] + r * np.cos(theta), centre_0[1] + r * np.sin(theta))
            for theta in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        ]
    )

    stellar = np.mean(
        np.abs(np.asarray(lens_0.bulge.deflections_yx_2d_from(grid=ring)))
    )
    dark = np.mean(np.abs(np.asarray(lens_0.dark.deflections_yx_2d_from(grid=ring))))

    print(f'  r = {r:.2f}"  lens_0 stellar fraction = {stellar / (stellar + dark):.3f}')

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/mass_stellar_dark/modeling.py` — fitting this model with a search, and the
   mass-to-light tying choice.
 - `multi_galaxy/features/advanced/mass_stellar_dark/likelihood_function.py` — the same fit broken into steps.
 - `multi_galaxy/fit.py` — the total-mass multi-galaxy fit this one decomposes.
 - `imaging/features/advanced/mass_stellar_dark/fit.py` — the single-deflector decomposition.
"""

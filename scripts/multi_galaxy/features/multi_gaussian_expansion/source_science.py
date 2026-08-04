"""
Source Science: Multi Gaussian Expansion (Multi Galaxy)
=======================================================

Science calculations on a multi-galaxy lens fitted with an MGE per co-dominant deflector: the **luminosity of each
deflector**, the **flux ratio** between them, and the source's magnification and flux.

The deflector luminosities are the emphasis here, because they are the quantity everything else in this folder has
been about. `likelihood_function.py` measured the two galaxies' bases coupling at 0.9877; this script computes the
number that coupling threatens.

__Contents__

- **Why Integrate A Basis:** `intensity` is not a luminosity.
- **Dataset & Fit:** Set up and fit, reusing this folder's composition.
- **Deflector Luminosities:** Summing an MGE to a total luminosity.
- **The Flux Ratio:** The measurement, and its error budget.
- **Source Magnification:** How much the pair magnifies the source.
- **Source Flux:** The intrinsic (unlensed) source flux.
- **Wrap Up:** Where to go next.

__Why Integrate A Basis__

A single `Sersic` has one `intensity`, and it is tempting to treat that as "how bright the galaxy is". It is not —
it is a surface brightness normalization, and two Sersics with equal `intensity` but different `effective_radius`
have very different total fluxes.

For an MGE the temptation does not even arise, because there are 20 intensities and no single one means anything.
The luminosity is the **integral of the whole basis**, which for Gaussians has a closed form:

    L = sum_k  2 * pi * sigma_k^2 / axis_ratio_k * intensity_k   / pixel_scale^2

summed over the basis's Gaussians. This is the same function
`multi_galaxy/features/scaling_relation/slam.py` uses to measure the luminosities its scaling relation needs — a
scaling relation is, after all, a statement about luminosities.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `mge` dataset — the co-dominant pair with twisted two-component light.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "mge"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/multi_gaussian_expansion/simulator.py",
        ],
        check=True,
    )

pixel_scale = 0.05

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

mask_radius = 3.0
main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=main_lens_centres,
    )
)

"""
__Fit__

The same composition as `fit.py` in this folder: an MGE per deflector, true mass profiles, shear and source.
"""
total_gaussians = 20

log10_sigma_list = np.linspace(np.log10(dataset.pixel_scales[0] / 10.0), np.log10(mask_radius), total_gaussians)


def mge_basis_from(centre):
    return al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Gaussian(
                centre=centre, ell_comps=(0.0, 0.0), sigma=10**log10_sigma
            )
            for log10_sigma in log10_sigma_list
        ]
    )


lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=mge_basis_from(centre=main_lens_centres[0]),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=mge_basis_from(centre=main_lens_centres[1]),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
The solved intensities live on the fit, not the model, so we convert the tracer's linear profiles into ordinary
ones carrying their solved values.
"""
tracer_solved = fit.tracer_linear_light_profiles_to_light_profiles

"""
__Deflector Luminosities__

Integrate each deflector's basis.
"""


def luminosity_from(galaxy, pixel_scale: float) -> float:
    """
    The total luminosity of a galaxy's MGE basis, summed over its Gaussians.

    Each Gaussian integrates analytically to `2 * pi * sigma^2 / axis_ratio * intensity`; dividing by
    `pixel_scale ** 2` converts from per-pixel to per-arcsecond-squared units.
    """
    return float(
        np.sum(
            [
                2
                * np.pi
                * gaussian.sigma**2
                / gaussian.axis_ratio()
                * gaussian.intensity
                for gaussian in galaxy.bulge.profile_list
            ]
        )
        / pixel_scale**2
    )


luminosities = [
    luminosity_from(galaxy=tracer_solved.galaxies[i], pixel_scale=pixel_scale)
    for i in range(len(main_lens_centres))
]

for i, luminosity in enumerate(luminosities):
    print(f"lens_{i} luminosity = {luminosity:.4e}")

"""
__The Flux Ratio__

The quantity a co-dominant pair is usually measured for: how much brighter one deflector is than the other.
"""
flux_ratio = luminosities[0] / luminosities[1]

print(f"\nlens_0 / lens_1 luminosity ratio = {flux_ratio:.4f}")

"""
__What The Error Budget Actually Is__

This number has no formal error bar attached, and its real uncertainty is not what you would guess.

**Not the linear solve.** The intensities are solved exactly for the given basis geometry, so they contribute no
statistical scatter of their own.

**Not the noise, mostly.** Re-simulating the dataset moves the log likelihood by a percent or two, and the
luminosities by less.

**The basis geometry, and the coupling between the two galaxies.** `likelihood_function.py` measures the two
deflectors' bases correlating at up to **0.9877**, which means the split of a given amount of light between
`lens_0` and `lens_1` is close to unconstrained for the most degenerate components. `fit.py` in the
`linear_light_profiles` folder measures the same effect end to end in the single-profile case: a wrong
`effective_radius` on *one* galaxy moved the other's intensity 5.2% and the ratio 33%.

So the honest error budget for a multi-galaxy flux ratio is dominated by how confident you are in each galaxy's
light-model geometry — its centre above all. That is why `multi_galaxy/slam.py`'s `light[1]` stage exists and why
it is given more live points than any other stage.

To put a real uncertainty on this number, propagate it from the posterior of a fitted model
(`modeling.py` in this folder), not from a single maximum-likelihood fit like this one.

__Source Magnification__

The magnification is the ratio of the lensed source's total flux to its unlensed flux. For a multi-galaxy lens it
is a property of the **combined** deflection field — neither galaxy has a magnification of its own, and the pair's
magnification is not the sum of what each would produce alone (`multi_galaxy/fit.py` measures that non-linearity
directly).
"""
source_solved = tracer_solved.galaxies[-1]

magnification_grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.03)

total_source_plane_flux = np.sum(
    source_solved.bulge.image_2d_from(grid=magnification_grid)
)

traced_grid_list = tracer_solved.traced_grid_2d_list_from(grid=magnification_grid)

total_image_plane_flux = np.sum(
    source_solved.bulge.image_2d_from(grid=traced_grid_list[-1])
)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"\nSource magnification = {source_magnification:.4f}")

"""
The area terms cancel because both fluxes are computed on grids of the same total area — see
`multi_galaxy/source_science.py`, which derives this calculation in full and also compares the pair's
magnification against what each deflector would produce alone (they do not add).

__Source Flux__

The source's intrinsic, unlensed flux — the quantity a source-galaxy science case wants, since the observed flux
is inflated by the magnification above. Note the source's `intensity` is itself a solved linear value here.
"""
print(f"Source solved intensity = {source_solved.bulge.intensity:.4e}")
print(f"Source total (unlensed) flux = {total_source_plane_flux:.4e}")

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/multi_gaussian_expansion/likelihood_function.py` — why the flux ratio's error budget is
   dominated by the two galaxies' coupling.
 - `multi_galaxy/features/scaling_relation` — where measured luminosities are *used*: a tier of faint galaxies
   whose masses are tied to a relation anchored on the brightest deflector.
 - `multi_galaxy/source_science.py` — the package's source-science walkthrough, with the full magnification and
   flux API.
 - `imaging/features/multi_gaussian_expansion/source_science.py` — the galaxy-scale version.
"""

"""
Fits: Multi Gaussian Expansion (Multi Galaxy)
=============================================

This script fits a multi-galaxy strong lens with an MGE per co-dominant deflector **without a non-linear search**,
so the basis can be built by hand and the MGE-versus-Sersic comparison run directly.

It is the script that produces the numbers quoted in `modeling.py` in this folder. Everything here is executed,
not asserted.

__Contents__

- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Basis:** Build an MGE basis by hand, without the `mge_model_from` helper.
- **Two Bases:** One per co-dominant deflector — the regime-specific part.
- **Fit:** Fit the tracer and read the log likelihood.
- **Comparison To A Sersic:** Why this dataset needs an MGE.
- **How Many Gaussians:** The diminishing-returns measurement.
- **Solved Intensities:** Reading each Gaussian's solved intensity.
- **Wrap Up:** Where to go next.

__Why Build The Basis By Hand__

`multi_galaxy/modeling.py` and this folder's `modeling.py` both build their bases through
`al.model_util.mge_model_from`, which hides a long composition API to keep those scripts readable. That is the
right default, but it makes the MGE look like a black box with a `total_gaussians` dial.

It is not. An MGE is a list of `Gaussian` light profiles sharing a centre, with `sigma` values on a log10 grid and
their intensities solved linearly. This script writes that list out.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `mge` dataset: the co-dominant pair whose light is twisted and two-component, so a Sersic cannot fit it.
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

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Mask & Over Sampling__

The standard 3.0" mask. The mask radius also sets the largest Gaussian `sigma`, so it is used twice below.

Over-sampling is centred on both deflectors: the smallest Gaussians have `sigma` = 0.01", a fifth of a pixel, and
are simply evaluated wrongly on the raw grid.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=main_lens_centres,
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Basis__

An MGE basis, written out. `total_gaussians` Gaussians share a centre, their `sigma` values spanning 0.01" to the
mask radius in equal log10 steps, and each is a **linear** profile so its `intensity` is solved rather than set.

The log10 spacing is the reason an MGE works with so few components: galaxy light falls off over decades in
radius, so equally-spaced widths would waste most of the basis on scales where nothing changes.
"""


def mge_basis_from(centre, total_gaussians: int = 20):
    """
    A basis of spherical linear Gaussians centred on `centre`, with sigmas log10-spaced across the mask.
    """
    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    return al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Gaussian(
                centre=centre,
                ell_comps=(0.0, 0.0),
                sigma=10**log10_sigma,
            )
            for log10_sigma in log10_sigma_list
        ]
    )


"""
__Two Bases__

This is the multi-galaxy part: **one basis per co-dominant deflector**, each centred on its own galaxy.

The mass profiles are the simulator's true values, so the only thing being varied in the comparisons below is the
light model.
"""
masses = [
    al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
    al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
]

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
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


def log_likelihood_from(lens_light_list):
    """
    Fit the dataset with the given light model for each deflector, holding mass, shear and source fixed.
    """
    galaxies = [
        al.Galaxy(redshift=0.5, bulge=lens_light_list[i], mass=masses[i])
        for i in range(len(lens_light_list))
    ]

    tracer = al.Tracer(galaxies=galaxies + [shear_galaxy, source])

    return al.FitImaging(dataset=dataset, tracer=tracer)


"""
__Fit__

Fit the pair with a 20-Gaussian MGE each.
"""
mge_light = [mge_basis_from(centre=centre) for centre in main_lens_centres]

fit = log_likelihood_from(mge_light)

aplt.subplot_fit_imaging(fit=fit)

print(f"MGE (20 Gaussians per deflector) log likelihood = {fit.log_likelihood}")

"""
__Comparison To A Sersic__

Now the same fit with a single linear `Sersic` per deflector instead.

The Sersic is given every advantage available: its centre, axis ratio, position angle, effective radius and Sersic
index are set to the true values of that galaxy's **bulge** component. No fitted Sersic would do better, because
these are the parameters a perfect fit to the inner light would recover.
"""
sersic_light = [
    al.lp_linear.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        effective_radius=0.3,
        sersic_index=4.0,
    ),
    al.lp_linear.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=120.0),
        effective_radius=0.25,
        sersic_index=4.0,
    ),
]

sersic_fit = log_likelihood_from(sersic_light)

aplt.subplot_fit_imaging(fit=sersic_fit)

print(f"Single Sersic per deflector log likelihood     = {sersic_fit.log_likelihood}")

"""
Measured values, for reference if you are reading rather than running:

    single linear Sersic per deflector : ~ -289,000
    MGE, 20 Gaussians per deflector    : ~   -4,490

Two significant figures, because `simulator.py` adds unseeded Poisson noise — re-simulate and every value moves by
a percent or two. The gap does not: roughly 284,000 in log likelihood, from changing only the light model. Look at the two residual maps plotted above:
the Sersic fit leaves the twist behind as a four-lobed pattern on each galaxy — the classic signature of fitting a
symmetric profile to a galaxy whose isophotes rotate with radius.

Neither reaches the truth tracer's ~+28,000, because these bases are spherical with fixed centres while the true
light is elliptical and offset. A fitted MGE, with free ellipticity per group and a free centre, closes most of
that gap; this script holds everything fixed so the comparison isolates one variable.
"""

"""
__How Many Gaussians__

The other question `modeling.py` answers with a number rather than a rule of thumb. Run the same fit at several
basis sizes.
"""
for total_gaussians in (10, 20, 30):

    light = [
        mge_basis_from(centre=centre, total_gaussians=total_gaussians)
        for centre in main_lens_centres
    ]

    print(
        f"MGE, {total_gaussians:2d} Gaussians per deflector : "
        f"log likelihood = {log_likelihood_from(light).log_likelihood}"
    )

"""
Measured: about -4,600 at 10, -4,490 at 20, -4,480 at 30. Ten gets most of the way, twenty captures the rest,
thirty adds nothing measurable while costing 50% more profile evaluations per likelihood call. (The step from 20
to 30 is smaller than the shift between two re-simulations of the dataset, which is the real definition of
"nothing measurable".)

At multi-galaxy scale there is a second reason not to over-provision, which
`likelihood_function.py` in this folder measures: every Gaussian added to one deflector's basis is another column
the *other* deflector's Gaussians can be degenerate with. Past the point where they improve the fit, extra
Gaussians make the two galaxies harder to tell apart.

__Solved Intensities__

Each Gaussian's solved `intensity` is available from the fit, as for any linear light profile.
"""
tracer_solved = fit.tracer_linear_light_profiles_to_light_profiles

for i in range(len(main_lens_centres)):
    intensities = [
        gaussian.intensity for gaussian in tracer_solved.galaxies[i].bulge.profile_list
    ]
    print(f"\nlens_{i} solved Gaussian intensities:")
    print(np.array(intensities))

"""
Two things are worth noticing in those arrays.

**They are all positive.** PyAutoLens uses a positive-only solver. A positive-negative solver on this system —
whose curvature matrix has a condition number of ~1e24 (see `likelihood_function.py`) — would produce the
alternating large positive and negative values known as MGE "ringing", and at multi-galaxy scale it could satisfy
the data with one galaxy positive and its neighbour compensating negatively.

**The intensity profile across sigma is smooth.** That smoothness is not imposed; it is what a real galaxy's light
looks like when decomposed this way. A basis whose solved intensities oscillate wildly between adjacent sigmas is
telling you the fit is compensating for something — a wrong centre, a missing component, or a neighbour's light.

__Wrap Up__

Where to go next:

 - `multi_galaxy/features/multi_gaussian_expansion/modeling.py` — the same model behind a search.
 - `multi_galaxy/features/multi_gaussian_expansion/likelihood_function.py` — the 41 x 41 curvature matrix, and
   how strongly the two deflectors' bases couple.
 - `multi_galaxy/features/linear_light_profiles/fit.py` — the single-profile version of this script.
 - `imaging/features/multi_gaussian_expansion/fit.py` — the galaxy-scale version, including basis visualization.
"""

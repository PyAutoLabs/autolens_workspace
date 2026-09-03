"""
SLaM (Source, Light and Mass): Extra Galaxies (Multi Galaxy)
============================================================

The SLaM pipeline for a multi-galaxy lens with a tier of perturbing extra galaxies alongside the co-dominant
deflectors.

This script documents only how it differs from `multi_galaxy/slam.py`, the multi-galaxy SLaM baseline. Read that
first, and `multi_galaxy/features/extra_galaxies/modeling.py` for what the tier is and why its centres are fixed
and its Einstein radii capped.

__Contents__

- **What Changes:** The differences from `multi_galaxy/slam.py`.
- **The Tier Through The Stages:** Where the perturbers are free, where they are fixed, and why.
- **Source LP Pipeline:** The tier enters, with restricted freedom.
- **Source Pix Pipeline 1 & 2:** The tier carried forward as a model, then as an instance.
- **Light LP Pipeline:** Fresh light for the co-dominant pair; the tier held still.
- **Mass Total Pipeline:** Co-dominant masses promoted to `PowerLaw`; the tier stays isothermal.
- **Dataset, Centres, Mask:** Set up.
- **SLaM Pipeline:** Run the five stages in order.

__Prerequisites__

Read `guides/modeling/slam_start_here` first: it describes what the five SLaM stages are and why they are chained
in this order. This script documents only what differs.

__What Changes__

Two things:

1. **An `extra_galaxies` collection is passed alongside `galaxies` in every stage.** `AnalysisImaging` appends it
   to the tracer it builds from each instance, so the perturbers contribute to the summed deflection field with no
   further wiring.

2. **The tier's freedom is scheduled across the stages** rather than being constant. This is the substance of the
   script and is described below.

Everything else — the loop over `lens_i`, the `shear_galaxy`, the mass-centre anchoring, the live-point scaling —
is the baseline's, unchanged.

__The Tier Through The Stages__

A perturber tier is not simply "more galaxies in the model". It is a set of galaxies you have decided *not* to
measure, included because their deflections reach the arcs. The pipeline reflects that decision by giving them
progressively less freedom as the stages that actually matter arrive:

 - `source_lp[1]`: the tier is a **model** — free `einstein_radius` (capped) and free light amplitude, fixed
   centres. This is where the data says how much perturbation it needs.
 - `source_pix[1]`: still a model, so the tier can adjust as the source becomes pixelized.
 - `source_pix[2]`: an **instance**. The source model is being finalized and the tier should not be moving
   underneath it.
 - `light[1]`: an instance. This stage measures the co-dominant pair's light; a perturber free to brighten here
   would eat into that measurement.
 - `mass_total[1]`: an instance, while the co-dominant masses are promoted to `PowerLaw`.

__Why The Tier Is Never Promoted__

The temptation, especially once `source_lp[1]` reports a well-constrained `einstein_radius` for a perturber, is
to free its centre or let it become a `PowerLaw` too. Do not, without deciding it is co-dominant and moving it
into `main_lens_centres.json` properly.

The reason is the one `modeling.py` in this folder establishes: promoting a perturber near a co-dominant deflector
is the **subtle** tier error. The fit still converges and the residuals still look fine — what you lose is
identifiability, because a free-centre perturber close to a co-equal galaxy is degenerate with that galaxy's mass.
The pipeline will not warn you. It will simply return a mass split that is not a measurement.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


def n_main_from(result) -> int:
    """
    The number of co-dominant deflectors in a result's model. Identical to the helper in `multi_galaxy/slam.py`.

    Note this counts only `lens_` entries in `galaxies`. The extra galaxies live in their own `extra_galaxies`
    collection and are never counted here — which is the point of the tier.
    """
    return sum(1 for key in vars(result.instance.galaxies) if key.startswith("lens_"))


def extra_galaxies_model_from(
    extra_galaxies_centres,
    redshift_lens: float,
    upper_einstein_radius: float = 0.3,
):
    """
    The perturber tier as a free model: fixed centres, capped Einstein radii, `ExponentialSph` light.

    Identical composition to `multi_galaxy/features/extra_galaxies/modeling.py`, factored out so the two stages
    that use it as a model cannot drift apart from each other.
    """
    extra_galaxies_list = []

    for centre in extra_galaxies_centres:

        light = af.Model(al.lp.ExponentialSph)
        light.centre = centre

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = centre
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0, upper_limit=upper_einstein_radius
        )

        extra_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, light=light, mass=mass)
        )

    return af.Collection(extra_galaxies_list)


"""
__SOURCE LP PIPELINE__

Identical to `multi_galaxy/slam.py` apart from the `extra_galaxies` collection, which enters here as a free model.

This is the only stage where the tier is genuinely being fitted, so it is the stage whose result to inspect if you
want to know whether the perturbers were needed at all. An `einstein_radius` posterior that runs up against the
capped upper limit is a signal that a galaxy is not a perturber and should be reconsidered as co-dominant; one
that sits against zero is a galaxy the data does not need.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    extra_galaxies_centres,
    redshift_lens: float,
    redshift_source: float,
    upper_einstein_radius: float = 3.0,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = {}

    for i, centre in enumerate(main_lens_centres):

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
            sigma_min=dataset.pixel_scales[0] / 10.0,
        )

        mass = af.Model(al.mp.Isothermal)
        mass.centre = (centre[0], centre[1])
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0, upper_limit=upper_einstein_radius
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=bulge,
            disk=None,
            point=None,
            mass=mass,
        )

    shear_galaxy = af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        shear=af.Model(al.mp.ExternalShear),
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=shear_galaxy,
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        ),
        extra_galaxies=extra_galaxies_model_from(
            extra_galaxies_centres=extra_galaxies_centres,
            redshift_lens=redshift_lens,
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=150 + 50 * len(lens_dict),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to `multi_galaxy/slam.py`, with the tier carried forward as a **model** via
`source_lp_result.model.extra_galaxies`.

Carrying the model rather than re-declaring it means the priors are the ones `source_lp[1]` narrowed, not the wide
ones it started from. The centres stay fixed because they were fixed when the model was composed — fixing is a
property of the model, and it travels with it.

__Adapt Image S/N Cap__

The source adapt image is capped at a signal-to-noise of 3.0 before it is used by the adaptive
image-mesh and the adaptive regularization. Without the cap the brightest peak dominates the
weights (they scale as a power of the adapt image), so fainter multiply-imaged features get too
few source pixels and too little regularization weight. Capping makes every feature above S/N 3.0
count equally. The cap is applied to an explicit copy so the raw S/N image is untouched.
"""


def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    lens_dict = {}

    for i in range(n_main_from(source_lp_result)):

        lens_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        lens_model = getattr(source_lp_result.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.Isothermal),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=lens_instance.bulge,
            disk=lens_instance.disk,
            point=lens_instance.point,
            mass=mass,
        )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_lp_result.model.galaxies.shear_galaxy,
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
        extra_galaxies=source_lp_result.model.extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150 + 50 * (n_main_from(source_lp_result) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `multi_galaxy/slam.py`, with the tier fixed as an **instance** from `source_pix[1]`.

This is where the tier stops being fitted. The source mesh is being finalized against the improved adapt image,
and a perturber still free to adjust its Einstein radius would be adjusting the deflection field the mesh is
adapting to.
"""


def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    mesh,
    regularization,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    lens_dict = {}

    for i in range(n_main_from(source_pix_result_1)):

        lp_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        pix_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_instance.redshift,
            bulge=lp_instance.bulge,
            disk=lp_instance.disk,
            point=lp_instance.point,
            mass=pix_instance.mass,
        )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_pix_result_1.instance.galaxies.shear_galaxy,
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
        ),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LIGHT LP PIPELINE__

Identical to `multi_galaxy/slam.py`, with the tier held as an instance.

The baseline explains why this stage carries the flux-ratio measurement for the co-dominant pair. The tier being
fixed here is part of protecting that measurement: a perturber whose `ExponentialSph` amplitude were free would be
competing for flux in the same image region as the pair's MGEs, and — per
`multi_galaxy/features/linear_light_profiles/likelihood_function.py` — the linear solver distributes contested
flux between whichever profiles overlap.
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

    lens_dict = {}

    for i in range(n_main_from(source_result_for_lens)):

        lens_instance = getattr(source_result_for_lens.instance.galaxies, f"lens_{i}")

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=tuple(lens_instance.mass.centre),
            sigma_min=dataset.pixel_scales[0] / 10.0,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=bulge,
            disk=None,
            point=None,
            mass=lens_instance.mass,
        )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.instance.galaxies.shear_galaxy,
            source=source,
        ),
        extra_galaxies=source_result_for_lens.instance.extra_galaxies,
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150 + 100 * (n_main_from(source_result_for_lens) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `multi_galaxy/slam.py`, with the tier held as an instance and left isothermal.

The co-dominant deflectors are promoted to `PowerLaw`; the perturbers are not. Freeing a density slope on a galaxy
whose Einstein radius is ~0.1" would be fitting a parameter the data cannot constrain, and — because the
perturbers sit close to the co-dominant pair — the unconstrained slope would be absorbed into the pair's masses.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_result_for_source.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    lens_dict = {}

    for i in range(n_main_from(source_result_for_lens)):

        lens_model = getattr(source_result_for_lens.model.galaxies, f"lens_{i}")
        light_instance = getattr(light_result.instance.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            bulge=light_instance.bulge,
            disk=light_instance.disk,
            point=light_instance.point,
            mass=mass,
        )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.model.galaxies.shear_galaxy,
            source=source,
        ),
        extra_galaxies=light_result.instance.extra_galaxies,
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150 + 100 * (n_main_from(source_result_for_lens) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

The `extra_galaxies` dataset simulated by this folder's `simulator.py`: the co-dominant pair plus two perturbers
carrying both light and mass.

Note this is a different dataset from the `simple` one used by `multi_galaxy/slam.py`. `simple` contains a
*massless* contaminant, handled by noise scaling; this one contains perturbers with mass, which noise scaling
cannot reach — see `multi_galaxy/features/README.md` for that division of labour.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/features/extra_galaxies/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.05,
)

"""
__Centres__

Two catalogues, and which file a galaxy's centre appears in *is* its tier assignment:

 - `main_lens_centres.json` — the co-dominant deflectors, modeled freely.
 - `extra_galaxies_centres.json` — the perturbers, with fixed centres and capped Einstein radii.

Moving a line from one file to the other is how you change a galaxy's tier. That is the whole interface, and it is
the decision `modeling.py` in this folder is about.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

"""
__Mask & Over Sampling__

The standard 3.0" mask, with over-sampling centred on the co-dominant deflectors **and** the perturbers — every
galaxy with a light profile needs its centre evaluated accurately, tier notwithstanding.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres) + extra_galaxies_centres.in_list,
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy") / "features" / "extra_galaxies" / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Mesh Shape__
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh_init = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape)
regularization_init = al.reg.Adapt

mesh = af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape)
regularization = al.reg.Adapt

"""
__SLaM Pipeline__
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    main_lens_centres=main_lens_centres,
    extra_galaxies_centres=extra_galaxies_centres,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_init=mesh_init,
    regularization_init=regularization_init,
)

"""
__Adaptive Pixelization Over-Sampling__

From SOURCE PIX PIPELINE 2 onwards the pixelization grid is over-sampled adaptively. The source's
signal-to-noise map from the previous pixelized fit (the same map that becomes the adapt image, read before
the S/N 3.0 cap is applied) is thresholded at S/N 3.0: pixels above it, the bright lensed source, use a
sub-size of 4 and every other pixel uses a sub-size of 2. This concentrates the extra over-sampling where the
source is bright and the pixelization gains the most accuracy from it, and keeps the rest of the mask cheap.

The map returned by `galaxy_name_image_dict_via_result_from` is already signal divided by noise, so it is
thresholded directly.

SOURCE PIX PIPELINE 1 keeps the dataset's default uniform sub-size. Its adapt image comes from the parametric
source fit of the SOURCE LP PIPELINE, which does not yet trace the lensed source well enough to steer
over-sampling.
"""
signal_to_noise_threshold = 3.0

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)
source_image_raw = galaxy_image_name_dict["('galaxies', 'source')"]

over_sample_size_pixelization = al.Array2D(
    values=np.where(source_image_raw > signal_to_noise_threshold, 4, 2),
    mask=dataset.mask,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=over_sample_size_pixelization,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=mesh,
    regularization=regularization,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
__Result__

The baseline's checks apply. Add the tier check, which is best made on `source_lp[1]` rather than here, since that
is the only stage where the perturbers were free:

 - An `einstein_radius` posterior pressed against its upper cap means that galaxy is not behaving like a
   perturber. Consider moving it to `main_lens_centres.json` and re-running — but do so deliberately, as a
   change of tier, not by loosening the cap.
 - A posterior against zero means the data does not need that galaxy at all.
 - A well-constrained value comfortably inside the cap is the tier working as intended.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/slam.py` — the baseline this pipeline adds a tier to.
 - `multi_galaxy/features/extra_galaxies/modeling.py` — the tier decision, in full.
 - `multi_galaxy/features/scaling_relation` — the other tier: many faint galaxies tied to a luminosity relation
   rather than freed individually.
 - `imaging/features/extra_galaxies/slam.py` — the galaxy-scale version.
"""

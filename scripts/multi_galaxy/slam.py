"""
SLaM (Source, Light and Mass): Multi Galaxy
===========================================

This script is the multi-galaxy regime's SLaM pipeline: the production path for fitting a lens with two or more
co-dominant deflectors, as opposed to the single-search fits of `multi_galaxy/modeling.py`.

A full overview of what SLaM is and why it is structured as five chained searches is in
`guides/modeling/slam_start_here`. **Read that first.** This script documents only what changes when the lens is
a multi-galaxy system, and is the baseline that the feature pipelines under `multi_galaxy/features/*/slam.py`
diff against.

__Contents__

- **Why A Regime Baseline Exists:** Why this script exists when `slam_start_here` already does.
- **What Changes:** The four differences from `slam_start_here`, listed up front.
- **Source LP Pipeline:** Light and mass per deflector, and the source, fitted for the first time.
- **Source Pix Pipeline 1:** A pixelized source used to build a high-quality adapt image.
- **Source Pix Pipeline 2:** The final pixelized source, on the improved adapt image.
- **Light LP Pipeline:** A fresh, accurate light model per deflector — where the flux ratio is measured.
- **Mass Total Pipeline:** Each deflector's mass promoted to a `PowerLaw`.
- **Dataset:** Load the `simple` multi-galaxy dataset.
- **Centres:** Load the deflector centres that drive the loop.
- **Mask & Over Sampling:** The standard 3.0" mask, over-sampled at every deflector.
- **SLaM Pipeline:** Run the five stages in order.

__Prerequisites__

Read `guides/modeling/slam_start_here` first: it describes what the five SLaM stages are and why they are chained
in this order. This script documents only what differs.

__Why A Regime Baseline Exists__

`imaging/` has no top-level `slam.py`: its feature pipelines diff directly against
`guides/modeling/slam_start_here`, because at galaxy scale the composition in that guide *is* the composition you
want. That is not true here. Every stage of a multi-galaxy pipeline builds its lens entries in a loop, carries a
separate shear galaxy, and scales its live-point count with the number of deflectors. Repeating those changes in
every feature's `slam.py` would mean re-deriving them five times over, so they live here once. `group/slam.py`
plays the same role for the group package.

__What Changes__

Four differences from `slam_start_here.py`, and nothing else:

1. **One `lens_i` entry per deflector, built in a loop** over the centres in `main_lens_centres.json`, rather than
   a single `lens`. Every stage does this, and every stage recovers the deflector count from the previous result
   rather than being told it, so the pipeline runs unchanged on a pair, a triple or more.

2. **The external shear is its own `shear_galaxy`** at the system centre (0.0", 0.0"), not an attribute of a
   deflector. The reasoning is in `multi_galaxy/modeling.py`: the shear describes the tidal field of everything
   outside the system, so attaching it to one of two co-dominant galaxies would misrepresent it. Practically, this
   means each stage chains `shear_galaxy` as its own model or instance.

3. **Mass centres are anchored, then released.** `source_lp[1]` fixes each deflector's mass centre to its light
   centre, because with several deflectors a free centre at this stage has no idea which galaxy it belongs to.
   `source_pix[1]` then unfixes them via `unfix_mass_centre=True`, once the mass model is good enough for the
   centres to be meaningfully constrained.

4. **Live points scale with the deflector count.** Each additional co-dominant deflector adds a full light and
   mass model, so a fixed `n_live` that works for one lens galaxy under-samples a pair. The formulas below are
   deliberately linear in the deflector count.

__What Does Not Change__

The stage structure, the adapt-image logic, the positions likelihood, and the reasoning for each chaining decision
are all identical to `slam_start_here.py`. Where this script says "identical to `slam_start_here`", it means the
code is the same modulo the loop.

__The Measurement This Pipeline Protects__

Worth keeping in view while reading. The data constrains the *total* deflection of a multi-galaxy lens well and the
*split* between the deflectors much less well (`multi_galaxy/modeling.py`). The split is what the science wants —
each galaxy's mass, and the ratio between them. That is why the light stages matter so much here: residual
unmodelled flux from one galaxy is absorbed asymmetrically by the two light models, and the resulting bias lands
directly on the quantity being measured. `light[1]` is not cosmetic polish in this regime.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


def n_main_from(result) -> int:
    """
    The number of co-dominant deflectors in a result's model.

    Every stage recovers this from the previous result rather than closing over a module-level constant, so the
    pipeline runs unchanged on any number of deflectors. The `lens_` prefix is the convention the whole
    multi-galaxy package uses; `shear_galaxy` and `source` do not match it and are therefore not counted.
    """
    return sum(1 for key in vars(result.instance.galaxies) if key.startswith("lens_"))


"""
__SOURCE LP PIPELINE__

Identical in purpose to `source_lp` in `slam_start_here.py`: one search initializing a robust model of the source's
light, the deflectors' light, and their mass.

 - Each deflector's light is an MGE with 2 x 20 Gaussians, centred on its known position.
 - Each deflector's total mass distribution is an `Isothermal`, with its centre fixed to that position.
 - The system carries one `ExternalShear`, in its own galaxy.
 - The source's light is an MGE with 1 x 20 Gaussians.

__Why The Mass Centres Are Fixed Here__

`slam_start_here.py` gives its single lens a free-centre `Isothermal`, whose default prior sits at the origin —
which is where that lens is. Neither deflector of a multi-galaxy lens is at the origin, and a free centre in a
model with two nearby mass profiles is worse than merely unconstrained: the two profiles can swap, or both drift
onto the brighter galaxy, and the search will happily report a good likelihood from either. Fixing the centres to
the light for this initialization search removes that failure mode entirely. `source_pix[1]` frees them again.

__Why The Einstein Radius Prior Is Capped__

Each deflector's `einstein_radius` gets an explicit `UniformPrior` with an upper limit, defaulting to 3.0". Without
it, one galaxy can take an Einstein radius large enough to reproduce the whole ring on its own while the other
collapses to zero — a fit that looks acceptable and has the mass split completely wrong. This is the mass-split
degeneracy at its most destructive, and a bounded prior is the cheapest guard against it.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
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

Identical to `slam_start_here.py`: a pixelization whose purpose is to produce a high-quality adapt image for
search 2, with the deflectors' light fixed and their mass free.

__Positions__

Image positions prevent unphysical source reconstructions and are computed automatically from the SOURCE LP
result rather than being input by hand. This automation matters more here than at galaxy scale: a multi-galaxy
lens produces more multiple images in more complex configurations, and identifying them by eye is correspondingly
harder.

__Mass Centres Released__

`unfix_mass_centre=True` converts each mass centre from the fixed value set in `source_lp[1]` into a free
parameter with a prior around it. The mass model is now good enough that the arcs can say something about where
each deflector's mass actually is — which is exactly the measurement that produced the mass/light offsets Shu et
al. (2016) reported for SDSS J1011+0143, the system `multi_galaxy/simulator.py` is modelled on. Keeping the
centres fixed all the way through would make that measurement impossible by construction.
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

Identical to `slam_start_here.py`: the final pixelized source, fitted with the improved adapt image from search 1,
with every deflector's light and mass fixed as instances.

Because everything except the source is fixed, this stage's cost does not grow with the number of deflectors, and
`n_live` stays at the `slam_start_here` value.
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

Identical to `slam_start_here.py`: a fresh, free MGE for each deflector, with mass and source fixed from the
SOURCE PIX pipeline, reusing its adapt images.

__Why This Stage Carries The Measurement__

At galaxy scale this stage is described as producing "an accurate model of the lens galaxy's light" — desirable,
but the mass model is what the science usually wants. In the multi-galaxy regime it is load-bearing, for a reason
specific to having two co-dominant galaxies whose light blends:

Unmodelled flux does not vanish. It is absorbed by whichever component can absorb it, and with two free light
models sitting on top of each other that absorption is **asymmetric** — one galaxy takes more of the other's
residual than it should. The quantity that distorts is the ratio between the two galaxies' luminosities, which is
frequently the measurement (it is what a scaling relation is calibrated against, and what a stellar-mass estimate
per deflector depends on).

So the fresh MGE per deflector here is not polish. It is the stage where the two galaxies' light is finally
separated with the mass and source held still, and it is worth giving it enough live points to do that properly —
hence the steeper scaling with deflector count below than any other stage uses.
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

Identical to `slam_start_here.py`: each deflector's mass is promoted from `Isothermal` to `PowerLaw`, with priors
initialized from the SOURCE PIX result, the light fixed from `light[1]` and the source fixed from `source_pix[2]`.

__What Freeing The Slopes Costs Here__

At galaxy scale, promoting to a `PowerLaw` adds one parameter and measures the density slope. With N deflectors it
adds N parameters, and each new slope is degenerate with that galaxy's Einstein radius, which is already degenerate
with the other galaxy's. Expect the slopes to be considerably less well constrained than the single-lens examples
would suggest, and check their posteriors against each other rather than reading each in isolation.

This is not a reason to skip the stage — the total mass profile really is not isothermal — but it is a reason to
run this pipeline through to the end rather than quoting the `source_pix` Einstein radii as final.

__Positions__

Computed from `source_pix[2]`, whose pixelized source reconstruction gives more precise multiple-image positions
than the SOURCE LP result.
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

The `simple` multi-galaxy dataset — the same pair of co-dominant deflectors fitted by
`multi_galaxy/modeling.py`, so the pipeline's result can be compared directly against that single-search fit.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

pixel_scale = 0.05

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=pixel_scale,
)

"""
__Extra Galaxies Noise Scaling__

The `simple` dataset contains a faint contaminating galaxy, whose light is scaled out of the fit before masking.
This is the same step `multi_galaxy/modeling.py` performs and explains in full; it is not a SLaM feature, but
skipping it here would leave the contaminant's flux to be absorbed by the light stages.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__

The centres of the co-dominant deflectors, which drive the loop in every stage. There is no `extra_galaxies` or
`scaling_galaxies` catalogue in the multi-galaxy regime — every deflector is a main lens galaxy. The feature
pipelines that add those tiers load their own catalogues on top of this one.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask used throughout the multi-galaxy package, sized by the *combined* Einstein radius (~1.8")
rather than either galaxy's individually.

Unlike the scaling-relation pipeline, one mask is enough here: there is no tier of distant galaxies sitting outside
it whose luminosities have to be measured.

The adaptive over-sampling scheme is centred on **every** deflector, since each has a steep central light profile.
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
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__

The settings applied to every search, including the output path.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy") / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

Both deflectors are at the same redshift, so ray tracing is single-plane. If yours are not, give each `lens_i` its
own redshift and PyAutoLens performs multi-plane tracing without any other change to this pipeline.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Mesh Shape__

The pixelization mesh and regularization used by the SOURCE PIX searches, discussed in
`imaging/features/pixelization`.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh_init = af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape)
regularization_init = al.reg.Adapt

mesh = af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape)
regularization = al.reg.Adapt

"""
__SLaM Pipeline__

The five stages, in order. Each consumes the results of the ones before it.
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    main_lens_centres=main_lens_centres,
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

`mass_result` holds the final model. Two checks are worth making that a galaxy-scale pipeline does not need:

 - **The mass split.** Compare the deflectors' Einstein radii and their errors. If one is large with a small error
   and the other is small with a large error, look at whether the prior's upper limit is doing the work — that is
   the failure mode `source_lp[1]`'s capped prior guards against, and it can survive to the end of the pipeline.
 - **The mass centres against the light centres.** They were freed in `source_pix[1]` and are free here. A
   significant offset is a real, publishable measurement in an interacting pair; a large offset with large errors
   is usually the mass split leaking into position.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `guides/modeling/slam_start_here` — what each stage is for, in full.
 - `multi_galaxy/modeling.py` — the single-search fit of this same lens, to compare against.
 - `multi_galaxy/features/*/slam.py` — the feature pipelines, each of which documents only its difference from
   this script.
"""

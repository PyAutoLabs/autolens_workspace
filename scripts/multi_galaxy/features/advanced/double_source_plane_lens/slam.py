"""
SLaM (Source, Light and Mass): DSPL (Multi Galaxy)
==================================================

The SLaM pipeline for a multi-galaxy double source-plane lens.

Read `multi_galaxy/slam.py` first — the multi-galaxy SLaM baseline — and `guides/modeling/slam_start_here`
before it. This script documents only what a second source plane changes.

__What Changes__

The baseline's five stages become six, because the second source has to be introduced before anything can be
fitted with it in place:

1. **SOURCE LP 1** — deflectors and `source_0` only, on a tight mask that excludes the second ring. Identical in
   purpose to the baseline's `source_lp[1]`.
2. **SOURCE LP 2** — the full mask, adding `source_1` and `source_0`'s mass, with the deflectors held from
   stage 1. This stage has no baseline equivalent; it exists because the second ring cannot be found while the
   first is still being solved.
3. **SOURCE PIX 1 & 2** — the pixelized `source_0`, as in the baseline. `source_1` stays parametric throughout:
   it is faint and compact, and a pixelization on it would have more freedom than the data supports.
4. **LIGHT LP** — a fresh MGE per deflector, as in the baseline.
5. **MASS TOTAL** — each deflector's mass promoted to a `PowerLaw`, **with both source planes in place**. This
   is the stage where the second source plane earns its keep for the mass split: it is the only stage that fits
   the deflectors' mass against images from both rings at once.

The stage functions are copied from `multi_galaxy/slam.py` rather than imported — that script is a script, so
importing it would execute its whole pipeline as a side effect.

__Why The Masking Changes Between Stages 1 And 2__

`chaining.py` in this folder explains it in full: with the second ring outside the mask, its pixels contribute
nothing to the likelihood, so `source_1` cannot bias the deflectors by leaving residuals they try to absorb. The
small mask is the point of the first stage, not an approximation to be tolerated.

__Contents__

- **Helpers:** The deflector-count helper the baseline uses.
- **Source LP Pipeline 1 & 2:** The two source stages.
- **Source Pix Pipeline 1 & 2:** The pixelized `source_0`.
- **Light LP Pipeline:** A fresh MGE per deflector.
- **Mass Total Pipeline:** Each deflector's mass promoted to a `PowerLaw`.
- **Dataset, Centres, Masks:** Set up, including both masks.
- **Mesh Shape:** The pixelization classes.
- **SLaM Pipeline:** Run the six stages in order.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


def n_main_from(result) -> int:
    """
    The number of co-dominant deflectors in a result's model. Identical to the helper in `multi_galaxy/slam.py`.

    `source_0`, `source_1` and `shear_galaxy` do not match the `lens_` prefix and are therefore not counted.
    """
    return sum(1 for key in vars(result.instance.galaxies) if key.startswith("lens_"))


"""
__SOURCE LP PIPELINE 1__

The baseline's `source_lp[1]`, restricted to the first source plane and run on the tight mask.

Each deflector's light is an MGE, its mass an `Isothermal` with its centre fixed to its light centre and its
Einstein radius capped — the same guards against the mass-split degeneracy the baseline uses.

`source_0` has light but no mass at this stage. Its mass is constrained by where it places `source_1`'s images,
and those are outside this mask.
"""


def source_lp_1(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    redshift_lens: float,
    redshift_source_0: float,
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
            source_0=af.Model(
                al.Galaxy, redshift=redshift_source_0, bulge=source_bulge
            ),
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
__SOURCE LP PIPELINE 2__

No baseline equivalent. Run on the full mask, this stage introduces `source_1` and `source_0`'s mass, with every
deflector held as an instance from stage 1.

Holding the deflectors fixed keeps the stage cheap and, more importantly, keeps it honest: the second ring's
position is being used to constrain `source_0`'s mass, not to quietly re-solve the deflectors before the model
is ready for that.
"""


def source_lp_2(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_lp_result_1: af.Result,
    redshift_source_1: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = {}

    for i in range(n_main_from(source_lp_result_1)):
        lens_dict[f"lens_{i}"] = getattr(
            source_lp_result_1.instance.galaxies, f"lens_{i}"
        )

    source_0_instance = source_lp_result_1.instance.galaxies.source_0

    source_0 = af.Model(
        al.Galaxy,
        redshift=source_0_instance.redshift,
        bulge=source_0_instance.bulge,
        mass=af.Model(al.mp.IsothermalSph),
    )

    source_1_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_lp_result_1.instance.galaxies.shear_galaxy,
            source_0=source_0,
            source_1=af.Model(
                al.Galaxy, redshift=redshift_source_1, bulge=source_1_bulge
            ),
        ),
    )

    search = af.Nautilus(
        name="source_lp[2]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

The baseline's `source_pix[1]`, applied to `source_0` only. Its purpose is to produce a better adapt image than
the parametric source can.

`source_1` is carried through as an instance. The deflectors' mass centres are released here, as in the
baseline, now that both rings are in the model.

__Adapt Image S/N Cap__

The source adapt images are capped at a signal-to-noise of 3.0 before they are used by the adaptive
image-mesh and the adaptive regularization. Without the cap the brightest peak dominates the
weights (they scale as a power of the adapt image), so fainter multiply-imaged features get too
few source pixels and too little regularization weight. Capping makes every feature above S/N 3.0
count equally. The cap is applied to an explicit copy so the raw S/N image is untouched.
"""


def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result_2: af.Result,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result_2
    )

    # Cap the source adapt images at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_0_adapt_image = galaxy_image_name_dict["('galaxies', 'source_0')"].copy()
    source_0_adapt_image[source_0_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_0')"] = source_0_adapt_image

    source_1_adapt_image = galaxy_image_name_dict["('galaxies', 'source_1')"].copy()
    source_1_adapt_image[source_1_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_1')"] = source_1_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

    lens_dict = {}

    for i in range(n_main_from(source_lp_result_2)):

        lens_instance = getattr(source_lp_result_2.instance.galaxies, f"lens_{i}")
        lens_model = getattr(source_lp_result_2.model.galaxies, f"lens_{i}")

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

    source_0_instance = source_lp_result_2.instance.galaxies.source_0

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_lp_result_2.model.galaxies.shear_galaxy,
            source_0=af.Model(
                al.Galaxy,
                redshift=source_0_instance.redshift,
                mass=source_0_instance.mass,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
            source_1=source_lp_result_2.instance.galaxies.source_1,
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150 + 50 * (n_main_from(source_lp_result_2) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

The baseline's `source_pix[2]`: the final pixelized `source_0`, fitted with the improved adapt image from
`source_pix[1]`, with every deflector's light and mass fixed as instances.
"""


def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result_2: af.Result,
    source_pix_result_1: af.Result,
    mesh,
    regularization,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    # Cap the source adapt images at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_0_adapt_image = galaxy_image_name_dict["('galaxies', 'source_0')"].copy()
    source_0_adapt_image[source_0_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_0')"] = source_0_adapt_image

    source_1_adapt_image = galaxy_image_name_dict["('galaxies', 'source_1')"].copy()
    source_1_adapt_image[source_1_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_1')"] = source_1_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    lens_dict = {}

    for i in range(n_main_from(source_pix_result_1)):

        lp_instance = getattr(source_lp_result_2.instance.galaxies, f"lens_{i}")
        pix_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_instance.redshift,
            bulge=lp_instance.bulge,
            disk=lp_instance.disk,
            point=lp_instance.point,
            mass=pix_instance.mass,
        )

    source_0_instance = source_pix_result_1.instance.galaxies.source_0

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_pix_result_1.instance.galaxies.shear_galaxy,
            source_0=af.Model(
                al.Galaxy,
                redshift=source_0_instance.redshift,
                mass=source_0_instance.mass,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
            source_1=source_pix_result_1.instance.galaxies.source_1,
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

The baseline's `light[1]`: a fresh, free MGE for each deflector, with mass and both sources fixed, reusing the
adapt images.
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

    # Cap the source adapt images at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_0_adapt_image = galaxy_image_name_dict["('galaxies', 'source_0')"].copy()
    source_0_adapt_image[source_0_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_0')"] = source_0_adapt_image

    source_1_adapt_image = galaxy_image_name_dict["('galaxies', 'source_1')"].copy()
    source_1_adapt_image[source_1_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_1')"] = source_1_adapt_image

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

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.instance.galaxies.shear_galaxy,
            source_0=source_result_for_source.instance.galaxies.source_0,
            source_1=source_result_for_source.instance.galaxies.source_1,
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

The baseline's `mass_total[1]`: each deflector's mass promoted from `Isothermal` to `PowerLaw`, with the light
fixed from `light[1]` and both sources fixed from the SOURCE PIX result.

__Where The Second Source Plane Pays Off__

This is the stage the DSPL dataset exists for. Both rings are in the model, both are inside the mask, and the
deflectors' mass is free. Their mass split is therefore being fitted against images at two sets of sky
positions rather than one, which is what constrains the spatial structure of the deflection field — see
`modeling.py`'s `__Why Two Source Planes Help Here__`.

Everything before this stage is the machinery required to get here honestly.
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

    # Cap the source adapt images at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_0_adapt_image = galaxy_image_name_dict["('galaxies', 'source_0')"].copy()
    source_0_adapt_image[source_0_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_0')"] = source_0_adapt_image

    source_1_adapt_image = galaxy_image_name_dict["('galaxies', 'source_1')"].copy()
    source_1_adapt_image[source_1_adapt_image > adapt_image_snr_cap] = (
        adapt_image_snr_cap
    )
    galaxy_image_name_dict["('galaxies', 'source_1')"] = source_1_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

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

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.model.galaxies.shear_galaxy,
            source_0=source_result_for_source.instance.galaxies.source_0,
            source_1=source_result_for_source.instance.galaxies.source_1,
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

The `dspl` multi-galaxy dataset.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "dspl"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

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

pixel_scale = 0.05


def dataset_from(mask_radius, main_lens_centres):
    """
    Load the dataset with a given mask radius. Called twice, because stage 1 uses a tighter mask than every
    stage after it.
    """
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        psf_path=dataset_path / "psf.fits",
        pixel_scales=pixel_scale,
    )

    dataset = dataset.apply_mask(
        mask=al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
        )
    )

    return dataset.apply_over_sampling(
        over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[4, 2, 2],
            radial_list=[0.3, 0.6],
            centre_list=list(main_lens_centres),
        )
    )


"""
__Centres__

The centres of the co-dominant deflectors, which drive the loop in every stage.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Masks__

Two masks, for the reason given at the top of this script: the first stage excludes the second ring so it cannot
bias the deflectors, and every stage after it includes both.
"""
mask_radius_1 = 1.6
mask_radius_2 = 3.0

dataset_1 = dataset_from(mask_radius_1, main_lens_centres)
dataset_2 = dataset_from(mask_radius_2, main_lens_centres)

aplt.subplot_imaging_dataset(dataset=dataset_2)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy")
    / "features"
    / "advanced"
    / "double_source_plane_lens"
    / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

Three of them, which is what makes the tracing multi-plane.
"""
redshift_lens = 0.5
redshift_source_0 = 1.0
redshift_source_1 = 2.0

"""
__Mesh Shape__

The pixelization classes used by the SOURCE PIX stages, as in `multi_galaxy/slam.py`. They apply to `source_0`
only; `source_1` stays parametric.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh_init = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape)
regularization_init = al.reg.Adapt

mesh = af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape)
regularization = al.reg.Adapt

"""
__SLaM Pipeline__

The six stages, in order. Stage 1 runs on the tight mask; every stage after it on the full one.
"""
source_lp_result_1 = source_lp_1(
    settings_search=settings_search,
    dataset=dataset_1,
    mask_radius=mask_radius_1,
    main_lens_centres=main_lens_centres,
    redshift_lens=redshift_lens,
    redshift_source_0=redshift_source_0,
)

source_lp_result_2 = source_lp_2(
    settings_search=settings_search,
    dataset=dataset_2,
    mask_radius=mask_radius_2,
    source_lp_result_1=source_lp_result_1,
    redshift_source_1=redshift_source_1,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset_2,
    source_lp_result_2=source_lp_result_2,
    mesh_init=mesh_init,
    regularization_init=regularization_init,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset_2,
    source_lp_result_2=source_lp_result_2,
    source_pix_result_1=source_pix_result_1,
    mesh=mesh,
    regularization=regularization,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset_2,
    mask_radius=mask_radius_2,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset_2,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
__Result__

`mass_result` holds the final model. The multi-galaxy checks in `multi_galaxy/slam.py` all apply — the mass
split, and the mass centres against the light centres.

The DSPL-specific check is `source_0`'s mass. It is constrained by where `source_1`'s images land, so a
posterior that has not moved from its prior means the second ring never informed the fit, and the mass split is
no better constrained than the baseline pipeline would have left it.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/slam.py` — the baseline pipeline these stages are copied from.
 - `multi_galaxy/features/advanced/double_source_plane_lens/chaining.py` — the same two-mask idea in two
   searches rather than six.
 - `multi_galaxy/features/advanced/double_source_plane_lens/modeling.py` — what a DSPL model contains.
 - `guides/modeling/slam_start_here` — what each stage is for, in full.
"""

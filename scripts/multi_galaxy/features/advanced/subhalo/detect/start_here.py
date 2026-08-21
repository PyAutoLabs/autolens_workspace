"""
Subhalo Detection: Multi Galaxy
===============================

This script runs the full SLaM pipeline followed by the three SUBHALO stages, to detect a dark matter subhalo
in a multi-galaxy strong lens.

Read `multi_galaxy/slam.py` first — the multi-galaxy SLaM baseline — and `guides/modeling/slam_start_here`
before it. The five SLaM stages here are copied from that baseline unchanged; what this script adds is the
subhalo search on top of them.

__Contents__

- **What Changes:** What subhalo detection adds to the baseline pipeline.
- **The False Positive This Regime Invites:** Read this before interpreting any detection.
- **Helpers:** The deflector-count helper the baseline uses.
- **Source LP / Source Pix 1 & 2 / Light LP / Mass Total:** The baseline's five stages.
- **Subhalo Pipeline (no subhalo):** The evidence baseline.
- **Subhalo Pipeline (grid search):** A subhalo confined to each cell of an image-plane grid.
- **Subhalo Pipeline (refine):** The best cell, refitted freely.
- **Dataset, Centres, Mask:** Set up.
- **Mesh Shape:** The pixelization classes.
- **Pipeline:** Run everything in order.
- **Bayesian Evidence:** How to decide whether anything was detected.
- **Grid Search Result:** Where in the image plane a subhalo helps.

__What Changes__

The five SLaM stages produce a smooth model — deflectors, shear and source, with no perturber. The three
SUBHALO stages then ask whether adding a compact dark perturber improves the fit:

1. **no subhalo** — refit the smooth model to establish a Bayesian evidence baseline.
2. **grid search** — divide the image plane into cells and fit a subhalo confined to each one, so the search
   cannot collapse onto a single local maximum.
3. **refine** — take the highest-evidence cell and refit with the subhalo's position free.

A detection is a comparison between stage 1's evidence and stage 3's, not a statement about stage 3 alone.

__The False Positive This Regime Invites__

Detection is a comparison against the smooth model, so the detection floor is set by how well that smooth model
is constrained. In the multi-galaxy regime the smooth model has a known weak spot: the split of mass between
the co-dominant deflectors is far less well constrained than their total (`multi_galaxy/modeling.py`).

**A wrong mass split produces residuals that look like a subhalo.** They are compact, they sit on the arcs, and
they are of the same character a perturber produces — this was checked directly rather than assumed, and a
mis-split of roughly one percent of the total Einstein radius is enough to produce residual power comparable to
a 10^10 solar mass subhalo. If anything the mis-split residual is the more concentrated of the two.

Two practical consequences, both of which this script acts on:

 - **The comparison model must be right.** Stage 1's smooth model includes **every** deflector, freely fitted.
   If it silently dropped one, the "no subhalo" baseline would be a mis-split model, and the grid search would
   find a subhalo compensating for the missing galaxy.
 - **A detection is not conclusive on its own.** Before believing one, check that the deflectors' masses in the
   subhalo model agree with the mass-total stage's. A detection that arrived alongside a shifted mass split is
   more likely the split than a subhalo.

__SLaM Pipelines__

The stage functions below are copied from `multi_galaxy/slam.py` rather than imported — that script is a script,
so importing it would execute its whole pipeline as a side effect.
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

    `shear_galaxy`, `source` and `subhalo` do not match the `lens_` prefix and are therefore not counted.
    """
    return sum(1 for key in vars(result.instance.galaxies) if key.startswith("lens_"))


def lens_dict_model_from(result):
    """
    Every deflector from a result, as models.

    This helper exists because the subhalo stages below must carry **all** deflectors. Dropping any of them
    would make the comparison model a mis-split model, which is the specific failure mode described at the top
    of this script.
    """
    return {
        f"lens_{i}": getattr(result.model.galaxies, f"lens_{i}")
        for i in range(n_main_from(result))
    }


"""
__SOURCE LP PIPELINE__

Identical to `multi_galaxy/slam.py`.
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

Identical to `multi_galaxy/slam.py`.
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

Identical to `multi_galaxy/slam.py`.
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

Identical to `multi_galaxy/slam.py`.
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

Identical to `multi_galaxy/slam.py`. Its result is the smooth model the subhalo search compares against, so it
is worth reading its posteriors before going further — the detection floor is set here.
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
__SUBHALO PIPELINE (no subhalo)__

Refits the smooth model to establish the Bayesian evidence baseline every later stage is compared against.

**Every deflector is carried, freely fitted.** This is the stage where dropping one would matter most: the
baseline would then be a mis-split model, and the grid search would find a "subhalo" compensating for the
missing galaxy. `lens_dict_model_from` exists to make that impossible to get wrong.
"""


def subhalo_no_subhalo(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
    )

    source = al.util.chaining.source_from(result=mass_result)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict_model_from(mass_result),
            shear_galaxy=mass_result.model.galaxies.shear_galaxy,
            source=source,
        ),
    )

    search = af.Nautilus(
        name="subhalo[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SUBHALO PIPELINE (grid search)__

Fits a subhalo confined to each cell of a grid over the image plane, so no single search has to find a compact
perturber anywhere in a two-dimensional space at once.

__Where To Put The Grid__

`grid_dimension_arcsec` should cover the arcs, which is where a subhalo is detectable at all. For a multi-galaxy
lens the arcs wrap around the pair as a whole rather than around one galaxy, so the grid must cover a region
centred on the system, not on either deflector.

Every deflector remains free in each cell's fit, for the reason given in the no-subhalo stage above.
"""


def subhalo_grid_search(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_no_subhalo_result: af.Result,
    subhalo_mass: af.Model,
    grid_dimension_arcsec: float = 3.0,
    number_of_steps: int = 2,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
    )

    subhalo = af.Model(al.Galaxy, mass=subhalo_mass)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )

    lens_redshift = subhalo_no_subhalo_result.instance.galaxies.lens_0.redshift
    source_redshift = subhalo_no_subhalo_result.instance.galaxies.source.redshift

    subhalo.redshift = lens_redshift
    subhalo.mass.redshift_object = lens_redshift
    subhalo.mass.redshift_source = source_redshift

    source = al.util.chaining.source_from(result=mass_result)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict_model_from(mass_result),
            shear_galaxy=mass_result.model.galaxies.shear_galaxy,
            subhalo=subhalo,
            source=source,
        ),
    )

    search = af.Nautilus(
        name="subhalo[2]_[search_lens_plane]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    grid_search = af.SearchGridSearch(
        search=search,
        number_of_steps=number_of_steps,
    )

    return grid_search.fit(
        model=model,
        analysis=analysis,
        grid_priors=[
            model.galaxies.subhalo.mass.centre_1,
            model.galaxies.subhalo.mass.centre_0,
        ],
        info=settings_search.info,
    )


"""
__SUBHALO PIPELINE (refine)__

Refits the highest-evidence cell with the subhalo's position free, initialised from that cell.
"""


def subhalo_refine(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_no_subhalo_result: af.Result,
    subhalo_grid_search_result,
    subhalo_mass: af.Model,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
    )

    lens_redshift = subhalo_no_subhalo_result.instance.galaxies.lens_0.redshift
    source_redshift = subhalo_no_subhalo_result.instance.galaxies.source.redshift

    subhalo = af.Model(al.Galaxy, redshift=lens_redshift, mass=subhalo_mass)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre = subhalo_grid_search_result.model_centred_absolute(
        a=1.0
    ).galaxies.subhalo.mass.centre
    subhalo.mass.redshift_object = lens_redshift
    subhalo.mass.redshift_source = source_redshift

    grid_model = subhalo_grid_search_result.model

    lens_dict = {
        f"lens_{i}": getattr(grid_model.galaxies, f"lens_{i}")
        for i in range(n_main_from(mass_result))
    }

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=grid_model.galaxies.shear_galaxy,
            subhalo=subhalo,
            source=grid_model.galaxies.source,
        ),
    )

    search = af.Nautilus(
        name="subhalo[3]_[single_plane_refine]",
        **settings_search.search_dict,
        n_live=400,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

The `subhalo` multi-galaxy dataset — the `simple` pair with a dark matter subhalo on the arcs.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "subhalo"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/subhalo/simulator.py",
        ],
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
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.
"""
mask_radius = 3.0

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
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy") / "features" / "advanced" / "subhalo" / "detect",
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
__Pipeline__

The five SLaM stages, then the three subhalo stages.
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

result_no_subhalo = subhalo_no_subhalo(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
)

result_subhalo_grid_search = subhalo_grid_search(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
    subhalo_no_subhalo_result=result_no_subhalo,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
)

result_with_subhalo = subhalo_refine(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
    subhalo_no_subhalo_result=result_no_subhalo,
    subhalo_grid_search_result=result_subhalo_grid_search,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
)

"""
__Bayesian Evidence__

A detection is decided by comparing the Bayesian evidence of the fits with and without a subhalo.

The following scale describes how log evidence increases correspond to detection significances:

 - Negative increase: no detection.
 - Between 0 and 3: no detection.
 - Between 3 and 5: weak evidence; treat as a non-detection.
 - Between 5 and 10: medium evidence, still inconclusive.
 - Between 10 and 20: strong evidence; consider it a detection.
 - Above 20: very strong evidence; definitive detection.
"""
evidence_no_subhalo = result_no_subhalo.samples.log_evidence
evidence_with_subhalo = result_with_subhalo.samples.log_evidence

log_evidence_increase = evidence_with_subhalo - evidence_no_subhalo

print(f"Evidence increase: {log_evidence_increase}")

"""
__Before Believing A Detection__

Apply the multi-galaxy check described at the top of this script.

Compare each deflector's mass in `result_with_subhalo` against the same deflector in `mass_result`. If the
subhalo arrived alongside a shifted mass split, the evidence increase may be measuring the split rather than a
perturber — the two produce residuals of the same character in the same place.

A detection whose deflector masses are unchanged is the one worth believing.
"""
for i in range(n_main_from(mass_result)):
    smooth = getattr(mass_result.instance.galaxies, f"lens_{i}")
    with_sub = getattr(result_with_subhalo.instance.galaxies, f"lens_{i}")

    print(
        f"lens_{i} einstein_radius — smooth model: {smooth.mass.einstein_radius:.4f}, "
        f"subhalo model: {with_sub.mass.einstein_radius:.4f}"
    )

"""
__Log Likelihood__

A simpler metric than the evidence, without its complexity penalty.
"""
log_likelihood_increase = (
    result_with_subhalo.samples.log_likelihood
    - result_no_subhalo.samples.log_likelihood
)

print(f"Log likelihood increase: {log_likelihood_increase}")

"""
__Grid Search Result__

The grid search results show where in the image plane a subhalo improves the fit, which is often more
informative than the single number above — a detection smeared across many cells is a different situation from
one confined to a single cell.
"""
subhalo_result = al.subhalo.SubhaloGridSearchResult(result=result_subhalo_grid_search)

log_evidence_array = subhalo_result.figure_of_merit_array(
    use_log_evidences=True,
    relative_to_value=result_no_subhalo.samples.log_evidence,
)

print("Log evidence array:")
print(log_evidence_array)

aplt.plot_array(array=log_evidence_array, title="Subhalo Grid Search Log Evidence")

print("Subhalo mass array:")
print(subhalo_result.subhalo_mass_array)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/slam.py` — the baseline pipeline whose five stages this script copies.
 - `multi_galaxy/modeling.py` — the mass-split degeneracy that sets the detection floor here.
 - `imaging/features/advanced/subhalo` — the galaxy-scale walkthrough, which also covers sensitivity mapping
   (not part of this package, as at group scale).
"""

"""
Subhalo Detection: Group
=========================

Strong gravitational lenses can be used to detect the presence of small-scale dark matter (DM) subhalos. This occurs
when the DM subhalo overlaps the lensed source emission, and therefore gravitationally perturbs the observed image of
the lensed source galaxy.

This script extends the standard DM subhalo detection pipeline to group-scale lenses. The key adaptation is that the
lens model includes ALL group galaxies (main lens galaxies + extra galaxies) with their mass profiles, alongside
the dark matter subhalo.

__Contents__

**SLaM Pipelines:** The Source, (lens) Light and Mass (SLaM) pipelines are advanced lens modeling pipelines.
**Grid Search:** The second stage uses a grid-search of non-linear searches.
**Group Adaptation:** The lens model includes all group galaxies.
**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Centres:** Load galaxy centres from JSON files.
**Over Sampling:** Set up the adaptive over-sampling grid.
**SLaM Pipeline Functions:** The pipeline functions adapted for group-scale lenses.
**Bayesian Evidence:** Determine if a DM subhalo was detected.
**Grid Search Result:** Inspect the grid search results.

__SLaM Pipelines__

The Source, (lens) Light and Mass (SLaM) pipelines are used for all DM subhalo detection analyses. The SLaM
pipelines are extended with a SUBHALO PIPELINE which performs three chained non-linear searches:

 1) Fits the lens model without a DM subhalo to establish a Bayesian evidence baseline.
 2) Performs a grid-search where each cell includes a DM subhalo confined to a small 2D region.
 3) Refines the best-fit subhalo model initialized from the highest evidence grid cell.

__Group Adaptation__

For group-scale lenses, the key differences from galaxy-scale subhalo detection are:

 - The lens model includes all main lens galaxies and extra galaxies with their mass profiles.
 - The subhalo is added as an additional galaxy in the model alongside the group galaxies.
 - The larger field of view and more complex mass distribution mean the subhalo must be searched
   across a wider area.

__Model__

Using SOURCE LP, SOURCE PIX, LIGHT LP, MASS TOTAL and SUBHALO PIPELINES this script fits ``Imaging`` of a
group-scale strong lens where in the final model:

 - The main lens galaxy's light is an MGE bulge.
 - The main lens galaxy's total mass distribution is a ``PowerLaw``.
 - Extra galaxies have MGE light and ``IsothermalSph`` mass with fixed centres.
 - A dark matter subhalo near the lens galaxy mass is included as an ``NFWMCRLudlowSph``.
 - The source galaxy is an ``Inversion``.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE__

Fits the group lens with parametric light profiles for the source, using MGE for all galaxies.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    extra_galaxies_centres,
    redshift_lens: float,
    redshift_source: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    # Main Lens Galaxies:

    lens_models = []

    for i, centre in enumerate(main_lens_centres):

        lens_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
        )

        lens = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=lens_bulge,
            mass=af.Model(al.mp.Isothermal),
            shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
        )

        lens_models.append(lens)

    # Extra Galaxies:

    extra_galaxies_list = []

    for centre in extra_galaxies_centres:

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
        )

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

        extra_galaxy = af.Model(
            al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=mass
        )

        extra_galaxies_list.append(extra_galaxy)

    extra_galaxies = af.Collection(extra_galaxies_list)

    # Source:

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    source = af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge)

    # Overall Model:

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Fits a pixelized source reconstruction, using the lens model from the SOURCE LP PIPELINE.
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
        use_jax=True,
    )

    # Use the lens model from the SOURCE LP result, fixing light and freeing mass.

    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens_0.mass,
        mass_result=source_lp_result.model.galaxies.lens_0.mass,
        unfix_mass_centre=True,
    )
    shear = source_lp_result.model.galaxies.lens_0.shear

    lens_0 = af.Model(
        al.Galaxy,
        redshift=source_lp_result.instance.galaxies.lens_0.redshift,
        bulge=source_lp_result.instance.galaxies.lens_0.bulge,
        mass=mass,
        shear=shear,
    )

    lens_dict = {"lens_0": lens_0}

    # Fix extra galaxies to their best-fit values.

    extra_galaxies = source_lp_result.instance.extra_galaxies

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=mesh_init,
            regularization=regularization_init,
        ),
    )

    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Refines the pixelized source reconstruction with adaptive mesh.
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

    lens_0 = af.Model(
        al.Galaxy,
        redshift=source_lp_result.instance.galaxies.lens_0.redshift,
        bulge=source_lp_result.instance.galaxies.lens_0.bulge,
        mass=source_pix_result_1.instance.galaxies.lens_0.mass,
        shear=source_pix_result_1.instance.galaxies.lens_0.shear,
    )

    lens_dict = {"lens_0": lens_0}

    extra_galaxies = source_pix_result_1.instance.extra_galaxies

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=mesh,
            regularization=regularization,
        ),
    )

    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
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

Refits the lens light using MGE, with the source pixelization fixed.
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    main_lens_centres,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    lens_0 = af.Model(
        al.Galaxy,
        redshift=source_result_for_lens.instance.galaxies.lens_0.redshift,
        bulge=lens_bulge,
        mass=source_result_for_lens.instance.galaxies.lens_0.mass,
        shear=source_result_for_lens.instance.galaxies.lens_0.shear,
    )

    lens_dict = {"lens_0": lens_0, "source": source}

    extra_galaxies = source_result_for_lens.instance.extra_galaxies

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Fits the total mass distribution as a ``PowerLaw``.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    mass = af.Model(al.mp.PowerLaw)

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
        use_jax=True,
    )

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_result_for_lens.model.galaxies.lens_0.mass,
        unfix_mass_centre=True,
    )

    bulge = light_result.instance.galaxies.lens_0.bulge

    source = al.util.chaining.source_from(result=source_result_for_source)

    lens_0 = af.Model(
        al.Galaxy,
        redshift=source_result_for_lens.instance.galaxies.lens_0.redshift,
        bulge=bulge,
        mass=mass,
        shear=source_result_for_lens.model.galaxies.lens_0.shear,
    )

    lens_dict = {"lens_0": lens_0, "source": source}

    extra_galaxies = light_result.instance.extra_galaxies

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SUBHALO PIPELINE (no subhalo)__

Refits the lens model without a DM subhalo to establish a Bayesian evidence baseline.
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
    lens_0 = mass_result.model.galaxies.lens_0

    lens_dict = {"lens_0": lens_0, "source": source}

    extra_galaxies = mass_result.instance.extra_galaxies

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
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

Performs a grid search where each cell includes a DM subhalo confined to a small 2D region of the image plane.

For group-scale lenses, the grid search area may need to be larger than galaxy-scale lenses because the
lensed source images can span a wider area due to the more complex mass distribution.
"""


def subhalo_grid_search(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_no_subhalo_result: af.Result,
    subhalo_mass: af.Model,
    grid_dimension_arcsec: float = 5.0,
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

    subhalo.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )

    subhalo.redshift = (
        subhalo_no_subhalo_result.instance.galaxies.lens_0.redshift
    )
    subhalo.mass.redshift_object = (
        subhalo_no_subhalo_result.instance.galaxies.lens_0.redshift
    )
    subhalo.mass.redshift_source = (
        subhalo_no_subhalo_result.instance.galaxies.source.redshift
    )

    lens_0 = mass_result.model.galaxies.lens_0
    source = al.util.chaining.source_from(result=mass_result)

    lens_dict = {"lens_0": lens_0, "subhalo": subhalo, "source": source}

    extra_galaxies = mass_result.instance.extra_galaxies

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="subhalo[2]_[search_lens_plane]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    subhalo_grid_search = af.SearchGridSearch(
        search=search,
        number_of_steps=number_of_steps,
    )

    return subhalo_grid_search.fit(
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

Refines the best-fit subhalo model, initializing the subhalo centre from the highest evidence grid cell.
"""


def subhalo_refine(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_no_subhalo_result: af.Result,
    subhalo_grid_search_result: af.Result,
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

    subhalo = af.Model(
        al.Galaxy,
        redshift=subhalo_no_subhalo_result.instance.galaxies.lens_0.redshift,
        mass=subhalo_mass,
    )

    subhalo.redshift = (
        subhalo_no_subhalo_result.instance.galaxies.lens_0.redshift
    )
    subhalo.mass.redshift_object = (
        subhalo_no_subhalo_result.instance.galaxies.lens_0.redshift
    )
    subhalo.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo.mass.centre = subhalo_grid_search_result.model_centred_absolute(
        a=1.0
    ).galaxies.subhalo.mass.centre

    subhalo.redshift = (
        subhalo_grid_search_result.model.galaxies.subhalo.redshift
    )
    subhalo.mass.redshift_object = subhalo.redshift

    lens_dict = {
        "lens_0": subhalo_grid_search_result.model.galaxies.lens_0,
        "subhalo": subhalo,
        "source": subhalo_grid_search_result.model.galaxies.source,
    }

    extra_galaxies = mass_result.instance.extra_galaxies

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="subhalo[3]_[single_plane_refine]",
        **settings_search.search_dict,
        n_live=600,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset + Masking__

Load, plot and mask the ``Imaging`` data.
"""
dataset_name = "dark_matter_subhalo"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/subhalo/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Centres__

Load the centres of the main lens galaxies and extra galaxies from JSON files.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__

Over sampling at all galaxy centres.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("subhalo_detect_group"),
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Mesh Shape__

The mesh shape for the pixelized source reconstruction.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__

The code below calls the full SLaM PIPELINE adapted for group-scale lenses.
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
    mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Adapt,
)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

over_sampling = al.util.over_sample.over_sample_size_via_adapt_from(
    data=adapt_images.galaxy_name_image_dict["('galaxies', 'source')"],
    noise_map=dataset.noise_map,
)

dataset = dataset.apply_over_sampling(over_sample_size_pixelization=over_sampling)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
    regularization=al.reg.Adapt,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    main_lens_centres=main_lens_centres,
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
    grid_dimension_arcsec=5.0,
    number_of_steps=2,
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

To determine if a DM subhalo was detected by the pipeline, we compare the log of the Bayesian evidences of
the model-fits performed with and without a subhalo.

The following scale describes how different log evidence increases correspond to detection significances:

 - Negative log evidence increase: No detection.
 - Log evidence increase between 0 and 3: No detection.
 - Log evidence increase between 3 and 5: Weak evidence, should consider it a non-detection.
 - Log evidence increase between 5 and 10: Medium evidence, but still inconclusive.
 - Log evidence increase between 10 and 20: Strong evidence, consider it a detection.
 - Log evidence increase > 20: Very strong evidence, definitive detection.
"""
evidence_no_subhalo = result_no_subhalo.samples.log_evidence
evidence_with_subhalo = result_with_subhalo.samples.log_evidence

log_evidence_increase = evidence_with_subhalo - evidence_no_subhalo

print("Evidence Increase: ", log_evidence_increase)

"""
__Log Likelihood__

The log likelihood increase provides a simpler metric for how well the subhalo model fits the data.
"""
log_likelihood_no_subhalo = result_no_subhalo.samples.log_likelihood
log_likelihood_with_subhalo = result_with_subhalo.samples.log_likelihood

log_likelihood_increase = log_likelihood_with_subhalo - log_likelihood_no_subhalo

print("Log Likelihood Increase: ", log_likelihood_increase)

"""
__Grid Search Result__

The grid search results can be used to inspect where in the image plane a subhalo provides the best fit.
"""
subhalo_grid_search_result = al.subhalo.SubhaloGridSearchResult(
    result=result_subhalo_grid_search
)

log_evidence_array = subhalo_grid_search_result.figure_of_merit_array(
    use_log_evidences=True,
    relative_to_value=result_no_subhalo.samples.log_evidence,
)

print("Log Evidence Array: \n")
print(log_evidence_array)

aplt.plot_array(array=log_evidence_array, title="")

mass_array = subhalo_grid_search_result.subhalo_mass_array

print("Mass Array: \n")
print(mass_array)

subhalo_centres_grid = subhalo_grid_search_result.subhalo_centres_grid

print("Subhalo Centres Grid: \n")
print(subhalo_centres_grid)

"""
Finish.
"""

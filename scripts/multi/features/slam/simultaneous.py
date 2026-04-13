"""
SLaM: Multi Wavelength Simultaneous
===================================

This example shows how to use the SLaM pipeline to fit a lens dataset at multiple wavelengths simultaneously.

Simultaneous multi-dataset fits are currently built into the SLaM pipeline without user input or customization.
Therefore, as long as lists of `Analysis` objects are created, summed and passed to the SLaM pipelines, the analysis
will fit every dataset simultaneously and it will adapt the model as follows:

- Sub-pixel offsets between the datasets are fully modeled as free parameters in each stage of the pipeline, assuming
  broad uniform priors for every step. This is because the precision of a lens model can often be less than the
  requirements on astrometry.

- The regularization parameters are free for every dataset in the `source_pix[1]` and `source_pix[2]` stages. This is because
  the source morphology can be different between datasets, and the regularization scheme adapts to this.

- From the `light_lp` stage onwards, the regularization scheme for each dataset is different fixed to that inferred
  for the `source_pix[2]` stage.

Simultaneous fitting SLaM pipelines are not designed for customization, for example changing the model from the
set up above. This is because we are still figuring out the best way to perform multi-wavelength modeling, but have
so far figured the above settings are important.

If you need customization of the model or pipeline, you should pick apart the SLaM pipeline and customize
them as you see fit.

__Contents__

**Preqrequisites:** Before reading this script, you should have familiarity with the following key concepts.
**This Script:** Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this.
**Dataset:** Load and plot the strong lens dataset.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**SLaM Pipeline Functions:** Overview of slam pipeline functions for this example.
**SLaM Pipeline:** Overview of slam pipeline for this example.

__Preqrequisites__

Before reading this script, you should have familiarity with the following key concepts:

- **Multi**: The `autolens_workspace/*/advanced/multi` package describes many different ways that multiple datasets
  can be modeled in a single analysis.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/slam_start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.
"""
dataset_waveband_list = ["g", "r"]
pixel_scale_list = [0.12, 0.08]

dataset_name = "lens_sersic"
dataset_main_path = Path("dataset", "multi", "imaging", dataset_name)
dataset_path = Path(dataset_main_path, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi/simulator.py"],
        check=True,
    )


dataset_list = []

for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):
    dataset = al.Imaging.from_fits(
        data_path=Path(dataset_main_path, f"{dataset_waveband}_data.fits"),
        noise_map_path=Path(dataset_main_path, f"{dataset_waveband}_noise_map.fits"),
        psf_path=Path(dataset_main_path, f"{dataset_waveband}_psf.fits"),
        pixel_scales=pixel_scale,
    )

    mask_radius = 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    dataset_list.append(dataset)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("slam", "multi", "simultaneous"),
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
__SLaM Pipeline Functions__
"""


def source_lp(
    settings_search,
    analysis_list,
    lens_bulge,
    source_bulge,
    redshift_lens,
    redshift_source,
    dataset_model,
    mass_centre=(0.0, 0.0),
    n_batch=50,
):
    """
    SOURCE LP PIPELINE: fits an initial lens model using a parametric source, shared across all datasets
    via a factor graph so that the model parameters are the same for every waveband.
    """
    mass = af.Model(al.mp.Isothermal)
    mass.centre = mass_centre

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
            ),
        ),
        dataset_model=dataset_model,
    )

    analysis_factor_list = []

    for analysis in analysis_list:
        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


def source_pix_1(
    settings_search,
    analysis_list,
    source_lp_result,
    mesh_shape,
    dataset_model,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 1: initializes a pixelized source model, with per-dataset models that share
    the mass centre across wavebands to ensure consistent lens mass geometry.
    """
    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):
        mass = al.util.chaining.mass_from(
            mass=source_lp_result[i].model.galaxies.lens.mass,
            mass_result=source_lp_result[i].model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        if i > 0:
            mass.centre = model.galaxies.lens.mass.centre

        shear = source_lp_result[i].model.galaxies.lens.shear

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.lens.redshift,
                    bulge=source_lp_result[i].instance.galaxies.lens.bulge,
                    disk=source_lp_result[i].instance.galaxies.lens.disk,
                    mass=mass,
                    shear=shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=af.Model(
                            al.mesh.RectangularAdaptDensity, shape=mesh_shape
                        ),
                        regularization=al.reg.Adapt,
                    ),
                ),
            ),
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


def source_pix_2(
    settings_search,
    analysis_list,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    dataset_model,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 2: fits a pixelized source using adapt images from SOURCE PIX PIPELINE 1,
    with fixed lens mass and improved mesh and regularization.
    """
    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.lens.redshift,
                    bulge=source_lp_result[i].instance.galaxies.lens.bulge,
                    disk=source_lp_result[i].instance.galaxies.lens.disk,
                    mass=source_pix_result_1[i].instance.galaxies.lens.mass,
                    shear=source_pix_result_1[i].instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result[i].instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                        regularization=al.reg.Adapt,
                    ),
                ),
            ),
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


def light_lp(
    settings_search,
    analysis_list,
    source_result_for_lens,
    source_result_for_source,
    lens_bulge,
    dataset_model,
    n_batch=20,
):
    """
    LIGHT LP PIPELINE: fits the lens galaxy light with mass and source fixed from SOURCE PIX PIPELINE,
    using per-dataset models sharing the same lens bulge model across wavebands.
    """
    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):
        source = al.util.chaining.source_custom_model_from(
            result=source_result_for_source[i], source_is_model=False
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens[i].instance.galaxies.lens.redshift,
                    bulge=lens_bulge,
                    disk=None,
                    mass=source_result_for_lens[i].instance.galaxies.lens.mass,
                    shear=source_result_for_lens[i].instance.galaxies.lens.shear,
                ),
                source=source,
            ),
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=250,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


def mass_total(
    settings_search,
    analysis_list,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    dataset_model,
    n_batch=20,
):
    """
    MASS TOTAL PIPELINE: fits a PowerLaw total mass model using priors from SOURCE PIX PIPELINE,
    with lens light fixed from LIGHT LP PIPELINE.
    """
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):
        mass_i = al.util.chaining.mass_from(
            mass=mass,
            mass_result=source_result_for_lens[i].model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        source = al.util.chaining.source_from(
            result=source_result_for_source[i],
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens[i].instance.galaxies.lens.redshift,
                    bulge=light_result[i].instance.galaxies.lens.bulge,
                    disk=light_result[i].instance.galaxies.lens.disk,
                    mass=mass_i,
                    shear=source_result_for_lens[i].model.galaxies.lens.shear,
                ),
                source=source,
            ),
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


def subhalo_no_subhalo(
    settings_search,
    analysis_list,
    mass_result,
    dataset_model,
    n_batch=20,
):
    """
    SUBHALO PIPELINE 1: fits the lens model without a subhalo to provide Bayesian evidence for
    model comparison with the subhalo detection searches.
    """
    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):
        source = al.util.chaining.source_from(result=mass_result[i])
        lens = mass_result[i].model.galaxies.lens

        model = af.Collection(
            galaxies=af.Collection(lens=lens, source=source),
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="subhalo[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


def subhalo_grid_search(
    settings_search,
    analysis_list,
    mass_result,
    subhalo_result_1,
    subhalo_mass,
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
    n_batch=20,
):
    """
    SUBHALO PIPELINE 2: performs a grid search over subhalo positions to detect a dark matter subhalo,
    using per-dataset models with shared subhalo parameters.
    """
    subhalo = af.Model(al.Galaxy, mass=subhalo_mass)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )

    subhalo.redshift = subhalo_result_1[0].instance.galaxies.lens.redshift
    subhalo.mass.redshift_object = subhalo_result_1[0].instance.galaxies.lens.redshift
    subhalo.mass.redshift_source = subhalo_result_1[0].instance.galaxies.source.redshift

    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):
        lens = mass_result[i].model.galaxies.lens
        source = al.util.chaining.source_from(result=mass_result[i])

        model = af.Collection(
            galaxies=af.Collection(lens=lens, subhalo=subhalo, source=source),
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

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


def subhalo_refine(
    settings_search,
    analysis_list,
    subhalo_result_1,
    subhalo_grid_search_result_2,
    subhalo_mass,
    dataset_model,
    n_batch=20,
):
    """
    SUBHALO PIPELINE 3: refines the subhalo model parameters using priors initialized from the
    highest-evidence cell of the grid search.
    """
    subhalo = af.Model(
        al.Galaxy,
        redshift=subhalo_result_1[0].instance.galaxies.lens.redshift,
        mass=subhalo_mass,
    )

    subhalo.redshift = subhalo_result_1[0].instance.galaxies.lens.redshift
    subhalo.mass.redshift_object = subhalo_result_1[0].instance.galaxies.lens.redshift
    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre = subhalo_grid_search_result_2.model_centred_absolute(
        a=1.0
    ).galaxies.subhalo.mass.centre
    subhalo.redshift = subhalo_grid_search_result_2.model.galaxies.subhalo.redshift
    subhalo.mass.redshift_object = subhalo.redshift

    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):
        model = af.Collection(
            galaxies=af.Collection(
                lens=subhalo_grid_search_result_2.model.galaxies.lens,
                subhalo=subhalo,
                source=subhalo_grid_search_result_2.model.galaxies.source,
            ),
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)
        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.Nautilus(
        name="subhalo[3]_[single_plane_refine]",
        **settings_search.search_dict,
        n_live=600,
        n_batch=n_batch,
    )

    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


"""
__SLaM Pipeline__
"""

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

dataset_model = af.Model(al.DatasetModel)

# Lens Light

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

# Source Light

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

analysis_list = [
    al.AnalysisImaging(dataset=dataset, use_jax=True) for dataset in dataset_list
]

source_lp_result = source_lp(
    settings_search=settings_search,
    analysis_list=analysis_list,
    lens_bulge=lens_bulge,
    source_bulge=source_bulge,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
    dataset_model=dataset_model,
)

positions_likelihood = source_lp_result.positions_likelihood_from(
    factor=3.0, minimum_threshold=0.2
)

adapt_images_list = []

for result in source_lp_result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result)
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)
    adapt_images_list.append(adapt_images)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
        use_jax=True,
    )
    for result, adapt_images in zip(source_lp_result, adapt_images_list)
]

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_lp_result=source_lp_result,
    mesh_shape=mesh_shape,
    dataset_model=dataset_model,
)

adapt_images_list = []

for result in source_pix_result_1:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result)
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)
    adapt_images_list.append(adapt_images)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )
    for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
]

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
    dataset_model=dataset_model,
)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        raise_inversion_positions_likelihood_exception=False,
    )
    for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
]

light_result = light_lp(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    dataset_model=dataset_model,
)

positions_likelihood = source_pix_result_1[0].positions_likelihood_from(
    factor=3.0, minimum_threshold=0.2
)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[positions_likelihood],
    )
    for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
]

mass_result = mass_total(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    dataset_model=dataset_model,
)

subhalo_result_1 = subhalo_no_subhalo(
    settings_search=settings_search,
    analysis_list=analysis_list,
    mass_result=mass_result,
    dataset_model=dataset_model,
)

subhalo_grid_search_result_2 = subhalo_grid_search(
    settings_search=settings_search,
    analysis_list=analysis_list,
    mass_result=mass_result,
    subhalo_result_1=subhalo_result_1,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

subhalo_result_3 = subhalo_refine(
    settings_search=settings_search,
    analysis_list=analysis_list,
    subhalo_result_1=subhalo_result_1,
    subhalo_grid_search_result_2=subhalo_grid_search_result_2,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    dataset_model=dataset_model,
)

"""
Finish.
"""

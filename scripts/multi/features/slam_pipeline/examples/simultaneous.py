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

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
import sys
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
from scripts.multi.features import slam_pipeline

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.
"""
dataset_waveband_list = ["g", "r"]
pixel_scale_list = [0.12, 0.08]

dataset_name = "lens_sersic"
dataset_main_path = Path("dataset", "multi", "imaging", dataset_name)
dataset_path = Path(dataset_main_path, dataset_name)


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
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE fits an identical to the `start_here.ipynb` example, except:

 - The model includes the (y,x) offset of each dataset relative to the first dataset, which is added to every
  `AnalysisImaging` object such that there are 2 extra parameters fitted for each dataset.
"""

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

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis_list=analysis_list,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__JAX & Preloads__

The `autolens_workspace/*/imaging/features/pixelization/modeling` example describes how JAX required preloads in
advance so it knows the shape of arrays it must compile functions for.
"""
mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 40

preloads = al.Preloads(
    mapper_indices=al.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE is identical to the `slam_start_here.ipynb` example, except:

- The model includes the (y,x) dataset offsets again.

- Inside the SLaM pipeline, a unique regularization scheme is set up for every source pixelization, as different
  wavelengh datasets required different regularization schemes.
"""
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
        preloads=preloads,
        use_jax=True,
    )
    for result, adapt_images in zip(source_lp_result, adapt_images_list)
]

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_lp_result=source_lp_result,
    mesh_init=af.Model(al.mesh.RectangularMagnification, shape=mesh_shape),
    regularization_init=al.reg.AdaptiveBrightness,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

The SOURCE PIX PIPELINE is identical to the `slam_start_here.ipynb` example, except with the same changes
as the SOURCE PIX PIPELINE 1.
"""
adapt_images_list = []

for result in source_pix_result_1:

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result)
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    adapt_images_list.append(adapt_images)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        preloads=preloads,
        use_jax=True,
    )
    for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
]

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularSource, shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE is setup identically to the `slam_start_here.ipynb` example.
"""
analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        preloads=preloads,
        raise_inversion_positions_likelihood_exception=False,
    )
    for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
]

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

light_result = slam_pipeline.light_lp.run(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=None,
    dataset_model=af.Model(al.DatasetModel),
)

"""
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
positions_likelihood = source_pix_result_1[0].positions_likelihood_from(
    factor=3.0, minimum_threshold=0.2
)

analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        preloads=preloads,
        positions_likelihood_list=[positions_likelihood],
    )
    for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
]

mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis_list=analysis_list,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
    dataset_model=af.Model(al.DatasetModel),
)

"""
__SUBHALO PIPELINE__

The SUBHALO pipeline is described in `imaging/features/advanced/subhalo`.
"""
analysis_list = [
    al.AnalysisImaging(
        dataset=result.max_log_likelihood_fit.dataset,
        adapt_images=adapt_images,
        preloads=preloads,
        positions_likelihood_list=[positions_likelihood],
    )
    for result, adapt_images in zip(source_pix_result_1, adapt_images_list)
]

subhalo_result_1 = slam_pipeline.subhalo.detection.run_1_no_subhalo(
    settings_search=settings_search,
    analysis_list=analysis_list,
    mass_result=mass_result,
)

subhalo_grid_search_result_2 = slam_pipeline.subhalo.detection.run_2_grid_search(
    settings_search=settings_search,
    analysis_list=analysis_list,
    mass_result=mass_result,
    subhalo_result_1=subhalo_result_1,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

subhalo_result_3 = slam_pipeline.subhalo.detection.run_3_subhalo(
    settings_search=settings_search,
    analysis_list=analysis_list,
    subhalo_result_1=subhalo_result_1,
    subhalo_grid_search_result_2=subhalo_grid_search_result_2,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
)

"""
Finish.
"""

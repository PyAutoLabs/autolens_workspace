"""
SLaM: Multi Wavelength Independent
==================================

This example shows how to use the SLaM pipeline to fit a lens dataset at one wavelength, and then fit other data of
the same lens at different wavelengths with the same mass model fitted to the original dataset.

The first dataset fitted is regarded as the "main" dataset, meaning it should have the highest resolution and
signal-to-noise. This will ensure the mass model is the most accurate.

The remaining datasets are then fitted, which may be similar quality to the main dataset or lower resolution and
signal-to-noise. These datasets are fitted with the following approach:

- The mass model (e.g. SIE +Shear) is fixed to the result of the VIS fit.

- The lens light (Multi Gaussian Expansion) has the `intensity` values of the Gaussians updated using linear algebra.
  to capture changes in the lens light over wavelength, but it does not update the Gaussian parameters (e.g. `centre`,
 `elliptical_comps`, `sigma`) themselves due to the lower resolution of the data.

- The source reconstruction (RectangularMagnification adaptive mesh) is updated using linear algebra to reconstruct
  the source, but again fixes  the source pixelization parameters themselves.

- Sub-pixel offsets between the datasets are fully modeled as free parameters, because the precision of a lens model
can often be less than the requirements on astrometry.

The restrictive nature of the lens mass, light and source models mean that much lower quality multi-wavelength data
can be fitted provided the first dataset is of high quality. This is key for upcoming surveys such as Euclid, where
the VIS instrument will be high resolution but many other wavebands will be lower resolution.

The subsequent fits to the lower resolution data use a reduced and simplified SLaM pipeline with the mass model
fixed to the result of the VIS fit.

__Preqrequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

- **Multi**: The `autolens_workspace/*/advanced/multi` package describes many different ways that multiple datasets
  can be modeled in a single analysis, including the example script `one_by_one.ipynb` which fits a primary dataset
  and then follows it up with fits to lower resolution datasets.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.
 - Two extra galaxies are included in the model, each with their light represented as a bulge with MGE light profile
   and their mass as a `IsothermalSph` profile.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/slam_start_here.ipynb` notebook.
"""

"""
Everything below is identical to `start_here.py` and thus not commented, as it is the same code.
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
import slam_pipeline

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.

We load a dataset with the waveband "g", which is the highest resolution data in this multi-wavelength example. 
"""
dataset_name = "lens_sersic"
dataset_main_path = Path("dataset", "multi", "imaging", dataset_name)

dataset_waveband = "g"

dataset = al.Imaging.from_fits(
    data_path=Path(dataset_main_path, f"{dataset_waveband}_data.fits"),
    noise_map_path=Path(dataset_main_path, f"{dataset_waveband}_noise_map.fits"),
    psf_path=Path(dataset_main_path, f"{dataset_waveband}_psf.fits"),
    pixel_scales=0.08,
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

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("slam", "multi", "independent"),
    unique_tag=f"{dataset_name}_data_{dataset_waveband}",
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

The SOURCE LP PIPELINE is identical to the `slam_start_here` example.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

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

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__JAX & Preloads__

The `autolens_workspace/*/imaging/features/pixelization/modeling` example describes how JAX required preloads in
advance so it knows the shape of arrays it must compile functions for.
"""
image_mesh = None
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

The SOURCE PIX PIPELINE (and every pipeline that follows) are identical to the `slam_start_here` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    use_jax=True,
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    image_mesh_init=None,
    mesh_init=af.Model(al.mesh.RectangularMagnification, shape=mesh_shape),
    regularization_init=al.reg.AdaptiveBrightness,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

As above, this pipeline also has the same API as the `slam_start_here` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    use_jax=True,
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=None,
    mesh=af.Model(al.mesh.RectangularSource, shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
)

"""
__LIGHT LP PIPELINE__ 

As above, this pipeline also has the same API as the `slam_start_here` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    use_jax=True,
)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

light_result = slam_pipeline.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE__

As above, this pipeline also has the same API as the `slam_start_here` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
)

"""
__Second Dataset Fits__

We now fit the secondary multi-wavelength datasets, which are lower resolution than the main dataset. 

This uses a for loop to iterate over every waveband of every dataset, load and mask the data and fit it.

Each fit uses a fixed mass model, the lens and source light models update via linear algebra and offsets are
includded (see full description above).

Its the usual API to set up dataset paths, but include its "main` path which is before the waveband folders.
"""
dataset_name = "lens_sersic"
dataset_main_path = Path("dataset", "multi", "imaging", dataset_name)

"""
__Dataset Wavebands__

The following list gives the names of the wavebands we are going to fit. 

The data for each waveband is loaded from a folder in the dataset folder with that name.
"""
dataset_waveband_list = ["r"]
pixel_scale_list = [0.12]

"""
__Dataset Model__

For each fit, the (y,x) offset of the secondary data from the primary data is a free parameter. 

This is achieved by setting up a `DatasetModel` for each waveband, which extends the model with components
including the grid offset.

This ensures that if the datasets are offset with respect to one another, the model can correct for this,
with sub-pixel offsets often being important in lens modeling as the precision of a lens model can often be
less than the requirements on astrometry.
"""
dataset_model = af.Model(al.DatasetModel)

dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
    lower_limit=-0.2, upper_limit=0.2
)
dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
    lower_limit=-0.2, upper_limit=0.2
)

"""
__Result Dict__

Visualization at the end of the pipeline will output all fits to all wavebands on a single matplotlib subplot.

The results of each fit are stored in a dictionary, which is used to pass the results of each fit to the
visualization functions.
"""
multi_result_dict = {"g": mass_result}

for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):
    dataset_path = dataset_main_path

    dataset = al.Imaging.from_fits(
        data_path=Path(dataset_main_path, f"{dataset_waveband}_data.fits"),
        noise_map_path=Path(dataset_main_path, f"{dataset_waveband}_noise_map.fits"),
        psf_path=Path(dataset_main_path, f"{dataset_waveband}_psf.fits"),
        pixel_scales=pixel_scale,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.1, 0.3],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Settings AutoFit__
    """
    settings_search = af.SettingsSearch(
        path_prefix=Path("slam", "multi", "independent"),
        unique_tag=f"{dataset_name}_data_{dataset_waveband}",
        info=None,
    )

    """
    __SOURCE LP PIPELINE (with lens light)__

    The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
    source galaxy's light, which in this example:

     - Uses an MGE with 2 x 30 Gaussians for the lens galaxy's light.

     - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.

     __Settings__:

     - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
    )

    centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    total_gaussians = 20
    gaussian_per_basis = 1

    log10_sigma_list = np.linspace(-3, np.log10(1.0), total_gaussians)

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    source_bulge = af.Model(
        al.lp_basis.Basis,
        profile_list=bulge_gaussian_list,
    )

    source_lp_result = slam_pipeline.source_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=light_result.instance.galaxies.lens.bulge,
        lens_disk=None,
        lens_point=light_result.instance.galaxies.lens.point,
        mass=mass_result.instance.galaxies.lens.mass,
        shear=mass_result.instance.galaxies.lens.shear,
        source_bulge=source_bulge,
        redshift_lens=0.5,
        redshift_source=1.0,
        dataset_model=dataset_model,
    )

    """
    __SOURCE PIX PIPELINE (with lens light)__

    The SOURCE PIX PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
    that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
    regularization, to set up the model and hyper images, and then:

     - Uses a `VoronoiBrightnessImage` pixelization.
     - Uses an `AdaptiveBrightness` regularization.
     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
     SOURCE PIX PIPELINE.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
        raise_inversion_positions_likelihood_exception=False,
    )

    dataset_model.grid_offset.grid_offset_0 = (
        source_lp_result.instance.dataset_model.grid_offset[0]
    )
    dataset_model.grid_offset.grid_offset_1 = (
        source_lp_result.instance.dataset_model.grid_offset[1]
    )

    source_pix_result_1 = slam_pipeline.source_pix.run_1(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        image_mesh_init=None,
        mesh_init=af.Model(al.mesh.RectangularMagnification, shape=mesh_shape),
        regularization_init=al.reg.AdaptiveBrightness,
        dataset_model=dataset_model,
        fixed_mass_model=True,
    )

    source_pix_result_1.max_log_likelihood_fit.inversion.cls_list_from(
        cls=al.AbstractMapper
    )[0].extent_from()

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    )

    dataset_model.grid_offset.grid_offset_0 = (
        source_lp_result.instance.dataset_model.grid_offset[0]
    )
    dataset_model.grid_offset.grid_offset_1 = (
        source_lp_result.instance.dataset_model.grid_offset[1]
    )

    multi_result = slam_pipeline.source_pix.run_2(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        image_mesh=None,
        mesh=af.Model(al.mesh.RectangularSource, shape=mesh_shape),
        regularization=al.reg.AdaptiveBrightness,
        dataset_model=dataset_model,
    )

    multi_result_dict[dataset_waveband] = multi_result

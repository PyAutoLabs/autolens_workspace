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

- The source reconstruction (RectangularAdaptDensity adaptive mesh) is updated using linear algebra to reconstruct
  the source, but again fixes  the source pixelization parameters themselves.

- Sub-pixel offsets between the datasets are fully modeled as free parameters, because the precision of a lens model
can often be less than the requirements on astrometry.

The restrictive nature of the lens mass, light and source models mean that much lower quality multi-wavelength data
can be fitted provided the first dataset is of high quality. This is key for upcoming surveys such as Euclid, where
the VIS instrument will be high resolution but many other wavebands will be lower resolution.

The subsequent fits to the lower resolution data use a reduced and simplified SLaM pipeline with the mass model
fixed to the result of the VIS fit.

__Contents__

**Preqrequisites:** Before using this SLaM pipeline, you should be familiar with.
**This Script:** Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this.
**Dataset:** Load and plot the strong lens dataset.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**SLaM Pipeline Functions:** Overview of slam pipeline functions for this example.
**SLaM Pipeline:** Overview of slam pipeline for this example.
**Second Dataset Fits:** We now fit the secondary multi-wavelength datasets, which are lower resolution than the main.
**Dataset Wavebands:** The following list gives the names of the wavebands we are going to fit.
**Result Dict:** The results of each fit are stored in a dictionary, which is used to pass the results of each fit.

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
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__ 

Load, plot and mask the `Imaging` data.

We load a dataset with the waveband "g", which is the highest resolution data in this multi-wavelength example. 
"""
dataset_name = "lens_sersic"
dataset_main_path = Path("dataset", "multi", "imaging", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_main_path.exists():
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/multi/simulator.py"],
        check=True,
    )

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

aplt.subplot_imaging_dataset(dataset=dataset)

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
__SLaM Pipeline Functions__
"""


def source_lp(
    settings_search,
    analysis,
    lens_bulge,
    source_bulge,
    redshift_lens,
    redshift_source,
    mass_centre=(0.0, 0.0),
    n_batch=50,
):
    """
    SOURCE LP PIPELINE: fits an initial lens model using a parametric source to establish a robust
    lens light, mass and source model for the main high-resolution dataset.
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
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def source_pix_1(
    settings_search,
    analysis,
    source_lp_result,
    mesh_shape,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 1: initializes a pixelized source model with mass priors from SOURCE LP PIPELINE.
    """
    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_lp_result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def source_pix_2(
    settings_search,
    analysis,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 2: fits an improved pixelized source using adapt images from SOURCE PIX PIPELINE 1,
    with fixed lens mass.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
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


def light_lp(
    settings_search,
    analysis,
    source_result_for_lens,
    source_result_for_source,
    lens_bulge,
    n_batch=20,
):
    """
    LIGHT LP PIPELINE: fits the lens galaxy light with mass and source fixed from SOURCE PIX PIPELINE.
    """
    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=None,
                mass=source_result_for_lens.instance.galaxies.lens.mass,
                shear=source_result_for_lens.instance.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def mass_total(
    settings_search,
    analysis,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    n_batch=20,
):
    """
    MASS TOTAL PIPELINE: fits a PowerLaw total mass model with priors from SOURCE PIX PIPELINE and
    lens light fixed from LIGHT LP PIPELINE.
    """
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=light_result.instance.galaxies.lens.bulge,
                disk=light_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_result_for_lens.model.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def source_lp_secondary(
    settings_search,
    analysis,
    light_result,
    mass_result,
    source_bulge,
    dataset_model,
    redshift_lens=0.5,
    redshift_source=1.0,
    n_batch=50,
):
    """
    SOURCE LP PIPELINE (secondary dataset): fits the source for a secondary waveband dataset with the lens
    light and mass fixed to the results of the main dataset pipeline.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=light_result.instance.galaxies.lens.bulge,
                disk=None,
                point=light_result.instance.galaxies.lens.point,
                mass=mass_result.instance.galaxies.lens.mass,
                shear=mass_result.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
            ),
        ),
        dataset_model=dataset_model,
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def source_pix_1_secondary(
    settings_search,
    analysis,
    source_lp_result,
    mesh_shape,
    dataset_model,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 1 (secondary dataset): initializes a pixelized source with fixed mass from
    the main dataset result, updating source reconstruction and dataset offsets.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_lp_result.instance.galaxies.lens.mass,
                shear=source_lp_result.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        dataset_model=dataset_model,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def source_pix_2_secondary(
    settings_search,
    analysis,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    dataset_model,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 2 (secondary dataset): fits an improved pixelized source using adapt images from
    SOURCE PIX PIPELINE 1, with all lens parameters fixed.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        dataset_model=dataset_model,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SLaM Pipeline__
"""

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

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

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

source_lp_result = source_lp(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    source_bulge=source_bulge,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_lp_result
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    use_jax=True,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_shape=mesh_shape,
)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    use_jax=True,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    use_jax=True,
)

light_result = light_lp(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

mass_result = mass_total(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
__Second Dataset Fits__

We now fit the secondary multi-wavelength datasets, which are lower resolution than the main dataset.

This uses a for loop to iterate over every waveband of every dataset, load and mask the data and fit it.

Each fit uses a fixed mass model, the lens and source light models update via linear algebra and offsets are
included (see full description above).
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
__Result Dict__

The results of each fit are stored in a dictionary, which is used to pass the results of each fit to the
visualization functions.
"""
multi_result_dict = {"g": mass_result}

for dataset_waveband, pixel_scale in zip(dataset_waveband_list, pixel_scale_list):

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

    aplt.subplot_imaging_dataset(dataset=dataset)

    settings_search = af.SettingsSearch(
        path_prefix=Path("slam", "multi", "independent"),
        unique_tag=f"{dataset_name}_data_{dataset_waveband}",
        info=None,
    )

    """
    __Dataset Model__

    For each secondary dataset, the (y,x) offset relative to the primary data is a free parameter.
    """
    dataset_model = af.Model(al.DatasetModel)

    dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
        lower_limit=-0.2, upper_limit=0.2
    )
    dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
        lower_limit=-0.2, upper_limit=0.2
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

    analysis = al.AnalysisImaging(dataset=dataset)

    source_lp_result = source_lp_secondary(
        settings_search=settings_search,
        analysis=analysis,
        light_result=light_result,
        mass_result=mass_result,
        source_bulge=source_bulge,
        dataset_model=dataset_model,
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    dataset_model.grid_offset.grid_offset_0 = (
        source_lp_result.instance.dataset_model.grid_offset[0]
    )
    dataset_model.grid_offset.grid_offset_1 = (
        source_lp_result.instance.dataset_model.grid_offset[1]
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        raise_inversion_positions_likelihood_exception=False,
    )

    source_pix_result_1 = source_pix_1_secondary(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        mesh_shape=mesh_shape,
        dataset_model=dataset_model,
    )

    source_pix_result_1.max_log_likelihood_fit.inversion.cls_list_from(cls=al.Mapper)[
        0
    ].extent_from()

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    dataset_model.grid_offset.grid_offset_0 = (
        source_lp_result.instance.dataset_model.grid_offset[0]
    )
    dataset_model.grid_offset.grid_offset_1 = (
        source_lp_result.instance.dataset_model.grid_offset[1]
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    multi_result = source_pix_2_secondary(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        mesh_shape=mesh_shape,
        dataset_model=dataset_model,
    )

    multi_result_dict[dataset_waveband] = multi_result

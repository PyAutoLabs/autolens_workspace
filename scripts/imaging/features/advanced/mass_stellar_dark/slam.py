"""
SLaM (Source, Light and Mass): Mass Light Dark
===============================================

This example shows how to use the SLaM pipelines to end with a mass model which decomposes the lens into its
stars and dark matter, using a light plus dark matter mass model.

Unlike other example SLaM pipelines, which end with the MASS TOTAL PIPELINE, this script ends with the
MASS LIGHT DARK PIPELINE.

__Contents__

**Model:** Compose the lens model fitted to the data.
**SOURCE LP PIPELINE:** Identical to `slam_start_here.py`, except.
**SOURCE PIX PIPELINE 1:** Identical to `slam_start_here.py`.
**SOURCE PIX PIPELINE 2:** Identical to `slam_start_here.py`.
**LIGHT LP PIPELINE:** Identical to `slam_start_here.py`, except the lens galaxy's bulge uses a linear `Sersic` light.
**MASS LIGHT DARK PIPELINE:** The MASS LIGHT DARK PIPELINE fits a mass model where the stellar mass is tied to the lens light.
**Dataset:** Load and plot the strong lens dataset.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
**SLaM Pipeline:** The code below calls the full SLaM PIPELINE.

__Model__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script
fits `Imaging` dataset of a strong lens system, where in the final model:

 - The lens galaxy's light is a `Sersic` linear light profile.
 - The lens galaxy's stellar mass distribution is a `Sersic` tied to the light model above.
 - The lens galaxy's dark matter mass distribution is modeled as a `NFWMCRLudlow`.
 - The source galaxy's light is a `Pixelization`.

Each SLaM pipeline is implemented as a Python function below, with a documentation string above each function
describing the pipeline in detail. The full pipeline is run at the bottom of the script.

__Start Here Notebook__

If any code in this script is unclear, refer to the `slam_start_here` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE__

Identical to `slam_start_here.py`, except:

 - The lens galaxy's bulge uses a linear `Sersic` light profile (`al.lp_linear.Sersic`) instead of an MGE.
 - The source galaxy's MGE uses `gaussian_per_basis=1`.

The linear `Sersic` profile is used here because the MASS LIGHT DARK PIPELINE requires a `LightMassProfile`
(`al.lmp.Sersic`) for the lens stellar mass, which shares the same profile type as the light model.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    redshift_lens: float,
    redshift_source: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_bulge = af.Model(al.lp_linear.Sersic)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=af.Model(al.mp.Isothermal),
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


"""
__SOURCE PIX PIPELINE 1__

Identical to `slam_start_here.py`.
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

    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    shear = source_lp_result.model.galaxies.lens.shear

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=shear,
            ),
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
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `slam_start_here.py`.

Note that between SOURCE PIX PIPELINE 1 and this search, the calling section applies adaptive over-sampling to
the dataset using the pixelized source reconstruction from search 1. This improves the accuracy of the
pixelization by ensuring the over-sampling is adapted to the source morphology.
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

Identical to `slam_start_here.py`, except the lens galaxy's bulge uses a linear `Sersic` light profile
(`al.lp_linear.Sersic`) instead of an MGE. This ensures the light model is consistent with the
MASS LIGHT DARK PIPELINE, which links stellar mass to a `Sersic` light profile.
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    lens_bulge = af.Model(al.lp_linear.Sersic)

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


"""
__MASS LIGHT DARK PIPELINE__

The MASS LIGHT DARK PIPELINE fits a mass model where the stellar mass is tied to the lens light model and a
separate dark matter halo is included.

The lens bulge is modeled as a `LightMassProfile` (`al.lmp.Sersic`) whose parameters are initialized from the
LIGHT LP PIPELINE result via `al.util.chaining.mass_light_dark_from`. The dark matter halo (`NFWMCRLudlow`) centre
is aligned with the stellar bulge centre.
"""


def mass_light_dark(
    settings_search: af.SettingsSearch,
    dataset,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    # Whether to use the gradient of the mass-to-light ratio profile.
    use_gradient = False
    # Whether to link the mass-to-light ratios of the bulge and disk to the same value.
    link_mass_to_light_ratios = True

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

    lp_chain_tracer = al.util.chaining.lp_chain_tracer_from(
        light_result=light_result, settings_search=settings_search
    )

    lens_bulge = al.util.chaining.mass_light_dark_from(
        light_result=light_result,
        lp_chain_tracer=lp_chain_tracer,
        name="bulge",
        use_gradient=use_gradient,
    )
    lens_disk = al.util.chaining.mass_light_dark_from(
        light_result=light_result,
        lp_chain_tracer=lp_chain_tracer,
        name="disk",
        use_gradient=use_gradient,
    )

    lens_bulge, lens_disk = al.util.chaining.link_ratios(
        link_mass_to_light_ratios=link_mass_to_light_ratios,
        light_result=light_result,
        bulge=lens_bulge,
        disk=lens_disk,
    )

    dark = af.Model(al.mp.NFWMCRLudlow)

    try:
        dark.centre = lens_bulge.centre
    except AttributeError:
        dark.centre = lens_bulge.profile_list[0].centre

    dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e10, upper_limit=1e15)
    dark.redshift_object = light_result.instance.galaxies.lens.redshift
    dark.redshift_source = light_result.instance.galaxies.source.redshift

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=light_result.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=lens_disk,
                dark=dark,
                shear=source_result_for_lens.model.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_light_dark[1]",
        **settings_search.search_dict,
        n_live=250,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/imaging/features/advanced/mass_stellar_dark/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
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
    path_prefix=Path("imaging") / "slam",
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

As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__

The code below calls the full SLaM PIPELINE. See the documentation string above each Python function for
a description of each pipeline step.

Between SOURCE PIX PIPELINE 1 and 2, adaptive over-sampling is applied to the dataset using the pixelized source
reconstruction from search 1. This improves the pixelization accuracy in search 2 and all subsequent pipelines.
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
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
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
)

mass_result = mass_light_dark(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
Finish.
"""

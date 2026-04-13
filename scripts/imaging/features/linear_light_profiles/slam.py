"""
Linear Light Profiles: SLaM
============================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines using linear light profiles
for the lens galaxy's light.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that guide before
working through this example.

The differences from `slam_start_here` are:

 - The SOURCE LP PIPELINE uses a `Sersic` linear light profile (`al.lp_linear.Sersic`) for the lens galaxy's
   bulge instead of an MGE.
 - The LIGHT LP PIPELINE uses a `Sersic` linear light profile for the lens galaxy's bulge instead of an MGE.

Linear light profiles solve for the `intensity` analytically via linear algebra, removing it from the non-linear
parameter space. This reduces the dimensionality of the fit and eliminates intensity-shape degeneracies, resulting
in more reliable and faster inference.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**SOURCE LP PIPELINE:** Identical to `slam_start_here.py`, except the lens galaxy's bulge uses a linear `Sersic` light.
**SOURCE PIX PIPELINE 1:** Identical to `slam_start_here.py`.
**SOURCE PIX PIPELINE 2:** Identical to `slam_start_here.py`.
**LIGHT LP PIPELINE:** Identical to `slam_start_here.py`, except the lens galaxy's bulge uses a linear `Sersic` light.
**MASS TOTAL PIPELINE:** Identical to `slam_start_here.py`.
**Dataset:** Load and plot the strong lens dataset.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
**SLaM Pipeline:** The code below calls the full SLaM PIPELINE.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

- **Linear Light Profiles** (`features/linear_light_profiles`)
  How linear light profiles work and their advantages over standard light profiles.

You can still run the script without fully understanding these guides, but reviewing them later will
make the structure and choices of the SLaM workflow clearer.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE__

Identical to `slam_start_here.py`, except the lens galaxy's bulge uses a linear `Sersic` light profile
(`al.lp_linear.Sersic`) instead of an MGE.

The linear `Sersic` profile has fewer free parameters than an MGE, as `intensity` is solved analytically.
This makes the source LP search faster and more reliable.
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
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
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
(`al.lp_linear.Sersic`) instead of an MGE.

Using the linear `Sersic` in the LIGHT LP PIPELINE ensures the lens-light model is consistent with the
SOURCE LP PIPELINE and reduces the number of free parameters in this fit.
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
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    # Total mass model for the lens galaxy.
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
    )

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    bulge = light_result.instance.galaxies.lens.bulge
    disk = light_result.instance.galaxies.lens.disk

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
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


"""
__Dataset__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple"
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
        [sys.executable, "scripts/imaging/simulator.py"],
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

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
Finish.
"""

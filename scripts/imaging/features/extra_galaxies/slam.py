"""
Extra Galaxies: SLaM
=====================

This script uses the SLaM pipelines to fit a lens dataset that includes extra galaxies
surrounding the main lens, whose light and mass are both included in the model.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. This script
only documents how the pipeline differs from that reference.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**Group SLaM:** This pipeline is designed for the galaxy-scale regime with a small number of nearby extra galaxies.
**This Script:** Using SOURCE LP, SOURCE PIX, LIGHT LP and MASS TOTAL pipelines this script fits `Imaging` data.
**SOURCE LP PIPELINE:** Identical to `slam_start_here.py`, except each extra galaxy is included in the model with a.
**SOURCE PIX PIPELINE 1:** Identical to `slam_start_here.py`, except extra-galaxy light is fixed from `source_lp` and their.
**SOURCE PIX PIPELINE 2:** Identical to `slam_start_here.py`, except extra galaxies are fully fixed as instances from.
**LIGHT LP PIPELINE:** Identical to `slam_start_here.py`, except extra-galaxy mass is fixed from `source_pix[1]` and their.
**MASS TOTAL PIPELINE:** Identical to `slam_start_here.py`, except extra-galaxy light is fixed from `light[1]` and their.
**Dataset:** Load and plot the strong lens dataset.
**Extra Galaxies Centres:** The centres of the extra galaxies are loaded from a `.json` file and used to set up the light and.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**Redshifts:** The redshifts of the lens and source galaxies.
**Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
**SLaM Pipeline:** Overview of slam pipeline for this example.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
- **Extra Galaxies** (`features/extra_galaxies.ipynb`)

__Group SLaM__

This pipeline is designed for the galaxy-scale regime with a small number of nearby extra
galaxies. For systems with many companions or group-scale complexity, use the group SLaM
pipeline (`scripts/group/slam.py`), which models companion masses via a shared luminosity
scaling relation rather than individually.

__This Script__

Using SOURCE LP, SOURCE PIX, LIGHT LP and MASS TOTAL pipelines this script fits `Imaging`
data where in the final model:

 - The lens galaxy's light is an MGE bulge.
 - The lens galaxy's total mass is a `PowerLaw` plus `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.
 - Each extra galaxy has a spherical MGE light profile and an `IsothermalSph` mass.

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
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE__

Identical to `slam_start_here.py`, except each extra galaxy is included in the model with a
spherical MGE light profile (`GaussianSph` basis) and a free `IsothermalSph` mass, both
centred on its known position from `extra_galaxies_centres`.
"""


def source_lp(
    settings_search,
    dataset,
    mask_radius,
    extra_galaxies_centres,
    redshift_lens,
    redshift_source,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    extra_galaxies_list = []

    for centre in extra_galaxies_centres:
        total_gaussians = 10

        log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

        gaussian_list = af.Collection(
            af.Model(al.lp_linear.GaussianSph) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre[0]
            gaussian.centre.centre_1 = centre[1]
            gaussian.sigma = 10 ** log10_sigma_list[i]

        extra_galaxy_bulge = af.Model(
            al.lp_basis.Basis, profile_list=list(gaussian_list)
        )

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

        extra_galaxies_list.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=extra_galaxy_bulge, mass=mass
            )
        )

    extra_galaxies = af.Collection(extra_galaxies_list)

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

Identical to `slam_start_here.py`, except extra-galaxy light is fixed from `source_lp` and
their `IsothermalSph` mass priors are carried forward as free parameters.
"""


def source_pix_1(
    settings_search,
    dataset,
    source_lp_result,
    mesh_shape,
    n_batch=20,
):
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

    extra_galaxies = source_lp_result.model.extra_galaxies
    for galaxy, result_galaxy in zip(
        extra_galaxies, source_lp_result.instance.extra_galaxies
    ):
        galaxy.bulge = result_galaxy.bulge

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

Identical to `slam_start_here.py`, except extra galaxies are fully fixed as instances
from `source_pix[1]`.
"""


def source_pix_2(
    settings_search,
    dataset,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    n_batch=20,
):
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
                    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
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

Identical to `slam_start_here.py`, except extra-galaxy mass is fixed from `source_pix[1]`
and their light is a new free spherical MGE centred on the same position.
"""


def light_lp(
    settings_search,
    dataset,
    mask_radius,
    source_result_for_lens,
    source_result_for_source,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    extra_galaxies = source_result_for_lens.model.extra_galaxies
    for galaxy, result_galaxy in zip(
        extra_galaxies, source_result_for_lens.instance.extra_galaxies
    ):
        galaxy.mass = result_galaxy.mass

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

Identical to `slam_start_here.py`, except extra-galaxy light is fixed from `light[1]` and
their `IsothermalSph` mass priors are carried forward as free parameters from `source_pix[1]`.
"""


def mass_total(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    n_batch=20,
):
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

    source = al.util.chaining.source_from(result=source_result_for_source)

    extra_galaxies = source_result_for_lens.model.extra_galaxies
    for galaxy, result_galaxy in zip(
        extra_galaxies, light_result.instance.extra_galaxies
    ):
        galaxy.bulge = result_galaxy.bulge

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
__Dataset__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "extra_galaxies"
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
        [sys.executable, "scripts/imaging/features/extra_galaxies/simulator.py"],
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxies Centres__

The centres of the extra galaxies are loaded from a `.json` file and used to set up the
light and mass models for each companion galaxy inside the pipeline functions.
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)] + list(extra_galaxies_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

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
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    extra_galaxies_centres=extra_galaxies_centres,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_shape=mesh_shape,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
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
Finish.
"""

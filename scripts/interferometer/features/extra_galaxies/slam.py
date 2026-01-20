"""
Extra Galaxies: SLaM
=====================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for fitting a
lens model where extra galaxies surrounding the lens are included in the lens model.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

This example only provides documentation specific to the extra galaxies, describing how the pipeline
differs from the standard SLaM pipelines described in the SLaM start here guide.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

- **Extra Galaxies** (`features/extra_galaxies.ipynb`):
    How we include extra galaxies in the lens model, by using the centres of the galaxies
    which have been determined beforehand.

You can still run the script without fully understanding the guide, but reviewing it later will
make the structure and choices of the SLaM workflow clearer.

__Group SLaM__

This SLaM pipeline is designed for the regime where one is modeling galaxy scale lenses with nearby surrounding
extra galaxies.

However, these systems can often become close to the group scale lensing regime, for which PyAutoLens has a dedicated
package for modeling (`autolens_workspace/*/group`) and its own dedicated SLaM pipelines.

The main difference between this SLaM pipeline and the group SLaM pipelines is that in the latter, the masses of
the extra galaxies are modeled using scaling relations tied to their light profiles. The group SLaM pipeline has
additional searches in the SOURCE LP PIPELINE to measure the luminosities of the extra galaxies for this purpose.

Which SLaM pipeline you should use depends on your particular strong lens, but as a rule of thumb if you are
including a lot of extra galaxies (e.g. more than 5) and your model complexity is increasing significantly, you should
consider using the group SLaM pipelines.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.
 - Two extra galaxies are included in the model, each with their mass as a `IsothermalSph` profile.

This modeling script uses the SLaM pipelines:

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
import slam_pipeline

"""
__Dataset + Masking__ 

Load the `Interferometer` data, define the visibility and real-space masks.
"""
dataset_name = "extra_galaxies"
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.1, radius=mask_radius
)

dataset_path = Path("dataset") / "interferometer" / dataset_name

# dataset_name = "alma"

# if dataset_name == "alma":
#
#     real_space_mask = al.Mask2D.circular(
#         shape_native=(800, 800),
#         pixel_scales=0.01,
#         radius=mask_radius,
#     )


dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

"""
__Extra Galaxies Centres__

This is the same API as described in the `features/extra_galaxies.ipynb` example, where the centres of the extra 
galaxies are loaded from a `.json` file.
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "extra_galaxies_centres.json"))
)

print(extra_galaxies_centres)


"""
__W_Tilde__

The `pixelization/modeling` example describes how the w-tilde formalism speeds up interferometer
pixelized source modeling, especially for many visibilities.

We use a try / except to load the pre-computed curvature preload, which is necessary to use
the w-tilde formalism. If this file does not exist (e.g. you have not made it manually via
the `many_visibilities_preparartion` example it is made here.
"""
try:
    curvature_preload = np.load(
        file=dataset_path / "curvature_preload.npy",
    )
except FileNotFoundError:
    curvature_preload = None

dataset = dataset.apply_w_tilde(
    curvature_preload=curvature_preload, use_jax=True, show_progress=True
)

"""
__Poisition Likelihood__

Load the multiple image positions used for the position likelihood, which resamles bad mass 
models and prevent demagnified solutions being inferred.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Settings__

Disable the default position only linear algebra solver so the source reconstruction can have 
negative pixel values.
"""
settings_inversion = al.SettingsInversion(use_positive_only_solver=False)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("interferometer") / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)


"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("interferometer", "slam"),
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
__JAX & Preloads__

The `features/pixelization/modeling` example describes how JAX required preloads in advance so it knows the 
shape of arrays it must compile functions for.
"""
mesh_shape = (30, 30)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

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

The SOURCE PIX PIPELINE is identical to the `slam_start_here.ipynb` example,  except the `extra_galaxies` are 
included in the model and passed to the pipeline.
"""
# Source Light

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxy.mass.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    preloads=preloads,
    settings_inversion=settings_inversion,
)

source_pix_result_1 = slam_pipeline.source_pix.run_1__bypass_lp(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=None,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Constant,
    extra_galaxies=extra_galaxies,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

As above, this pipeline also has the same API as the `slam_start_here.ipynb` example.

The extra galaxies are passed from the SOURCE PIX PIPELINE, via the `source_pix_result_1` object, therefore there is 
no need to manually pass them below.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1, use_model_images=True
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
)

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    settings_inversion=settings_inversion,
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_pix_result_1,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
)

"""
__MASS TOTAL PIPELINE__

As above, this pipeline also has the same API as the `slam_start_here.ipynb` example except for the extra galaxies.

The extra galaxies are set up using the same trick as the SOURCE PIX PIPELINE.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    positions_likelihood_list=[
        source_pix_result_1.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    settings_inversion=settings_inversion,
)

mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=None,
    mass=af.Model(al.mp.PowerLaw),
    extra_galaxies=source_pix_result_1.model.extra_galaxies,
)

"""
__Output__

The `start_hre.ipynb` example describes how results can be output to hard-disk after the SLaM pipelines have been run.
Checkout that script for a complete description of the output of this script.
"""

"""
Pixelization: SLaM
==================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for pixelized source modeling.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

Because the SLaM pipelines are designed around pixelized source modeling, the example `slam_start_here` fully
describes all design choices and modeling decisions made in this script. This script therefore does not repeat
that documentation, therefore `slam_start_here` should be read first.

The interferometer SLaM pipeline has one different from the imaging SLaM pipeline, it omits the `source_lp`
pipeline and does not fit a model with a light profile source. This is because fitting light profiles
to datasets with many visibilities is slow, whereas pixelized sources are fast. This has two consequences:

- You must provide the multiple image locations used for the position likelihoods manually, whereas for the imaging
  SLaM pipeline they are estimated via the lens model fit in the `source_lp` pipeline.

- `source_pix[1]` does not have an adapt image and therefore uses a regularization which is not adaptive.

Other than that, the interferometer SLaM pipeline is identical to the imaging SLaM pipeline.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

You can still run the script without fully understanding the guide, but reviewing it later will
make the structure and choices of the SLaM workflow clearer.

__Interferometer SLaM Description__

The `slam_start_here` notebook provides a detailed description of the SLaM pipelines, but it does this using CCD
imaging data.

There is no dedicated example which provides full descriptions of the SLaM pipelines using interferometer data, however,
the concepts and API described in the `slam_start_here` are identical to what is required for interferometer data.

Therefore, by reading the `slam_start_here` example you will fully understand everything required to use this
interferometer SLaM script.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autolens_workspace` repository:

https://github.com/Jammy2211/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autolens_workspace/dataset/interferometer/alma`

You can then perform modeling using this high-resolution dataset by uncommenting the relevant line of code
below.
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
dataset_name = "simple"
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

The SOURCE PIX PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
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
)

"""
__SOURCE PIX PIPELINE 2__

The SOURCE PIX PIPELINE 2 is identical to the `slam_start_here.ipynb` example.

Note that the LIGHT PIPELINE follows the SOURCE PIX PIPELINE in the `slam_start_here.ipynb` example is not included
in this script, given the lens light is not present in the data.
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

The MASS TOTAL PIPELINE is again identical to the `slam_start_here.ipynb` example, noting that the `light_result` is
now passed in as None to omit the lens light from the model.
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
)

"""
Finish.
"""

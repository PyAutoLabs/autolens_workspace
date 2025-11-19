"""
Pixelization: SLaM
==================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for pixelized source modeling.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

Because the SLaM pipelines are designed around pixelized source modeling, the example `slam_start_here` fully
describes all design choices and modeling decisions made in this script. This script therefore does not repeat
that documentation, therefore `slam_start_here` should be read first.

The interferometer SLaM pipeline is identical to the imaging SLaM pipeline, except that it uses
`AnalysisInterferometer` objects to fit interferometer datasets. However, the design and interface
themselves are the same.

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
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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

Load the `Interferometer` data, define the visibility and real-space masks and plot them.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(151, 151), pixel_scales=0.05, radius=mask_radius
)

dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
__Inversion Settings (Run Times)__

The `SettingsInversion` used to fit the interferometer data with a pixelized source should be chosen carefully, 
to ensure you get the fastest run times possible for your number of visibilities.

The example `autolens_workspace/*/interferometer/features/pixelization/run_times.py` provides a guide
on how to choose these settings based on your dataset.
"""
settings_inversion = al.SettingsInversion(
    use_linear_operators=False,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

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
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=None,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)


"""
__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data, primarily because the lens light is omitted.

- `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
  are forced to values of zero, a technique tests have shown are required to give accruate lens models.

The `image_mesh` can be ignored, it is legacy API from previous versions which may or may not be reintegrated in future
versions.
"""
image_mesh = None
mesh_shape = (20, 20)
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
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    settings_inversion=settings_inversion,
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    image_mesh_init=None,
    mesh_init=af.Model(al.mesh.RectangularMagnification, shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
)

"""
__SOURCE PIX PIPELINE 2__

The SOURCE PIX PIPELINE 2 is identical to the `slam_start_here.ipynb` example.

Note that the LIGHT PIPELINE follows the SOURCE PIX PIPELINE in the `slam_start_here.ipynb` example is not included
in this script, given the lens light is not present in the data.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
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
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE is again identical to the `slam_start_here.ipynb` example, noting that the `light_result` is
now passed in as None to omit the lens light from the model.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
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

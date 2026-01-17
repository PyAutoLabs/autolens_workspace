"""
Subhalo Detection: Start Here
=============================

Strong gravitational lenses can be used to detect the presence of small-scale dark matter (DM) subhalos. This occurs
when the DM subhalo overlaps the lensed source emission, and therefore gravitationally perturbs the observed image of
the lensed source galaxy.

When a DM subhalo is not included in the lens model, residuals will be present in the fit to the data in the lensed
source regions near the subhalo. By adding a DM subhalo to the lens model, these residuals can be reduced. Bayesian
model comparison can then be used to quantify whether or not the improvement to the fit is significant enough to
claim the detection of a DM subhalo.

The example illustrates DM subhalo detection with **PyAutoLens**.

__SLaM Pipelines__

The Source, (lens) Light and Mass (SLaM) pipelines are advanced lens modeling pipelines which automate the fitting
of complex lens models. The SLaM pipelines are used for all DM subhalo detection analyses. Therefore
you should be familiar with the SLaM pipelines before performing DM subhalo detection yourself. If you are unfamiliar
with the SLaM pipelines, checkout the
example `autolens_workspace/notebooks/guides/modeling/slam_start_here`.

Dark matter subhalo detection runs the standard SLaM pipelines, and then extends them with a SUBHALO PIPELINE which
performs the following three chained non-linear searches:

 1) Fits the lens model fitted in the MASS PIPELINE again, without a DM subhalo, to estimate the Bayesian evidence
    of the model without a DM subhalo.

 2) Performs a grid-search of non-linear searches, where each grid cell includes a DM subhalo whose (y,x) centre is
    confined to a small 2D section of the image plane via uniform priors (we explain this in more detail below).

 3) Fit the lens model again, including a DM subhalo whose (y,x) centre is initialized from the highest log evidence
    grid cell of the grid-search. The Bayesian evidence estimated in this model-fit is compared to the model-fit
    which did not include a DM subhalo, to determine whether or not a DM subhalo was detected.

__Grid Search__

The second stage of the SUBHALO PIPELINE uses a grid-search of non-linear searches to determine the highest log
evidence model with a DM subhalo. This grid search confines each DM subhalo in the lens model to a small 2D section
of the image plane via priors on its (y,x) centre. The reasons for this are as follows:

 - Lens models including a DM subhalo often have a multi-model parameter space. This means there are multiple lens
   models with high likelihood solutions, each of which place the DM subhalo in different (y,x) image-plane location.
   Multi-modal parameter spaces are synonomously difficult for non-linear searches to fit, and often produce
   incorrect or inefficient fitting. The grid search breaks the multi-modal parameter space into many single-peaked
   parameter spaces, making the model-fitting faster and more reliable.

 - By inferring how placing a DM subhalo at different locations in the image-plane changes the Bayesian evidence, we
   map out spatial information on where a DM subhalo is detected. This can help our interpretation of the DM subhalo
   detection.

__Pixelized Source__

Detecting a DM subhalo requires the lens model to be sufficiently accurate that the residuals of the source's light
are at a level where the subhalo's perturbing lensing effects can be detected.

This requires the source reconstruction to be performed using a pixelized source, as this provides a more detailed
reconstruction of the source's light than fits using light profiles.

This example therefore using a pixelized source and the corresponding SLaM pipelines.

The `subhalo/detection/examples` folder contains an example using light profile sources, if you have a use-case where
using light profile source is feasible (e.g. fitting simple simulated datasets).

__Model__

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS TOTAL PIPELINE and SUBHALO PIPELINE this SLaM script
fits `Interferometer` of a strong lens system, where in the final model:

 - The lens galaxy's light is an MGE bulge.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!
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
    mesh_init=af.Model(al.mesh.RectangularMagnification, shape=mesh_shape),
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
__SUBHALO PIPELINE (single plane detection)__

The SUBHALO PIPELINE (single plane detection) consists of the following searches:

 1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a
 subhalo. This uses the same model as fitted in the MASS TOTAL PIPELINE.
 2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo.
 3) If there is a successful detection a final search is performed to refine its parameters.

For this modeling script the SUBHALO PIPELINE customizes:

 - The [number_of_steps x number_of_steps] size of the grid-search, as well as the dimensions it spans in arc-seconds.
 - The `number_of_cores` used for the gridsearch, where `number_of_cores > 1` performs the model-fits in paralle using
 the Python multiprocessing module.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    preloads=preloads,
    positions_likelihood_list=[
        mass_result.positions_likelihood_from(
            factor=3.0,
            minimum_threshold=0.2,
        )
    ],
    adapt_images=adapt_images,
)

result_no_subhalo = slam_pipeline.subhalo.detection.run_1_no_subhalo(
    settings_search=settings_search,
    analysis=analysis,
    mass_result=mass_result,
)

result_subhalo_grid_search = slam_pipeline.subhalo.detection.run_2_grid_search(
    settings_search=settings_search,
    analysis=analysis,
    mass_result=mass_result,
    subhalo_result_1=result_no_subhalo,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

result_with_subhalo = slam_pipeline.subhalo.detection.run_3_subhalo(
    settings_search=settings_search,
    analysis=analysis,
    subhalo_result_1=result_no_subhalo,
    subhalo_grid_search_result_2=result_subhalo_grid_search,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
)

"""
Finish.
"""

"""
SLaM (Source, Light and Mass): Start Here
=========================================

This scripts gives an introduce to the Source, (lens) Light and Mass (SLaM) pipelines. These are advanced modeling
pipelines which use many aspects of core PyAutoLens functionality to automate the modeling of strong lenses.

__Preqrequisites__

Before using SLaM, you should understand:

- **pixelizations** (`*/features/pixelization`)
  Methods that reconstruct the source galaxy using a pixel grid.

- **Search Chaining** (`guides/modeling/chaining`)
  Fitting lens models in stages of increasing complexity, e.g. first a light-profile source,
  then a pixelized source.

- **Adaptive pixelizations** (`features/pixelization/adaptive`)
  pixelizations that adapt their mesh and regularization to the unlensed source morphology.

- **Multi-Gaussian Expansions (MGEs)** (`features/multi_gaussian_expansion`)
  Galaxy light modeled as a sum of Gaussians, enabling accurate lens-light subtraction.

You can still run the script without fully understanding these concepts; however, reviewing the
referenced examples later will clarify why SLaM pipelines are structured as they are.

__Overview__

SLaM chains together four or more sequential modeling searches, with each stage passing its results
forward to the next. This strategy enables fully automated modeling suitable for large samples of
strong lenses.

Each pipeline targets a specific part of the lens model:

1. **Source Pipeline**:
   Builds a reliable model of the source galaxy. For pixelized sources, this includes determining
   stable mesh and regularization parameters.

2. **Light Pipeline**:
   Models the lens galaxy’s light using the fixed source model from step 1. Accurate subtraction
   of lens light is essential for robust mass modeling.

3. **Mass Pipeline**:
   Fits the lens mass distribution (often complex), using the refined source and lens-light models
   from previous pipelines.


The SLaM workflow is flexible—you can swap MGE light profiles for other light models if desired. Models set up in
earlier pipelines guide those used in later ones. For example, if the Source Pipeline uses a `RectangularAdaptDensity`
mesh, the same mesh type is carried into later pipelines for consistency.
"
__Design Choices__

The structure of the SLaM pipelines is driven by the requirements of **adaptive pixelized source modeling**,
which is essential for fitting complex light and mass distributions for the lens.

Although SLaM also supports light-profile sources, pixelized sources are at the core of the pipeline design and
enable fully automated modeling of realistic, high-complexity mass models.

Below are the key design considerations that determine the ordering of SLaM pipelines:

- **Source First**
  Complex mass models (e.g., `PowerLaw`, or composite stellar + dark matter models) require pixelized
  source reconstruction, not simple light profiles. Therefore, SLaM begins with a source model using a
  simpler mass profile (e.g., `Isothermal` + `ExternalShear`) to provide a stable basis for later stages.

- **Image Positions**
  Pixelized modeling needs robust multiple image-position estimates to prevent unphysical source reconstructions.
  SLaM automatically determines these positions from the results of the Source Light Profile Pipeline.

- **Adapt Images**
  Advanced pixelizations use lens-light-subtracted “adapt images” to adapt the source pixelization mesh and
  regularization to the unlensed source morphology. These are only set once a sufficiently accurate source model
  is available from earlier stages in the Source Pipeline.

- **Lens Light Before Mass**
  Accurate lens-light subtraction is required before fitting complex mass models, especially for mass models
  fitting stellar and dark matter components simultanoeusly. Pixelized source modeling enables reliable
  deblending of the lens and source, so the lens light model is refined after the adaptive pixelized source is
  accuratel but before fitting more complex mass models.

- **Mass Model Last**
  The most flexible and complex mass models are fit only after high-quality source and lens-light models
  are established, ensuring stable priors and accurate mass inference.

Together, these design choices allow SLaM to perform precise, automated strong-lens modeling while maintaining
robustness and efficiency at each stage.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.

This modeling script uses the following SLaM pipelines found in the `autolens_workspace/slam_pipeline` package:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`
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
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

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

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

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
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in 
this example:

 - The lens galaxy's light is a MGE with 2 x 30 Gaussian [6 parameters]

 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
 
 - The source galaxy's light is a MGE with 1 x 30 Gaussian [4 parameters]

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
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

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data, primarily because the lens light is omitted.

- `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
  are forced to values of zero, a technique tests have shown are required to give accruate lens models.
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

The SOURCE PIX PIPELINE uses two searches to initialize a robust pixelized model of the source galaxy.
This pixelization adapts its resolution to the source morphology, assigning more pixels to brighter,
more detailed regions.

To build an adaptive pixelization, we require an **adapt image**: a lens-light-subtracted image in
which only the lensed source emission remains. This image determines how both the mesh density and
regularization weights adapt to the source structure.

The SOURCE LP Pipeline does not provide a sufficiently accurate source model for computing this adapt
image (e.g., the true source may be more complex than a simple light profile). Therefore, the first
step of the SOURCE PIX PIPELINE fits a new model using a pixelization, whose purpose is to generate
a high-quality adapt image.

This first search of the SOURCE PIX PIPELINE fits the following model:

- The lens galaxy light is modeled using MGE light profiles [parameters fixed to result of SOURCE LP PIPELINE].

- The lens galaxy mass is modeled using a total mass distribution [model initialized from the results of the SOURCE LP PIPELINE].

- The source galaxy's light is a pixelization using a `RectangularAdaptDensity` mesh and `AdaptiveBrightnessSplit` regularization scheme 
  [parameters of regularization free to vary].

This search improves the lens mass model by modeling the source using a pixelization and computes the adapt
images that are used in search 2.

The `AdaptiveBrightnessSplit` regularization adapt the source regularization weights to the source's morphology. We 
therefore set up the adapt image using the result from SOURCE LP PIPELINE. This image is not always perfect, but it
will be improved upon in search 2 and is good enough for computing the initial lens model in search 1.

__Positions__

In the pixelization examples, the importance of using multiple image positions to prevent unphysical source
reconstructions was discussed. 

This uses a `positions_likelihood` to ensure the lens model map the multiple images to within a threshold of one 
another in the source plane, else a penalty is added to the likelihood.

These examples required the user to manually input these positions. 

In SLaM, we automate this by computing the positions from the results of the SOURCE LP PIPELINE, which we can see
below come from the `source_lp_result` object.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_lp_result
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.AdaptiveBrightness,
)

"""
__SOURCE PIX PIPELINE 2__

Search 2 of the SOURCE PIX PIPELINE fits a lens model where:

- The lens galaxy light is modeled using MGE light profiles [parameters fixed to result of SOURCE LP PIPELINE].

- The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 1].

- The source galaxy's light is the input final mesh and regularization.

- The source galaxy's light is a pixelization using a `RectangularAdaptImage` mesh and `AdaptiveBrightnessSplit` regularization scheme 
  [parameters of regularization free to vary].

The `RectangularAdaptImage` mesh and `AdaptiveBrightness` regularization adapt the source pixels and regularization weights
to the source's morphology. We therefore set up the adapt image using the result from SOURCE PIX PIPELINE search 1.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    use_jax=True,
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE LP PIPELINE.
In this example it:

 - The lens galaxy's light is a MGE with 2 x 30 Gaussian [6 parameters] [6 Free Parameters].

 - Uses an `Isothermal` mass model with `ExternalShear` for the lens's total mass distribution [fixed from SOURCE PIX PIPELINE].

 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS PIPELINE [fixed values].   
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
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

The MASS TOTAL PIPELINE uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors and the lens 
light model of the LIGHT LP PIPELINE. 

In this example it:

 - Uses a linear Multi Gaussian Expansion bulge [fixed from LIGHT LP PIPELINE].

 - Uses an `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 LIGHT PROFILE PIPELINE + centre unfixed from (0.0, 0.0)].

 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
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
Finish.
"""

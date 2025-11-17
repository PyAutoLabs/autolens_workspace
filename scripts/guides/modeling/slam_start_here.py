"""
SLaM (Source, Light and Mass): Start Here
=========================================

This scripts gives an introduce to the Source, (lens) Light and Mass (SLaM) pipelines. These are advanced modeling
pipelines which use many aspects of core PyAutoLens functionality to automate the modeling of strong lenses.

__Preqrequisites__

Before reading this script, you should have familiarity with the following key concepts:

- **Non-linear Search Chaining:** This approach, demonstrated in `imaging/advanced/chaining`, shows the power of
  linking models together in a sequence, such as transitioning from a light profile source to a pixelized source.

- **Pixelizations:** These structures, explained in `features/pixelization.ipynb`, allow for the reconstruction of the
  source galaxy on a pixel grid.

- **Adaptive Pixelizations:** Described in `imaging/advanced/chaining/pix_adapt/start_here.ipynb`, these pixelizations
  adapt to the unlensed morphology of the source.

- **Multi Gaussian Expansions (MGE):** Introduced in `features/multi_gaussian_expansion.ipynb`, MGEs are employed to
  model the lens's light and can serve as an initialization for source galaxy light prior to using pixelization.

If any of these concepts are unfamiliar, you may still proceed with the script, but reviewing the referenced examples
later can deepen your understanding of how and why SLaM pipelines are structured as they are.

Additionally, this script allows for flexibility with model components, such as swapping out MGE models for other
light profiles (e.g., an MGE). For an example, see `examples/source_light_profile.ipynb`.

__Overview__

The Source, (Lens) Light, and Mass (SLaM) pipelines strategically chain together up to five sequential searches,
carefully designed to maximize the advantages of search chaining. This setup provides a fully automated framework for
fitting large samples of strong lenses with complex models, making SLaM the default choice for many **PyAutoLens**
research publications.

__Pipeline Structure__

Each pipeline in the SLaM sequence targets a specific aspect of the strong lens model:

- **Source Pipeline**: The first step focuses on establishing a robust source model. For pixelized sources,
  this includes obtaining accurate values for mesh and regularization parameters. For sources modeled with light
  profiles, the focus is on determining initial parameter estimates.

- **Light Pipeline**: This stage focuses on modeling the lens light, using source and mass models fixed from previous pipelines.

- **Mass Pipeline**: The final stage develops a detailed mass model, potentially of high complexity, leveraging source
  and lens light models initialized from earlier stages.

Models set up in earlier pipelines guide those used in later ones. For instance, if the Source Pipeline uses a
pixelized `RectangularMagnification` mesh for the source, that mesh type will carry through to the Mass Total Pipeline that follows.

__Design Choices__

There are many design choices that go into the SLaM pipelines, which we discuss now.

The SLaM pipelines are designed around pixelixed source modeling. Pixelized sources are necessary for fitting complex
mass models, which the SLaM pipelines automates the fitting of. However, the SLaM pipelines support fitting of
light profile sources, and using the SLaM pipelines in this way will still provide automated and robust lens modeling.

We now list the design considerations which dictate the ordering of the SLaM pipelines, which were driven by the use
of pixelized source modeling:

The SLaM pipelines involve several design choices to support the complexities of pixelized source modeling, which is
crucial for robustly fitting complex mass models. Here’s an overview of these key considerations and their influence
on the pipeline sequence:

- **Source First**: The pipeline starts with the Source Pipeline, as complex mass models (e.g., `PowerLaw` or
composite models with stars and dark matter) require pixelized source modeling rather than simple light profiles.
This step establishes a robust pixelized source model using a simpler mass model (like `Isothermal` with `Shear`).

- **Image Positions**: For pixelized source modeling, specifying the positions of the multiple images of the
lensed source(s) is crucial to prevent unphysical reconstructions. The SLaM pipelines can estimate these positions
automatically from the SOURCE LP PIPELINE's mass and source results.

- **Adapt Images**: Advanced pixelized source models use "adapt images" to optimize the mesh and regularization
  weights according to the source's morphology. The SLaM pipelines set the adapt-images once a good model for the source is
  available, enabling the best adaptation to the source structure.

- **Lens Light Before Mass**: Modeling the lens light accurately requires deblending the lens and source emissions,
  which a robust pixelized source model facilitates. This deblending, essential for certain mass models with both
  stellar and dark matter components, benefits from a simpler mass model during the lens light fitting stage.

- **Mass Model Last**: The most complex mass model fitting is saved for last. This final stage benefits from the
  prior refinement of the source and lens light models, ensuring accurate reconstructions and parameter estimations.

These design choices enable the SLaM pipelines to deliver precise and automated lens modeling while optimizing each
stage for robustness and efficiency.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script  fits `Imaging` dataset  of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.

This modeling script uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/chaining.ipynb` notebook.
"""

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
analysis = al.AnalysisImaging(dataset=dataset)

# Lens Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

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

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

# Source Light

centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

total_gaussians = 30
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

source_lp_result = slam.pipelinesource_lp.run(
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
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE uses two searches to initialize a robust model for the `Pixelization` that
reconstructs the source galaxy's light. 

This pixelization adapts its source pixels to the morphology of the source, placing more pixels in its 
brightest regions. To do this, an "adapt image" is required, which is the lens light subtracted image meaning
only the lensed source emission is present.

The SOURCE LP Pipeline result is not good enough quality to set up this adapt image (e.g. the source
may be more complex than a simple light profile). The first step of the SOURCE PIX PIPELINE therefore fits a new
model using a pixelization to create this adapt image.

The first search, which is an initialization search, fits an `Overlay` image-mesh, `RectangularMagnification` mesh 
and `AdaptiveBrightnessSplit` regularization.

__Adapt Images / Image Mesh Settings__

If you are unclear what the `adapt_images` and `SettingsInversion` inputs are doing below, refer to the 
`autolens_workspace/*/imaging/advanced/chaining/pix_adapt/start_here.py` example script.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

source_pix_result_1 = slam.pipelinesource_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.RectangularMagnification,
)

"""
__SOURCE PIX PIPELINE 2 (with lens light)__

The second search, which uses the mesh and regularization used throughout the remainder of the SLaM pipelines,
fits the following model:

- Uses a `Hilbert` image-mesh. 

- Uses a `RectangularMagnification` mesh.

 - Uses an `AdaptiveBrightnessSplit` regularization.
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.
 
The `Hilbert` image-mesh and `AdaptiveBrightness` regularization adapt the source pixels and regularization weights
to the source's morphology.

Below, we therefore set up the adapt image using this result.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

source_pix_result_2 = slam.pipelinesource_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.RectangularMagnification,
    regularization=al.reg.AdaptiveBrightnessSplit,
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
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
)

centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)

total_gaussians = 30
gaussian_per_basis = 2

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

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

lens_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

light_result = slam.pipelinelight_lp.run(
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
 PARAMETRIC PIPELINE + centre unfixed from (0.0, 0.0)].

 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

 - Carries the lens redshift and source redshift of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.

__Settings__:

 - adapt: We may be using adapt features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

mass_result = slam.pipelinemass_total.run(
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

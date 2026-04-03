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
   Models the lens galaxy's light using the fixed source model from step 1. Accurate subtraction
   of lens light is essential for robust mass modeling.

3. **Mass Pipeline**:
   Fits the lens mass distribution (often complex), using the refined source and lens-light models
   from previous pipelines.


The SLaM workflow is flexible—you can swap MGE light profiles for other light models if desired. Models set up in
earlier pipelines guide those used in later ones. For example, if the Source Pipeline uses a `RectangularAdaptDensity`
mesh, the same mesh type is carried into later pipelines for consistency.

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
  Advanced pixelizations use lens-light-subtracted "adapt images" to adapt the source pixelization mesh and
  regularization to the unlensed source morphology. These are only set once a sufficiently accurate source model
  is available from earlier stages in the Source Pipeline.

- **Lens Light Before Mass**
  Accurate lens-light subtraction is required before fitting complex mass models, especially for mass models
  fitting stellar and dark matter components simultanoeusly. Pixelized source modeling enables reliable
  deblending of the lens and source, so the lens light model is refined after the adaptive pixelized source is
  accurate but before fitting more complex mass models.

- **Mass Model Last**
  The most flexible and complex mass models are fit only after high-quality source and lens-light models
  are established, ensuring stable priors and accurate mass inference.

Together, these design choices allow SLaM to perform precise, automated strong-lens modeling while maintaining
robustness and efficiency at each stage.

__This Script__

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE, LIGHT LP PIPELINE and TOTAL MASS PIPELINE this SLaM modeling
script fits `Imaging` dataset of a strong lens system where in the final model:

 - The lens galaxy's light is a bulge with Multiple Gaussian Expansion (MGE) light profile.
 - The lens galaxy's total mass distribution is an `PowerLaw` plus an `ExternalShear`.
 - The source galaxy's light is a `Pixelization`.

Each SLaM pipeline is implemented as a Python function below (e.g. `source_lp`, `source_pix_1`), with a
documentation string above each function describing the pipeline in detail. The full pipeline is run at the
bottom of the script.
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

The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light, which in
this example:

 - Models the lens galaxy's light as an MGE with 2 x 20 Gaussians.
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
 - Models the source galaxy's light as an MGE with 1 x 20 Gaussians.

The mass and source models from this search initialize the SOURCE PIX PIPELINE searches that follow.
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

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

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

The SOURCE PIX PIPELINE uses two searches to initialize a robust pixelized model of the source galaxy.
This pixelization adapts its resolution to the source morphology, assigning more pixels to brighter,
more detailed regions.

To build an adaptive pixelization, we require an **adapt image**: a lens-light-subtracted image in
which only the lensed source emission remains. This image determines how both the mesh density and
regularization weights adapt to the source structure.

The SOURCE LP Pipeline does not provide a sufficiently accurate source model for computing this adapt
image (e.g., the true source may be more complex than a simple light profile). Therefore, the first
search of the SOURCE PIX PIPELINE fits a model using a pixelization whose purpose is to generate a
high-quality adapt image used in search 2.

__Positions__

Image positions are used to prevent unphysical source reconstructions (see `features/pixelization` for
details). Rather than being input manually, they are computed automatically from the SOURCE LP result.
This is a key automation feature of SLaM.

__Adapt Images__

An adapt image is computed from the SOURCE LP result and passed to the analysis. This provides an initial
estimate of the source morphology for the `Adapt` regularization scheme, even though the MGE source model
may not fully capture the source structure. Search 2 improves upon this using a pixelized adapt image.
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
            source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
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

The second search of the SOURCE PIX PIPELINE fits the final pixelized source model using the improved
adapt images computed from search 1's pixelized source reconstruction.

The `RectangularAdaptImage` mesh and `Adapt` regularization adapt the source pixels and regularization
weights to the source's morphology using the high-quality adapt images from search 1.
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

The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, with
the lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PIX
PIPELINE.

In this example:

 - The lens galaxy's light is a MGE with 2 x 20 Gaussians.
 - Uses an `Isothermal` mass model with `ExternalShear` for the lens's total mass distribution [fixed from
   SOURCE PIX PIPELINE].
 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

This search aims to produce an accurate model of the lens galaxy's light, which may not have been possible
in the SOURCE PIPELINE as the mass and source models were not yet precisely estimated. The adapt images
from SOURCE PIX PIPELINE search 1 are reused, providing a stable basis for the lens-light subtraction.
"""
def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
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

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

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

The MASS TOTAL PIPELINE uses one search to fit a complex lens mass model to a high level of accuracy,
using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize model priors, and the
lens light model of the LIGHT LP PIPELINE.

In this example:

 - Uses a linear MGE bulge [fixed from LIGHT LP PIPELINE].
 - Uses a `PowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE PIX
   PIPELINE].
 - Uses a `Pixelization` for the source's light [fixed from SOURCE PIX PIPELINE].

__Positions__

Positions are computed from the SOURCE PIX PIPELINE search 2 result, which provides more precise multiple
image positions than the SOURCE LP PIPELINE (as the pixelized source gives a better source-plane
reconstruction).
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

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

__Contents__

- **SLaM Pipelines:** The Source, (lens) Light and Mass (SLaM) pipelines are advanced lens modeling pipelines which.
- **Grid Search:** The second stage of the SUBHALO PIPELINE uses a grid-search of non-linear searches to determine the.
- **Pixelized Source:** Detecting a DM subhalo requires the lens model to be sufficiently accurate that the residuals of.
- **Model:** Compose the lens model fitted to the data.
- **SOURCE LP PIPELINE:** Initializes the mass model + source-light using MGE light profiles via `TransformerNUFFT`.
- **SOURCE PIX PIPELINE 1:** Initializes a pixelized source using the adapt image from the SOURCE LP result.
- **SOURCE PIX PIPELINE 2:** Improves the pixelized source using adapt images from `source_pix_result_1`.
- **MASS TOTAL PIPELINE:** Identical to `slam_start_here.py`, except no lens light model is included as interferometer data.
- **Two Datasets:** Build one Interferometer with `TransformerNUFFT` (source_lp) and one with `TransformerNUFFT` + sparse operator (source_pix onwards).
- **Sparse Operators:** The `pixelization/modeling` example describes how the sparse operator formalism speeds up.
- **Settings:** Disable the default position only linear algebra solver so the source reconstruction can have.
- **Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
- **Redshifts:** The redshifts of the lens and source galaxies.
- **Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
- **SLaM Pipeline:** The code below calls the full SLaM PIPELINE.

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
 `subhalo/detect`

Check them out for a full description of the analysis!
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__SOURCE LP PIPELINE__

Initializes a robust mass + source-light model using a Multi Gaussian Expansion (MGE), fit with light profiles.

This stage uses `dataset_nufft` (built with `TransformerNUFFT`, backed by JAX-native `nufftax`), which makes
light-profile fitting fast even on ALMA-class datasets with millions of visibilities. The result provides the
adapt image and position likelihood threaded into the pixelized pipelines that follow.

No lens light is fitted: interferometer data does not contain lens light emission.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    redshift_lens: float,
    redshift_source: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=5, centre_prior_is_uniform=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=None,
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

The first search of the SOURCE PIX PIPELINE fits a pixelization whose purpose is to generate a high-quality
adapt image used in search 2. It uses the adapt image computed from the SOURCE LP result, with the position
likelihood derived automatically via `source_lp_result.positions_likelihood_from(...)`.

This stage uses `dataset_sparse` (built with `TransformerNUFFT` + `apply_sparse_operator`), which makes the
inversion cost independent of visibility count (the sparse operator supports linear light profiles as well as
pixelizations, so a single sparse dataset can serve every stage).
"""


def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    mesh_init,
    regularization_init,
    settings,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        settings=settings,
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
                bulge=None,
                disk=None,
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

Identical to `slam_start_here.py`, using adapt images from `source_pix_result_1` to improve the source
pixelization and regularization.

Note that the LIGHT LP PIPELINE from `slam_start_here.py` is omitted here, as interferometer data does not
contain lens light emission.
"""


def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    mesh,
    regularization,
    settings,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        settings=settings,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                # interferometer data does not contain lens light emission
                bulge=None,
                disk=None,
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
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`, except no lens light model is included as interferometer data does not
contain lens light emission.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    source_pix_result_2: af.Result,
    settings,
    n_batch: int = 20,
) -> af.Result:
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_1.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        settings=settings,
    )

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_pix_result_1.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    source = al.util.chaining.source_from(result=source_pix_result_2)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_pix_result_1.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=mass,
                shear=source_pix_result_1.model.galaxies.lens.shear,
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
__SUBHALO PIPELINE (grid search)__

The second search of the SUBHALO PIPELINE performs a [number_of_steps x number_of_steps] grid search of
non-linear searches. Each grid cell includes a DM subhalo whose (y,x) centre is confined to a small 2D section
of the image plane via uniform priors.

This grid search maps out where in the image plane including a DM subhalo provides a better fit to the data.
"""


def subhalo_grid_search(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_mass: af.Model,
    settings,
    grid_dimension_arcsec: float = 3.0,
    number_of_steps: int = 2,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
        settings=settings,
    )

    subhalo = af.Model(al.Galaxy, mass=subhalo_mass)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )

    subhalo.redshift = mass_result.instance.galaxies.lens.redshift
    subhalo.mass.redshift_object = mass_result.instance.galaxies.lens.redshift
    subhalo.mass.redshift_source = mass_result.instance.galaxies.source.redshift

    lens = mass_result.model.galaxies.lens
    source = al.util.chaining.source_from(result=mass_result)

    model = af.Collection(
        galaxies=af.Collection(lens=lens, subhalo=subhalo, source=source),
    )

    search = af.Nautilus(
        name="subhalo[1]_[search_lens_plane]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    subhalo_grid_search = af.SearchGridSearch(
        search=search,
        number_of_steps=number_of_steps,
    )

    return subhalo_grid_search.fit(
        model=model,
        analysis=analysis,
        grid_priors=[
            model.galaxies.subhalo.mass.centre_1,
            model.galaxies.subhalo.mass.centre_0,
        ],
        info=settings_search.info,
    )


"""
__SUBHALO PIPELINE (refine)__

The third search of the SUBHALO PIPELINE refits the lens model including a DM subhalo, initializing the
subhalo centre from the highest log evidence grid cell of the grid search.

The Bayesian evidence from this fit is compared to the no-subhalo fit to determine whether a DM subhalo
was detected.
"""


def subhalo_refine(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    mass_result: af.Result,
    subhalo_grid_search_result: af.Result,
    subhalo_mass: af.Model,
    settings,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            mass_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
        settings=settings,
    )

    subhalo = af.Model(
        al.Galaxy,
        redshift=mass_result.instance.galaxies.lens.redshift,
        mass=subhalo_mass,
    )

    subhalo.redshift = mass_result.instance.galaxies.lens.redshift
    subhalo.mass.redshift_object = mass_result.instance.galaxies.lens.redshift
    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre = subhalo_grid_search_result.model_centred_absolute(
        a=1.0
    ).galaxies.subhalo.mass.centre

    subhalo.redshift = subhalo_grid_search_result.model.galaxies.subhalo.redshift
    subhalo.mass.redshift_object = subhalo.redshift

    model = af.Collection(
        galaxies=af.Collection(
            lens=subhalo_grid_search_result.model.galaxies.lens,
            subhalo=subhalo,
            source=subhalo_grid_search_result.model.galaxies.source,
        ),
    )

    search = af.Nautilus(
        name="subhalo[2]_[single_plane_refine]",
        **settings_search.search_dict,
        n_live=600,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


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

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

# dataset_name = "alma"

# if dataset_name == "alma":
#
#     real_space_mask = al.Mask2D.circular(
#         shape_native=(800, 800),
#         pixel_scales=0.01,
#         radius=mask_radius,
#     )


"""
__Two Datasets__

The SLaM pipeline runs in two phases that prefer different transformers:

- `dataset_nufft` uses `TransformerNUFFT` (backed by JAX-native `nufftax`) for the `source_lp` stage.
- `dataset_sparse` uses `TransformerNUFFT` + `apply_sparse_operator(...)` for `source_pix_1`,
  `source_pix_2`, `mass_total`, and every search of the SUBHALO PIPELINE.

Both datasets are built from the same FITS files; only the transformer (and sparse-operator preload) differ.
"""
dataset_nufft = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

dataset_sparse = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

"""
__Sparse Operators__

The `pixelization/modeling` example describes how the sparse operator formalism speeds up interferometer
pixelized source modeling, especially for many visibilities.

We use a try / except to load the pre-computed curvature preload, which is necessary to use
the sparse operator formalism. If this file does not exist (e.g. you have not made it manually via
the `many_visibilities_preparation` example) it is made here.

The sparse operator is applied only to `dataset_sparse` here; `source_lp` can also be run on a sparse dataset
if you prefer a single dataset for the whole pipeline.
"""
try:
    nufft_precision_operator = np.load(
        file=dataset_path / "nufft_precision_operator.npy",
    )
except FileNotFoundError:
    nufft_precision_operator = None

dataset_sparse = dataset_sparse.apply_sparse_operator(
    nufft_precision_operator=nufft_precision_operator, use_jax=True, show_progress=True
)

"""
__Settings__

Disable the default position only linear algebra solver so the source reconstruction can have
negative pixel values.
"""
settings = al.Settings(use_positive_only_solver=False)

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

Note the transformer split: `source_lp` is passed `dataset_nufft` (TransformerNUFFT), while every later stage
— including the SUBHALO PIPELINE — is passed `dataset_sparse` (TransformerNUFFT + sparse operator).
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset_nufft,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset_sparse,
    source_lp_result=source_lp_result,
    mesh_init=af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Adapt,
    settings=settings,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset_sparse,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape),
    regularization=al.reg.Adapt,
    settings=settings,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset_sparse,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    settings=settings,
)

result_subhalo_grid_search = subhalo_grid_search(
    settings_search=settings_search,
    dataset=dataset_sparse,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    settings=settings,
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

result_with_subhalo = subhalo_refine(
    settings_search=settings_search,
    dataset=dataset_sparse,
    source_pix_result_1=source_pix_result_1,
    mass_result=mass_result,
    subhalo_grid_search_result=result_subhalo_grid_search,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    settings=settings,
)

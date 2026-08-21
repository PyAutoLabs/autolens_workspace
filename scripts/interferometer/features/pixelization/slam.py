"""
Pixelization: SLaM
==================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for pixelized source modeling.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that
guide before working through this example.

Because the SLaM pipelines are designed around pixelized source modeling, the example `slam_start_here` fully
describes all design choices and modeling decisions made in this script. This script therefore does not repeat
that documentation, therefore `slam_start_here` should be read first.

The interferometer SLaM pipeline closely mirrors the imaging SLaM pipeline, with one notable difference:

- It uses **two datasets with different transformers**. The `source_lp` stage uses `TransformerNUFFT` (backed
  by the JAX-native `nufftax`, https://github.com/GragasLab/nufftax) so light-profile fitting runs at full GPU
  speed even on ALMA-class datasets with millions of visibilities. The `source_pix` and `mass` stages switch to
  `TransformerNUFFT` combined with the pre-computed sparse operator, because pixelized source reconstructions
  exploit sparsity rather than the NUFFT path.

The interferometer SLaM pipeline still omits the `light_lp` stage, because interferometer data does not contain
lens light emission. The lens galaxy therefore has no `bulge`/`disk` light components anywhere in the pipeline.

Now that `source_lp` is included, the position likelihood and adapt image needed by the pixelized pipelines are
derived automatically from the `source_lp` result (mirroring the imaging pipeline). Manual position input is no
longer required.

__Contents__

- **Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
- **Interferometer SLaM Description:** The `slam_start_here` notebook provides a detailed description of the SLaM pipelines, but it does.
- **High Resolution Dataset:** A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts.
- **SOURCE LP PIPELINE:** Initializes the mass and source-light model with MGE light profiles, fitted via `TransformerNUFFT`.
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

https://github.com/PyAutoLabs/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autolens_workspace/dataset/interferometer/alma`

You can then perform modeling using this high-resolution dataset by uncommenting the relevant line of code
below.
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

The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light using a Multi
Gaussian Expansion (MGE). The lens galaxy mass and external shear are fitted at the same time.

This stage uses the `dataset_nufft` Interferometer (built with `TransformerNUFFT`, backed by `nufftax`), which
makes light-profile fitting fast even for ALMA-class datasets with millions of visibilities.

The mass and source models from this search initialize the SOURCE PIX PIPELINE searches that follow, and the
result also provides the adapt image and position likelihood used by those later stages.

Note that no lens light is fitted: interferometer data does not contain lens light emission, so `lens.bulge` and
`lens.disk` are kept at `None`.
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
                # interferometer data does not contain lens light emission
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

The SOURCE PIX PIPELINE uses two searches to initialize a robust pixelized model of the source galaxy.

The first search fits a pixelization whose purpose is to generate a high-quality adapt image used in
search 2. It uses the adapt image computed from the SOURCE LP result (passed via `source_lp_result`) and the
position likelihood is also derived automatically from the SOURCE LP result via
`source_lp_result.positions_likelihood_from(...)` — no manual positions input is required.

This stage uses the `dataset_sparse` Interferometer (built with `TransformerNUFFT` + `apply_sparse_operator`).
The NUFFT keeps the one-time dirty-image setup tractable at ALMA-scale visibility counts, and the precomputed
sparse operator makes per-likelihood curvature assembly use the FFT-based W̃ precision matrix instead of the
dense `transformed_mapping_matrix`.
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

Like SOURCE PIX PIPELINE 1, this stage uses the `dataset_sparse` Interferometer.
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
                # interferometry does not support lens light
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

- `dataset_nufft` uses `TransformerNUFFT` (backed by JAX-native `nufftax`) for the `source_lp` stage. With
  light profiles this is the fast path at any visibility count, including ALMA-class datasets.
- `dataset_sparse` uses `TransformerNUFFT` combined with `apply_sparse_operator(...)` for `source_pix_1`,
  `source_pix_2` and `mass_total`. Pixelized source reconstructions exploit sparsity in the linear inversion
  rather than the NUFFT, so this combination is the right choice for the pixelized stages.

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

The sparse operator is applied only to `dataset_sparse` — the NUFFT-backed `dataset_nufft` used by
`source_lp` does not need it.
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
is passed `dataset_sparse` (TransformerNUFFT + sparse operator).
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

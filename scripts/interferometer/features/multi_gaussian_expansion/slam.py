"""
Multi Gaussian Expansion: SLaM (Interferometer)
================================================

This script provides an example of the Source, (Lens) Light, and Mass (SLaM) pipelines for `Interferometer`
data, using a **multi-Gaussian expansion (MGE)** for the source galaxy in the SOURCE LP stage.

A full overview of SLaM is provided in `guides/modeling/slam_start_here`. You should read that guide
before working through this example.

The interferometer pixelization SLaM pipeline at `interferometer/features/pixelization/slam.py` is the
closest analogue — this script keeps the same pipeline structure (SOURCE LP → SOURCE PIX 1 → SOURCE PIX 2
→ MASS TOTAL) and only differs in the size of the MGE source basis used in SOURCE LP. The LIGHT LP
pipeline from `slam_start_here.py` is omitted because interferometer data does not contain lens light
emission.

The MGE captures the asymmetric morphology of typical sub-mm/radio-selected lensed sources better than a
single `Sersic`, while keeping the non-linear parameter space small (only the shared Gaussian centre and
elliptical components are free; the per-Gaussian intensities are solved analytically and the sigmas are
fixed). This makes SOURCE LP fast, robust, and a better initialiser for the pixelized stages that follow.

The SOURCE LP stage uses `TransformerNUFFT` (backed by JAX-native `nufftax`,
https://github.com/GragasLab/nufftax). MGE fits to visibilities are practical at any visibility count
because the per-iteration NUFFT of every Gaussian basis component runs inside the same compiled likelihood
as the rest of the model.

The SOURCE PIX and MASS TOTAL stages use the same `TransformerNUFFT` plus a pre-computed sparse
operator (via `apply_sparse_operator`) so the pixelized source curvature matrix is assembled via the
FFT-based W̃ precision matrix instead of the dense `transformed_mapping_matrix`.

__Contents__

- **Prerequisites:** Before using this SLaM pipeline, you should be familiar with `slam_start_here`.
- **SOURCE LP PIPELINE:** Initializes the mass and source-light model with an MGE source bulge, fitted
  via `TransformerNUFFT`.
- **SOURCE PIX PIPELINE 1:** Initializes a pixelized source using the adapt image from the SOURCE LP
  result.
- **SOURCE PIX PIPELINE 2:** Improves the pixelized source using adapt images from `source_pix_result_1`.
- **MASS TOTAL PIPELINE:** Identical to `slam_start_here.py`, except no lens light model is included.
- **Two Datasets:** Build one Interferometer with `TransformerNUFFT` (source_lp) and one with
  `TransformerNUFFT` + sparse operator (source_pix onwards).
- **Sparse Operators:** Pre-compute the sparse operator for the pixelized stages.
- **Settings:** Disable the positive-only solver so the pixelized source reconstruction can have negative
  pixel values.
- **Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization,
  database use, etc.
- **Redshifts:** The redshifts of the lens and source galaxies.
- **Mesh Shape:** As discussed in `features/pixelization/modeling`, the mesh shape is fixed before
  modeling.
- **SLaM Pipeline:** The code below calls the full SLaM PIPELINE.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines and how they
  integrate into strong-lens modeling.

- **Multi Gaussian Expansion** (`interferometer/features/multi_gaussian_expansion/modeling`)
  How an MGE source basis is composed for interferometer data, and why nufftax makes large-N MGE bases
  practical.

- **Interferometer Pixelization SLaM** (`interferometer/features/pixelization/slam`)
  The canonical interferometer SLaM pipeline. This script is essentially that pipeline with a larger MGE
  source basis in SOURCE LP.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large
files which are too big to include in the main `autolens_workspace` repository:

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

The SOURCE LP PIPELINE uses one search to initialize a robust model for the source galaxy's light using a
multi-Gaussian expansion (MGE). The lens galaxy mass and external shear are fitted at the same time.

The MGE source is built via the `al.model_util.mge_model_from` helper, which constructs a basis of N
linear `Gaussian` profiles arranged in two groups (each sharing a centre and ell_comps; sigmas fixed to
log-spaced values). For interferometer data we use `total_gaussians=5`, which is enough to capture the
source morphology while keeping per-iteration cost manageable.

This stage uses the `dataset_nufft` Interferometer (built with `TransformerNUFFT`, backed by `nufftax`),
which makes the per-iteration NUFFT of every Gaussian fast even for ALMA-class datasets with millions of
visibilities.

The mass and source models from this search initialize the SOURCE PIX PIPELINE searches that follow, and
the result also provides the adapt image and position likelihood used by those later stages.

Note that no lens light is fitted: interferometer data does not contain lens light emission, so
`lens.bulge` and `lens.disk` are kept at `None`.
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
search 2. It uses the adapt image computed from the SOURCE LP result and the position likelihood is also
derived automatically from the SOURCE LP result — no manual positions input is required.

This stage uses the `dataset_sparse` Interferometer (built with `TransformerNUFFT` +
`apply_sparse_operator`). The NUFFT keeps the one-time dirty-image setup tractable for ALMA-scale
visibility counts, and the precomputed sparse operator makes per-likelihood curvature assembly use the
FFT-based W̃ precision matrix instead of the dense `transformed_mapping_matrix`.
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

The LIGHT LP PIPELINE from `slam_start_here.py` is omitted here, as interferometer data does not contain
lens light emission.

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

Load the `Interferometer` data and define the real-space mask.
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
simulator script.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

# dataset_name = "alma"
#
# if dataset_name == "alma":
#
#     real_space_mask = al.Mask2D.circular(
#         shape_native=(800, 800),
#         pixel_scales=0.01,
#         radius=mask_radius,
#     )


"""
__Two Datasets__

Both stages use `TransformerNUFFT` (backed by JAX-native `nufftax`), which keeps the dirty-image setup
tractable at any visibility count. The two datasets differ only in whether `apply_sparse_operator` has
been called:

- `dataset_nufft` is plain `TransformerNUFFT` for the `source_lp` stage. With an MGE source bulge each
  Gaussian's NUFFT runs inside the same compiled likelihood.
- `dataset_sparse` is the same `TransformerNUFFT` plus `apply_sparse_operator(...)` for `source_pix_1`,
  `source_pix_2`, and `mass_total`. The precomputed sparse operator makes the pixelized curvature matrix
  assembly use the FFT-based W̃ precision matrix instead of the dense `transformed_mapping_matrix`.

Both datasets are built from the same FITS files; only the transformer (and sparse-operator preload)
differ.
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

We use a try / except to load the pre-computed curvature preload, which is necessary to use the sparse
operator formalism. If this file does not exist it is made here.

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

Disable the default positive-only linear algebra solver so the pixelized source reconstruction can have
negative pixel values. (The MGE source bulge in SOURCE LP is still constrained to positive intensities
internally because each Gaussian's intensity is a physical normalization.)
"""
settings = al.Settings(use_positive_only_solver=False)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("interferometer") / "slam_multi_gaussian_expansion",
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

Both datasets use `TransformerNUFFT`. `source_lp` is passed `dataset_nufft` (no sparse operator) while
every later stage is passed `dataset_sparse` (`TransformerNUFFT` + `apply_sparse_operator`).
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
    mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
    regularization_init=al.reg.Adapt,
    settings=settings,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset_sparse,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
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

"""
Finish.
"""

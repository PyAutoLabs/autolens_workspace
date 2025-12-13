"""
Pixelization: CPU Fast Modeling
===============================

This example demonstrates how to achieve **fast pixelization performance on a CPU without JAX**, by combining:

- `numba` for optimized numerical routines, and
- Python `multiprocessing` to exploit multiple CPU cores.

On machines with many CPU cores (e.g. HPC clusters with >10 CPUs), this method can **outperform JAX GPU acceleration**
for pixelized source modeling. The advantage arises because pixelizations rely heavily on **sparse linear algebra**,
which is not currently optimized in JAX.

> Note: This performance advantage applies **only to pixelized sources**.
> For parametric sources or multi-Gaussian models, JAX (especially with a GPU) is significantly faster, and even JAX
> on a CPU outperforms the `numba` approach shown here.

__Run Time Overview__

Pixelized source reconstructions can be computed using either GPU acceleration via JAX or CPU acceleration via `numba`.
The faster option depends on two crucial factors:

#### **1. GPU VRAM Limitations**
JAX only provides significant acceleration on GPUs with **large VRAM (≥16 GB)**.
To avoid excessive VRAM usage, examples often restrict pixelization meshes (e.g. 20 × 20).
On consumer GPUs with limited memory, **JAX may be slower than CPU execution**.

#### **2. Sparse Matrix Performance**

Pixelized source reconstructions require operations on **very large, highly sparse matrices**.

- JAX currently lacks sparse-matrix support and must compute using **dense matrices**, which scale poorly.
- PyAutoLens’s CPU implementation (via `numba`) fully exploits sparsity, providing large speed gains
  at **high image resolution** (e.g. `pixel_scales <= 0.03`).

As a result, CPU execution can outperform JAX even on powerful GPUs for high-resolution datasets.

__Rule of Thumb__

For **low-resolution imaging** (for example, datasets with `pixel_scales > 0.05`), modeling is generally faster using
**JAX with a GPU**, because the computations involve fewer sparse operations and do not require large amounts of VRAM.

For **high-resolution imaging** (for example, `pixel_scales <= 0.03`), modeling can be faster using a **CPU with numba**
and multiple cores. At high resolution, the linear algebra is dominated by sparse matrix operations, and the CPU
implementation exploits sparsity more effectively, especially on systems with many CPU cores (e.g. HPC clusters).

**Recommendation:** The best choice depends on your hardware and dataset, so it is always worth benchmarking both
approaches (GPU+JAX vs CPU+numba) to determine which performs fastest for your case.
"""

try:
    import numba
except ModuleNotFoundError:
    input(
        "##################\n"
        "##### NUMBA ######\n"
        "##################\n\n"
        """
        Numba is not currently installed.

        Numba is a library which makes PyAutoLens run a lot faster. Certain functionality is disabled without numba
        and will raise an exception if it is used.

        If you have not tried installing numba, I recommend you try and do so now by running the following 
        commands in your command line / bash terminal now:

        pip install --upgrade pip
        pip install numba

        If your numba installation raises an error and fails, you should go ahead and use PyAutoLens without numba to 
        decide if it is the right software for you. If it is, you should then commit time to bug-fixing the numba
        installation. Feel free to raise an issue on GitHub for support with installing numba.

        A warning will crop up throughout your *PyAutoLens** use until you install numba, to remind you to do so.

        [Press Enter to continue]
        """
    )

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking + Positions__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__no_lens_light"
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

positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

"""
__W_Tilde__

Pixelized source modeling requires heavy linear algebra operations. These calculations can be greatly accelerated
using an alternative mathematical approach called the **`w_tilde` formalism**.

You do not need to understand the full details of the method, but the key point is:

- `w_tilde` exploits the **sparsity** of the matrices used in pixelized source reconstruction.
- This leads to a **significant speed-up on CPUs**.
- The current implementation does **not support JAX**, and therefore does not benefit from GPU acceleration.

To enable this feature, we call `apply_w_tilde()` on the `Imaging` dataset. This computes and stores a `w_tilde`
matrix, which is then reused in all subsequent pixelized source fits.

- Computing `w_tilde` takes anywhere from a few seconds to a few minutes, depending on the dataset size.
- After it is computed once, every model-fit using pixelization becomes substantially faster.
"""
dataset_w_tilde = dataset.apply_w_tilde()

"""
__JAX & Preloads__

In earlier examples (`imaging/features/pixelization/modeling`), we used JAX, which requires *preloading* array shapes
before compilation. In contrast, CPU modeling with `w_tilde` does **not** require JAX, allowing us to use larger meshes.

Below, notice how the `mesh_shape` is increased to **30 × 30**. Because CPU computation exploits sparse matrices and
benefits from larger system memory, we can now use higher-resolution pixelizations than were practical with JAX GPU
acceleration.
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
__Fit__

In the example `imaging/features/pixelization/fit.py`, we demonstrated fitting imaging data using a
pixelized source with a rectangular mesh.

Below, we perform a similar fit using the **same pixelization**, but this time accelerated on the **CPU**
using `numba` and sparse operations.
"""
mesh = al.mesh.RectangularMagnification(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Model__

We now perform a full model-fit using the `w_tilde` formalism on the CPU.

There are two key differences from the earlier JAX-based pixelization examples:

- **JAX is disabled**  
  The `AnalysisImaging` class is created with `use_jax=False`, preventing JAX compilation and ensuring
  that all computations run on the CPU.

- **CPU parallelization**  
  The non-linear search is given a `number_of_cores` parameter, which parallelizes likelihood evaluations
  using Python's `multiprocessing`.  
  In practice, this provides a speed-up of roughly half the number of CPU cores used  
  (e.g., 4 cores → ~2× speed-up, 8 cores → ~4× speed-up).
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=Path("features"),
    name="cpu_fast_modeling",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=2,  # CPU specific code
)

analysis = al.AnalysisImaging(
    dataset=dataset_w_tilde, preloads=preloads, use_jax=False  # CPU specific code
)

result = search.fit(model=model, analysis=analysis)


"""
__SLaM Pipeline__

The example `guides/modeling//slam_start_here.ipynb` introduces the SLaM (Source, Light and Mass) pipelines for
automated lens modeling of large samples of strong lenses.

We finish this example by showing how to run the SLaM pipelines using CPU acceleration with `w_tilde`, similar to the
model-fit above.

Note that the first pipeline, SOURCE LP, uses JAX acceleration as in previous examples and therefore does not 
pass `use_jax=False` or a `number_of_cores` parameter.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import slam_pipeline

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam_cpu_fast",
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
analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

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
__CPU Fast SLaM Pipelines__

The SLaM pipeline is mostly identical to other examples, but via the `SettingsSearch` it
uses a `number_of_cores` parameter to parallelize the likelihood evaluations on the CPU
and disables JAX compilation for each `AnalysisImaging` object.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam_cpu_fast",
    unique_tag=dataset_name,
    info=None,
    session=None,
    number_of_cores=2,
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
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
    preloads=preloads,
    use_jax=False,  # CPU specific code
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=af.Model(al.mesh.RectangularMagnification, shape=mesh_shape),
    regularization_init=al.reg.AdaptiveBrightness,
)

"""
__SOURCE PIX PIPELINE 2__

The SOURCE PIX PIPELINE 2 is identical to the `slam_start_here.ipynb` example.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    use_jax=False,  # CPU specific code
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=af.Model(al.mesh.RectangularSource, shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE is setup identically to the `slam_start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    use_jax=False,  # CPU specific code
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

The MASS TOTAL PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    preloads=preloads,
    use_jax=False,  # CPU specific code
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
__Wrap Up__

This example has demonstrated how to perform fast pixelized source modeling on a CPU without JAX, by combining
`numba` and Python `multiprocessing`.
"""

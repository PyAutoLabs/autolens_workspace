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

__Contents__

**Sparse Operators:** Pixelized source modeling requires dense linear algebra operations.
**Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
**Fit:** Fit the lens model to the dataset.
**Model:** Compose the lens model fitted to the data.
**SLaM Pipeline:** The example `guides/modeling//slam_start_here.ipynb` introduces the SLaM (Source, Light and Mass).
**SLaM Pipeline Functions:** Overview of slam pipeline functions for this example.
**CPU Fast SLaM Pipelines:** The SLaM pipeline is mostly identical to other examples, but via the `SettingsSearch` it uses a.
**Wrap Up:** Summary of the script and next steps.

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

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

if not (dataset_path / "positions.json").exists():
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/imaging/data_preparation/examples/optional/positions.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
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

positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

"""
__Sparse Operators__

Pixelized source modeling requires dense linear algebra operations. These calculations can be greatly accelerated
using an alternative mathematical approach called the **sparse operator formalism**.

You do not need to understand the full details of the method, but the key point is:

- It exploits the **sparsity** of the matrices used in pixelized source reconstruction, reducing memory usage.
- This leads to a **significant speed-up on CPUs**.
- The current implementation does **not support JAX**, and therefore does not benefit from GPU acceleration.

To enable this feature, we call `apply_sparse_operator()` on the `Imaging` dataset. This computes and stores operator
matrices, which are then reused in all subsequent pixelized source fits.

- Computing the operator matrices takes anywhere from a few seconds to a few minutes, depending on the dataset size.
- After it is computed once, every model-fit using pixelization becomes substantially faster.
"""
dataset = dataset.apply_sparse_operator_cpu()

"""
__Mesh Shape__

As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Fit__

In the example `imaging/features/pixelization/fit.py`, we demonstrated fitting imaging data using a
pixelized source with a rectangular mesh.

Below, we perform a similar fit using the **same pixelization**, but this time accelerated on the **CPU**
using `numba` and sparse operations.
"""
mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
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
)

aplt.subplot_fit_imaging(fit=fit)

"""
__Model__

We now perform a full model-fit using the sparse operator formalism on the CPU.

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
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
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
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    use_jax=False,  # CPU specific code
)

result = search.fit(model=model, analysis=analysis)


"""
__SLaM Pipeline__

The example `guides/modeling//slam_start_here.ipynb` introduces the SLaM (Source, Light and Mass) pipelines for
automated lens modeling of large samples of strong lenses.

We finish this example by showing how to run the SLaM pipelines using CPU acceleration with sparse operators, similar 
to the model-fit above.

Note that the first pipeline, SOURCE LP, uses JAX acceleration as in previous examples and therefore does not 
pass `use_jax=False` or a `number_of_cores` parameter.
"""
"""
__SLaM Pipeline Functions__
"""


def source_lp(
    settings_search,
    analysis,
    lens_bulge,
    source_bulge,
    redshift_lens,
    redshift_source,
    mass_centre=(0.0, 0.0),
    n_batch=50,
):
    """
    SOURCE LP PIPELINE: fits an initial lens model using a parametric source to establish a robust
    lens light, mass and source model before pixelized source fitting.
    """
    mass = af.Model(al.mp.Isothermal)
    mass.centre = mass_centre

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=mass,
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


def source_pix_1(
    settings_search,
    analysis,
    source_lp_result,
    mesh_shape,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 1: initializes a pixelized source model with mass priors from SOURCE LP PIPELINE,
    run on CPU using sparse operators and multiprocessing.
    """
    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_lp_result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Adapt,
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


def source_pix_2(
    settings_search,
    analysis,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    n_batch=20,
):
    """
    SOURCE PIX PIPELINE 2: fits an improved pixelized source using adapt images from SOURCE PIX PIPELINE 1,
    with fixed lens mass, run on CPU using sparse operators and multiprocessing.
    """
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
                    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
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


def light_lp(
    settings_search,
    analysis,
    source_result_for_lens,
    source_result_for_source,
    lens_bulge,
    n_batch=20,
):
    """
    LIGHT LP PIPELINE: fits the lens galaxy light with mass and source fixed from SOURCE PIX PIPELINE,
    run on CPU using sparse operators and multiprocessing.
    """
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


def mass_total(
    settings_search,
    analysis,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    n_batch=20,
):
    """
    MASS TOTAL PIPELINE: fits a PowerLaw total mass model with priors from SOURCE PIX PIPELINE and
    lens light fixed from LIGHT LP PIPELINE, run on CPU using sparse operators and multiprocessing.
    """
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=light_result.instance.galaxies.lens.bulge,
                disk=light_result.instance.galaxies.lens.disk,
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
__SLaM Pipeline__
"""

redshift_lens = 0.5
redshift_source = 1.0

settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam_cpu_fast",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

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

analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

source_lp_result = source_lp(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    source_bulge=source_bulge,
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
    use_jax=False,  # CPU specific code
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_shape=mesh_shape,
)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    use_jax=False,  # CPU specific code
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    use_jax=False,  # CPU specific code
)

light_result = light_lp(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    use_jax=False,  # CPU specific code
)

mass_result = mass_total(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)


"""
__Wrap Up__

This example has demonstrated how to perform fast pixelized source modeling on a CPU without JAX, by combining
`numba` and Python `multiprocessing`.
"""

"""
Pixelization: CPU Fast Modeling
===============================

Read the [feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb) for the shared concepts and a guide to the examples.

This example shows how to **override the automatic compute setup**, by fitting a pixelized source with a setup you
choose yourself:

- `numba` for the likelihood, instead of JAX;
- the sparse operator formalism for the inversion's linear algebra; and
- Python `multiprocessing` to run likelihoods on several CPU cores at once.

You usually do not need to do this. Passing `setup="auto"` to the analysis and `number_of_cores="auto"` to the
search already picks this setup, `numba_cpu_sparse_fp64` with one worker per core, for a pixelized source on a CPU.
On an A100 GPU it picks `jax_gpu_dense_fp64` instead, and for MGE or parametric sources it picks JAX on both CPU
and GPU. The `guides/advanced/auto_setup.py` guide explains the automatic setup and lists the setup it picks for
every dataset.

Override it when you want the same setup on every machine, for example to compare run times, or when the automatic
choice is not the one you want.

__Contents__

- **Compute Setup:** Choose the numba sparse setup, which attaches the sparse operator to the dataset.
- **Mesh Shape:** As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before.
- **Fit:** Fit the lens model to the dataset.
- **Model:** Compose the lens model fitted to the data.
- **SLaM Pipeline:** The example `guides/modeling//slam_start_here.ipynb` introduces the SLaM (Source, Light and Mass).
- **SLaM Pipeline Functions:** Overview of slam pipeline functions for this example.
- **CPU Fast SLaM Pipelines:** The pixelized stages of the SLaM pipeline use the overridden setup and `number_of_cores`.
- **Wrap Up:** Summary of the script and next steps.
"""

try:
    import numba
except ModuleNotFoundError:
    input("##################\n" "##### NUMBA ######\n" "##################\n\n" """
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
        """)

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

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
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/imaging/features/no_lens_light/simulator.py"
)

# Intentional raw guard: positions.json is a file (should_simulate rmtree's a directory); the
# should_simulate guard above already deletes the whole dataset folder under PYAUTO_SMALL_DATASETS,
# so this file-level check re-fires whenever the dataset is regenerated.
if not (dataset_path / "positions.json").exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/imaging/data_preparation/examples/optional/positions.py",
        ],
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
    sub_size_list=[4, 2, 2],
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
__Compute Setup__

Pixelized source modeling uses linear algebra on large matrices that are mostly zeros. The **sparse operator
formalism** exploits this sparsity, which reduces memory use and speeds up every pixelized fit.

The sparse operator has two implementations: a numba one (`apply_sparse_operator_cpu()`) and a JAX one
(`apply_sparse_operator()`). Profiling found numba with the sparse operator, one likelihood per core, the fastest
setup on a many-core CPU, and JAX with the dense linear algebra the fastest on an A100 GPU. That is why
`setup="auto"` picks them.

A setup is named by a setup id, `<backend>_<device>_<operator>_<precision>`. Applying `numba_cpu_sparse_fp64` to the
dataset builds the numba sparse operator once, which takes a few seconds to a few minutes depending on the dataset
size, and every fit below reuses it. The analyses below are given the same setup; they see the operator is already
attached and do not rebuild it.
"""
setup = "numba_cpu_sparse_fp64"

dataset = dataset.apply_setup(setup=setup)

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
mesh = al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

# The external shear is a property of the system, not of a galaxy, so it is held in a
# `MassField` given to the `Tracer` via `fields=` (see `imaging/modeling.py`).

field = al.MassField(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens, source], fields=[field])

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
)

aplt.subplot_fit_imaging(fit=fit)

"""
__Model__

We now perform a full model-fit with the setup chosen by hand. Two inputs set it:

- **The setup:** `AnalysisImaging` is given `setup="numba_cpu_sparse_fp64"` instead of `setup="auto"`.

- **The number of cores:** the search is given `number_of_cores=2` instead of `number_of_cores="auto"` (which
  would use every detected core). Each core runs one likelihood at a time, through Python's `multiprocessing`.

When the fit starts, it prints `Compute setup (set explicitly by setup=...):` followed by the setup it used and an
estimate of the run time.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

# External Shear:

field = af.Model(al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear))

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

search = af.Nautilus(
    path_prefix=Path("features"),
    name="cpu_fast_modeling",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=2,  # Override: "auto" uses every detected core for a numba setup
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    setup=setup,  # Override: setup="auto" picks this setup on a CPU
)

result = search.fit(model=model, analysis=analysis)


"""
__SLaM Pipeline__

The example `guides/modeling//slam_start_here.ipynb` introduces the SLaM (Source, Light and Mass) pipelines for
automated lens modeling of large samples of strong lenses.

We finish this example by running the SLaM pipelines with the setup overridden for the pixelized stages, as in
the model-fit above.

The first pipeline, SOURCE LP, fits an MGE source, which has no pixelization. Its analysis is given `setup="auto"`,
which picks JAX for it, and its search does not pass a `number_of_cores` parameter.

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
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
            ),
        ),
        fields=af.Model(
            al.MassField,
            redshift=redshift_lens,
            shear=af.Model(al.mp.ExternalShear),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
        live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
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
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(
                        al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape
                    ),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        fields=source_lp_result.model.fields,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
        live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
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
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(
                        al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape
                    ),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        fields=source_pix_result_1.instance.fields,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
        live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
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
            ),
            source=source,
        ),
        fields=source_result_for_lens.instance.fields,
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
        live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
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
            ),
            source=source,
        ),
        fields=source_result_for_lens.model.fields,
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
        live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
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
    sigma_min=dataset.pixel_scales[0] / 10.0,
)

# Source Light

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

analysis = al.AnalysisImaging(dataset=dataset, setup="auto")

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

The rest of the SLaM pipeline is the same as in other examples, except that the `SettingsSearch`
passes `number_of_cores` to every search and each `AnalysisImaging` is given `setup=setup`.
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

"""
__Adapt Image S/N Cap__

The source adapt image is capped at a signal-to-noise of 3.0 before it is used by the adaptive
image-mesh and the adaptive regularization. Without the cap the brightest peak dominates the
weights (they scale as a power of the adapt image), so fainter multiply-imaged features get too
few source pixels and too little regularization weight. Capping makes every feature above S/N 3.0
count equally. The cap is applied to an explicit copy so the raw S/N image is untouched.
"""
adapt_image_snr_cap = 3.0

source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    setup=setup,  # Override: setup="auto" picks this setup on a CPU
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

# Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
adapt_image_snr_cap = 3.0

source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

"""
__Adaptive Pixelization Over-Sampling__

From SOURCE PIX PIPELINE 2 onwards the pixelization grid is over-sampled adaptively. The source's
signal-to-noise map from the previous pixelized fit (the same map that becomes the adapt image, read before
the S/N 3.0 cap is applied) is thresholded at S/N 3.0: pixels above it, the bright lensed source, use a
sub-size of 4 and every other pixel uses a sub-size of 2. This concentrates the extra over-sampling where the
source is bright and the pixelization gains the most accuracy from it, and keeps the rest of the mask cheap.

The map returned by `galaxy_name_image_dict_via_result_from` is already signal divided by noise, so it is
thresholded directly.

SOURCE PIX PIPELINE 1 keeps the dataset's default uniform sub-size. Its adapt image comes from the parametric
source fit of the SOURCE LP PIPELINE, which does not yet trace the lensed source well enough to steer
over-sampling.

The sparse operator depends only on the noise-map, PSF and mask, not on the over-sampling, so
`apply_over_sampling` keeps the operator and setup already attached to the dataset. Nothing is rebuilt.
"""
signal_to_noise_threshold = 3.0

source_image_raw = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)["('galaxies', 'source')"]

over_sample_size_pixelization = al.Array2D(
    values=np.where(source_image_raw > signal_to_noise_threshold, 4, 2),
    mask=dataset.mask,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=over_sample_size_pixelization
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    setup=setup,  # Override: setup="auto" picks this setup on a CPU
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
    sigma_min=dataset.pixel_scales[0] / 10.0,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    setup=setup,  # Override: setup="auto" picks this setup on a CPU
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
    setup=setup,  # Override: setup="auto" picks this setup on a CPU
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

This example has shown how to override the automatic setup, with `setup="numba_cpu_sparse_fp64"` on the analysis
and `number_of_cores` on the search. `setup="auto"` with `number_of_cores="auto"` picks the same setup on a CPU and
`jax_gpu_dense_fp64` on an A100 GPU, without changing the script. The `guides/advanced/auto_setup.py` guide covers
the automatic setup, its other overrides (`al.Setup` objects and the `PYAUTO_SETUP` environment variable) and
run-time estimates.
"""

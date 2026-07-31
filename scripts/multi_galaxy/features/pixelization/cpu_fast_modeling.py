"""
CPU Fast Modeling: Pixelization (Multi Galaxy)
==============================================

This script fits a multi-galaxy strong lens with a pixelized source **on the CPU, without JAX**, using:

 - `numba` for optimized numerical routines.
 - Python `multiprocessing` to use multiple CPU cores.
 - The sparse operator formalism for the inversion's linear algebra.

Pixelizations rely on sparse linear algebra, which is not currently optimized in JAX. On a machine with many
cores this route can therefore outperform JAX GPU acceleration for a pixelized source.

> This applies to pixelized sources only. For parametric or multi-Gaussian sources, JAX with a GPU is
> substantially faster — which is what the rest of this package uses.

__Contents__

- **Dataset:** Load the multi-galaxy dataset that is fitted.
- **Extra Galaxies Noise Scaling:** Scale the contaminating galaxy's light out of the fit.
- **Mask, Centres & Over Sampling:** Standard set up, over-sampled at every deflector centre.
- **Sparse Operators:** Pre-compute the sparse matrices the CPU inversion uses.
- **Mesh Shape:** The resolution of the source-plane mesh.
- **Fit:** A single CPU fit, before any search.
- **Model:** The model-fit, with JAX disabled and the search parallelized over cores.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__What Changes For Multiple Deflectors__

Each likelihood evaluation has two halves: the deflection field, which is computed per deflector and summed, and
the inversion, whose cost is set by the number of image and source pixels. Adding a second co-dominant deflector
grows the first half and leaves the second unchanged, and it is the second half that the sparse operators below
accelerate.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/features/pixelization/modeling.py` for the JAX version of
this model, and `imaging/features/pixelization/cpu_fast_modeling.py` for the galaxy-scale walkthrough.
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

        [Press Enter to continue]
        """
    )

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset, the same co-dominant pair fitted by `multi_galaxy/modeling.py`.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    ),
    over_sample_size_pixelization=4,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Sparse Operators__

The inversion's matrices are mostly zeros, because each image pixel maps to only a handful of source pixels.
`apply_sparse_operator_cpu` pre-computes the operators that exploit this, so the dense operations are replaced by
sparse ones for every subsequent fit.

Computing them takes anywhere from a few seconds to a few minutes depending on the dataset size. It is done once,
here, and every model-fit below is faster for it.
"""
dataset = dataset.apply_sparse_operator_cpu()

"""
__Mesh Shape__

The shape of the source-plane mesh, fixed rather than fitted.
"""
mesh_shape = (28, 28)

"""
__Fit__

A single fit with the CPU-accelerated pixelization, using the same two deflectors as
`multi_galaxy/features/pixelization/fit.py`. Nothing about the composition changes for the CPU route — the
sparse operators live on the dataset, not on the galaxies.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log Likelihood: {fit.log_likelihood}")

"""
__Model__

The standard multi-galaxy composition — one `lens_i` per deflector with its mass centre fixed, the shear in its
own `shear_galaxy` — with a pixelized source.

Two things differ from the JAX examples in this package:

 - **JAX is disabled:** `AnalysisImaging` is created with `use_jax=False`.
 - **CPU parallelization:** the search is given `number_of_cores`, which parallelizes likelihood evaluations
   with Python's `multiprocessing`.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

# External Shear:

shear_galaxy_model = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source (pixelized):

pix = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pix)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(
        **lens_dict, shear_galaxy=shear_galaxy_model, source=source
    )
)

print(model.info)

search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features" / "pixelization",
    name="cpu_fast_modeling",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=2,  # CPU specific: parallelize likelihood evaluations
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

positions = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "positions.json")
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    use_jax=False,  # CPU specific: disable JAX compilation
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/modeling.py` — the same model fitted with JAX enabled.
 - `multi_galaxy/features/pixelization/delaunay.py` — the Delaunay meshes.
 - `guides/hpc` — running fits on a cluster, where `number_of_cores` matters most.
 - `imaging/features/pixelization/cpu_fast_modeling.py` — the galaxy-scale walkthrough, including its CPU-fast
   SLaM pipeline.
"""

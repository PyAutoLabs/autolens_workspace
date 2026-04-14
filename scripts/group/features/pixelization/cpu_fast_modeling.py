"""
CPU Fast Modeling: Pixelization (Group)
=======================================

This script demonstrates how to achieve fast pixelization performance on a CPU without JAX for group-scale
strong lenses, by combining:

 - `numba` for optimized numerical routines.
 - Python `multiprocessing` to exploit multiple CPU cores.
 - Sparse operator formalism for efficient linear algebra.

For group-scale lenses, the larger 7.5" mask means significantly more image pixels than galaxy-scale
lenses, which increases the size of the matrices used in the pixelized source reconstruction. CPU
optimization is therefore even more important for group lenses, as the matrix operations scale with
the number of image pixels.

On machines with many CPU cores (e.g. HPC clusters with >10 CPUs), this method can outperform JAX GPU
acceleration for pixelized source modeling, because pixelizations rely on sparse linear algebra which
is not currently optimized in JAX.

> Note: This performance advantage applies only to pixelized sources. For parametric sources or
> multi-Gaussian models, JAX (especially with a GPU) is significantly faster.

__Contents__

**Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
**Galaxy Centres:** Load centres for main lens and extra galaxies.
**Sparse Operators:** Pre-compute sparse matrices for CPU-accelerated pixelization.
**Fit:** Fit the group lens with a pixelized source on CPU.
**Model:** Full model-fit with CPU parallelization.

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

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset `simple`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Galaxy Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Sparse Operators__

Pixelized source modeling requires dense linear algebra operations. These calculations can be greatly
accelerated using the sparse operator formalism.

For group-scale lenses, this is especially important because the 7.5" mask contains many more pixels
than a typical galaxy-scale 3.0" mask, making the matrices much larger.

Computing the operator matrices takes anywhere from a few seconds to a few minutes, depending on the
dataset size. After it is computed once, every model-fit using pixelization becomes substantially faster.
"""
dataset = dataset.apply_sparse_operator_cpu()

"""
__Mesh Shape__
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Fit__

We first demonstrate a single fit using the CPU-accelerated pixelization.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log Likelihood: {fit.log_likelihood}")

"""
__Model__

We now perform a full model-fit using the sparse operator formalism on the CPU.

There are two key differences from JAX-based pixelization examples:

 - **JAX is disabled**: The `AnalysisImaging` class is created with `use_jax=False`.
 - **CPU parallelization**: The non-linear search uses `number_of_cores` to parallelize likelihood
   evaluations using Python's `multiprocessing`.

For group-scale lenses, CPU parallelization is especially beneficial because each likelihood evaluation
is more expensive (due to the larger mask and more galaxies).
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_dict[f"lens_{i}"] = lens

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source: Rectangular pixelization with constant regularization.

pix = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pix)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    extra_galaxies=extra_galaxies,
)

search = af.Nautilus(
    path_prefix=Path("group") / "features" / "pixelization",
    name="cpu_fast_modeling",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=2,  # CPU specific: parallelize likelihood evaluations
)

analysis = al.AnalysisImaging(
    dataset=dataset,
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

This script demonstrated CPU-optimized pixelization modeling for group-scale lenses.

Key points for group lenses:
 - The larger 7.5" mask means more image pixels, making sparse operators especially valuable.
 - CPU parallelization via `number_of_cores` provides significant speedup for the expensive likelihood
   evaluations of group-scale fits.
 - On HPC clusters with many cores, this approach can outperform JAX GPU acceleration for pixelized sources.
 - For parametric sources (MGE, Sersic), JAX with GPU remains faster.
"""

"""
CPU Fast Modeling: Pixelization (Group)
=======================================

Read the [feature introduction](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/group/features/pixelization/start_here.ipynb) for the shared concepts and a guide to the examples.

This script shows how to override the automatic compute setup for a group-scale strong lens with a pixelized
source, by choosing the setup yourself:

 - `numba` for the likelihood, instead of JAX.
 - The sparse operator formalism for the inversion's linear algebra.
 - Python `multiprocessing` to run likelihoods on several CPU cores at once.

You usually do not need to do this. Passing `setup="auto"` to the analysis and `number_of_cores="auto"` to the
search already picks this setup, `numba_cpu_sparse_fp64` with one worker per core, for a pixelized source on a CPU.
On an A100 GPU it picks `jax_gpu_dense_fp64` instead, and for MGE or parametric sources it picks JAX on both. The
`guides/advanced/auto_setup.py` guide explains the automatic setup and lists the setup it picks for every dataset.

Override it when you want the same setup on every machine, for example to compare run times, or when the automatic
choice is not the one you want.

Group-scale lenses use a larger 7.5" mask, so there are many more image pixels than for a galaxy-scale lens and the
matrices of the pixelized source reconstruction are larger. Each likelihood evaluation therefore costs more, and the
choice of setup matters more.

__Contents__

- **Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
- **Galaxy Centres:** Load centres for main lens and extra galaxies.
- **Compute Setup:** Choose the numba sparse setup, which attaches the sparse operator to the dataset.
- **Fit:** Fit the group lens with a pixelized source on CPU.
- **Model:** Full model-fit, with the setup passed to the analysis and the search parallelized over cores.
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

        [Press Enter to continue]
        """)

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

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
al.util.dataset.simulate_if_needed(dataset_path, simulator="scripts/group/simulator.py")

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
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Compute Setup__

The sparse operator formalism speeds up the linear algebra of a pixelized source reconstruction. For group-scale
lenses this matters more, because the 7.5" mask contains many more pixels than a typical galaxy-scale 3.0" mask.

A setup is named by a setup id, `<backend>_<device>_<operator>_<precision>`. Applying `numba_cpu_sparse_fp64` to the
dataset builds the numba sparse operator once, which takes a few seconds to a few minutes depending on the dataset
size, and every fit below reuses it. The analysis below is given the same setup; it sees the operator is already
attached and does not rebuild it.
"""
setup = "numba_cpu_sparse_fp64"

dataset = dataset.apply_setup(setup=setup)

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
    mass=al.mp.dPIEMassSph(
        centre=(3.5, 2.5),
        sigma=212.0,
        r_core=0.0,
        r_cut=10.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(-4.4, -5.0),
        sigma=239.0,
        r_core=0.0,
        r_cut=10.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

mesh = al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape)
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

We now perform a full model-fit with the setup chosen by hand. Two inputs set it:

 - **The setup:** `AnalysisImaging` is given `setup="numba_cpu_sparse_fp64"` instead of `setup="auto"`.
 - **The number of cores:** the search is given `number_of_cores=2` instead of `number_of_cores="auto"` (which
   would use every detected core). Each core runs one likelihood at a time, through Python's `multiprocessing`.

When the fit starts, it prints `Compute setup (set explicitly by setup=...):` followed by the setup it used.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

    lens_dict[f"lens_{i}"] = lens

# External Shear (an `al.MassField`, in the model's `fields` slot below):

field = af.Model(al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear))

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_fixed=centre,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
    mass.r_core = 0.0  # vanishing core — fixed; the dPIE is analytic at r_core = 0
    mass.r_cut = 10.0  # truncation fixed at a fiducial radius
    mass.redshift_object = 0.5
    mass.redshift_source = 1.0
    mass.H0 = 67.66  # pinned: model constants, not parameters to sample
    mass.Om0 = 0.30966

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source: Rectangular pixelization with constant regularization.

pix = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pix)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    fields=field,
    extra_galaxies=extra_galaxies,
)

search = af.Nautilus(
    path_prefix=Path("group") / "features" / "pixelization",
    name="cpu_fast_modeling",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=2,  # Override: "auto" uses every detected core for a numba setup
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    setup=setup,  # Override: setup="auto" picks this setup on a CPU
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

This script fitted a group-scale lens with the compute setup chosen by hand.

 - `setup="numba_cpu_sparse_fp64"` and `number_of_cores` override the automatic setup.
 - `setup="auto"` with `number_of_cores="auto"` picks the same setup on a CPU, and `jax_gpu_dense_fp64` on an A100
   GPU, without changing the script.
 - `guides/advanced/auto_setup.py` explains the automatic setup, and `guides/hpc` covers running fits on a cluster.
"""

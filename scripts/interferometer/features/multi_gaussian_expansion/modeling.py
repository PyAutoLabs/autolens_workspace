"""
Modeling Features: Multi Gaussian Expansion (Interferometer)
============================================================

A multi-Gaussian expansion (MGE) decomposes a galaxy's light into ~15-100 Gaussians, where the `intensity` of
every Gaussian is solved for via linear algebra using a process called an "inversion" (see the
`linear_light_profiles` feature for a full description of this).

This script performs lens modeling of an `Interferometer` dataset using an MGE source bulge consisting of
30 Gaussians arranged in two groups of 15. Each group has its own elliptical components, so the source's
light is decomposed into two distinct elliptical components — which could be viewed as a bulge and a disk
of the source galaxy.

**Role swap vs the imaging MGE example:** the imaging script fits the *lens* galaxy's complex morphology
with an MGE. For interferometer data the lens light is omitted (no detection in mm/sub-mm), so the MGE is
instead applied to the *source* galaxy. This is the standard convention for interferometer modeling.

MGE fits to interferometer data were previously impractical because every likelihood evaluation has to
Fourier-transform each Gaussian basis component into the uv-plane. With `nufftax`
(https://github.com/GragasLab/nufftax) — a JAX-native NUFFT — the full basis is transformed inside the same
jit/vmap pipeline as the rest of the model, amortising the per-iteration NUFFT cost on the GPU. MGE source
fits are now routine even at ALMA-class visibility counts.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of an MGE source for interferometer data.
- **NUFFT (nufftax):** Why MGE-on-visibilities is now practical thanks to nufftax.
- **Positive Only Solver:** Ensuring positive-only solutions for linear light profile intensities.
- **Model:** Compose the lens model — `Isothermal` + `ExternalShear` lens mass and a 30-Gaussian MGE
  source bulge. Lens light omitted (interferometer convention).
- **Mask:** Define the `real_space_mask` which sets the grid the strong lens is evaluated on.
- **Dataset:** Load the strong lens `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Over Sampling:** Interferometer modeling does not use over-sampling.
- **Search:** Configure the non-linear search (Nautilus).
- **Analysis:** Create the `AnalysisInterferometer` object.
- **VRAM:** Memory budget for a multi-component linear basis on GPU.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Wrap Up:** Summary of the script and next steps.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals when fitting the source
galaxy of a strong lens — they fail to capture irregular and asymmetric morphology (e.g. isophotal twists,
multi-component source emission). An MGE fully captures these features and can therefore much better
represent the emission of complex source galaxies.

The MGE model is composed such that the `intensity` parameters and the `sigma` parameters controlling the
Gaussian sizes are *not* sampled by Nautilus. Centres and elliptical components are shared across each
group of Gaussians. This removes the most significant degeneracies in parameter space, making the model
much more reliable and efficient to fit than a free-intensity Sersic.

Therefore, not only does an MGE source fit more complex source morphologies, it does so using fewer
non-linear parameters in a much simpler non-linear parameter space which has far less significant parameter
degeneracies.

__Disadvantages__

To fit an MGE model, the light of the ~15-100 Gaussians must be evaluated and NUFFT'd to the uv-plane per
likelihood evaluation. This is slower than evaluating a single `SersicCore`. With `nufftax` the per-NUFFT
cost is small enough that the total slow-down per likelihood evaluation is typically 2-5x for a 30-Gaussian
MGE — paid back in fewer iterations because the parameter space is simpler.

The MGE source can also be less intuitive to interpret physically than a Sersic. The Sersic `effective_radius`
and `sersic_index` map directly to galaxy size and concentration; an MGE's solved-for Gaussian intensities
require an extra processing step to compute equivalent physical quantities.

__NUFFT (nufftax)__

The image-to-visibilities Fourier transform is performed by a Non-Uniform Fast Fourier Transform (NUFFT),
exposed in **PyAutoLens** as `TransformerNUFFT`. The default backend is `nufftax`, a pure-JAX NUFFT that
jit-compiles and vmap-batches like the rest of the library:

  https://github.com/GragasLab/nufftax

Because `nufftax` is JAX-native, NUFFT-ing every Gaussian in the MGE basis happens inside the same compiled
likelihood that does the inversion, mass model ray-tracing, and chi-squared sum. There is no host round-trip
between NUFFT calls, so a model with N Gaussians costs only N forward-NUFFTs per iteration on the GPU —
fast enough that MGE-on-visibilities is now routinely practical.

If `nufftax` is not installed, install it via `pip install nufftax`.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algebra solver which allows for positive and
negative values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a
galaxy's light, which is clearly unphysical. For an MGE, this produces a positive-negative "ringing", where
the Gaussians alternate between large positive and negative values. This is clearly undesirable and
unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it
is as fast as positive-negative solvers. This ensures that all Gaussian intensities are positive and
therefore physical.

__Model__

This script fits an `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data). This is the standard
   convention for interferometer modeling, as the lens galaxy's optical/IR emission is typically below the
   detection threshold of mm/sub-mm interferometers.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's bulge is a multi-Gaussian expansion of 30 linear `Gaussian` profiles, arranged in
   two groups of 15 (each group shares a centre and ell_comps).

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.

__Imaging Equivalent__

For the CCD-imaging version of this script (MGE on the lens galaxy, not the source), see
`autolens_workspace/*/imaging/features/multi_gaussian_expansion/modeling.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Mask__

We define the `real_space_mask` which defines the grid the image of the strong lens is evaluated on.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files, which we will fit with
the lens model.

This includes the method used to Fourier transform the real-space image of the strong lens to the uv-plane
and compare directly to the visibilities. We use `TransformerNUFFT`, the JAX-native Non-Uniform Fast Fourier
Transform backed by `nufftax`, which is required for fast MGE modeling and scales efficiently from a few
hundred visibilities to tens of millions.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over
sampling is used, which evaluates light profiles on a higher resolution grid than the image data to ensure
the calculation is accurate.

Interferometer data does not observe galaxies in a way where over sampling is necessary, therefore all
interferometer calculations are performed without over sampling.

__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's bulge is 10 linear `Gaussian` profiles [6 parameters total].
   - The centres and elliptical components of the Gaussians are linked together in two groups of 5.
   - The `sigma` size of the Gaussians increases in log10 increments from 0.01 to the mask radius.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.

The MGE comprises 2 groups of 5 Gaussians. Each group has its own elliptical components, so the source's
light is decomposed into two distinct elliptical components which could be viewed as a bulge and a disk of
the source.

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
total_gaussians = 5
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning 0.0001 to the mask radius, 3.5".
log10_sigma_list = np.linspace(-4, np.log10(mask_radius), total_gaussians)

# By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
centre_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    # A list of Gaussian model components whose parameters are customized below.

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    # Iterate over every Gaussian and customize its parameters.

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
        gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
        gaussian.ell_comps = gaussian_list[
            0
        ].ell_comps  # All Gaussians in this group share ell_comps.
        gaussian.sigma = (
            10 ** log10_sigma_list[i]
        )  # All Gaussian sigmas are fixed to values above.

    bulge_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.
source_bulge = af.Model(al.lp_basis.Basis, profile_list=bulge_gaussian_list)

# Lens:
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen
refer to `start_here.ipynb` for a description of how to fix this).

This shows every single Gaussian light profile in the model, which is a lot of parameters! However, the
vast majority of these parameters are fixed to the values we set above, so the model actually has far fewer
free parameters than it looks!
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start_here.py` for a
full description).

Owing to the simplicity of fitting an MGE (no intensity or sigma free parameters), we use fewer live points
than the `interferometer/modeling.py` example: 75 live points speeds up convergence.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features",
    name="mge",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=20,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisInterferometer` object defining how Nautilus fits the model to the data.
"""
analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

"""
__VRAM__

The `interferometer/modeling.py` example explains how VRAM is used during GPU-based fitting and how to
print the estimated VRAM required by a model.

For each linear `Gaussian` profile, extra VRAM is used to store its NUFFT'd mapping matrix column. For
around 30 linear Gaussians this typically requires a modest amount of VRAM (e.g. 10-50 MB per batched
likelihood). Models that use hundreds of Gaussians, especially in combination with a large batch size, may
therefore exceed GBs of VRAM and require you to adjust the batch size to fit within your GPU's VRAM.

Without the sparse operator, VRAM on interferometer datasets is driven primarily by the visibility count and
the real-space mask size, not the number of Gaussians in the MGE basis. With `apply_sparse_operator()` applied
(which now supports linear light profiles as well as pixelizations) it depends on the real-space mask alone.

__Run Time__

The likelihood evaluation time for an MGE is slower than a single linear `SersicCore` source, because the
image of every Gaussian must be evaluated and NUFFT'd to the uv-plane. With `nufftax`, the per-NUFFT cost
is small enough that the total slow-down per likelihood is typically 2-5x for a 30-Gaussian MGE compared
to a one-component source — paid back in fewer iterations because the parameter space is simpler.

Because the MGE has no free `intensity` or `sigma` parameters and shares centres/ell_comps across groups,
Nautilus converges significantly faster than for a free-intensity Sersic source. We also use fewer live
points (75 vs the 100 used in `interferometer/modeling.py`), further speeding up the model-fit.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the
output folder for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format (if
this does not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix
this):

This confirms there are many `Gaussian`s in the source light model and it lists their inferred parameters.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

In particular, checkout the results example `linear.py` which details how to extract all information about
linear light profiles from a fit.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

To further showcase MGE source modeling, see the SLaM pipeline at
`autolens_workspace/scripts/interferometer/features/multi_gaussian_expansion/slam.py`, which uses the
`al.model_util.mge_model_from` helper to compose an MGE source in the SOURCE LP stage before chaining
to a pixelized source reconstruction.
"""

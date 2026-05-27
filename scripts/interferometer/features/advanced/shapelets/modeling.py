"""
Modeling Features: Shapelets (Interferometer)
==============================================

A shapelet is a basis function appropriate for capturing the exponential / disk-like features of a galaxy.
It has been employed in many strong lensing studies to model the light of the lensed source galaxy, because
it can represent features of disky star-forming galaxies that a single Sersic function cannot.

- https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T
- https://iopscience.iop.org/article/10.1088/0004-637X/813/2/102

Shapelets are described in full in:

  https://arxiv.org/abs/astro-ph/0105178

This script performs lens modeling of an `Interferometer` dataset using a polar shapelet basis for the
source galaxy. The `intensity` of every shapelet is solved for via linear algebra (see the
`linear_light_profiles` feature for a full description of this).

Shapelet fits to interferometer data were previously impractical because every likelihood evaluation has
to Fourier-transform each shapelet basis component into the uv-plane, and prior NUFFT backends were not
JAX-friendly. With `nufftax` (https://github.com/GragasLab/nufftax) — a JAX-native NUFFT — the full
shapelet basis is transformed inside the same jit/vmap pipeline as the rest of the model, amortising the
per-iteration NUFFT cost on the GPU. Shapelet source fits are now routine even at ALMA-class visibility
counts.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of a shapelet source for interferometer data.
- **NUFFT (nufftax):** Why shapelets-on-visibilities is now practical thanks to nufftax.
- **Positive Negative Solver:** Why shapelets require a positive-negative solver (unlike MGE or linear
  Sersic).
- **Model:** Compose the lens model — `Isothermal` + `ExternalShear` mass and a polar shapelet source
  bulge. Lens light omitted (interferometer convention).
- **Mask:** Define the `real_space_mask` which sets the grid the strong lens is evaluated on.
- **Dataset:** Load the strong lens `Interferometer` dataset using `TransformerNUFFT` (backed by `nufftax`).
- **Over Sampling:** Interferometer modeling does not use over-sampling.
- **Search:** Configure the non-linear search (Nautilus).
- **Analysis:** Create the `AnalysisInterferometer` object with the positive-negative solver enabled.
- **VRAM:** Memory budget for a multi-component shapelet basis on GPU.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Wrap Up:** Summary of the script and next steps.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals because they fail to
capture irregular and asymmetric morphology of source galaxies (e.g. disky star formation, isophotal
twists). Shapelets capture some of these features and can therefore better represent complex source
galaxies.

The shapelet model can be composed in a way that has fewer non-linear parameters than an elliptical
Sersic. In this example, ~10 shapelets which represent the source's `bulge` are composed in a model
corresponding to just N=3 non-linear parameters (centre + shared beta). A linear Sersic source would have
N=6.

__Disadvantages__

- There are many types of galaxy structure which shapelets may struggle to represent, such as a bar or
  asymmetric knots of star formation. Shapelets also rely on the galaxy having a distinct centre over
  which the basis can be centred, which is not the case if the galaxy is a multi-component merging system
  or has bright companion galaxies.

- The linear algebra used to solve for the `intensity` of each shapelet has to allow for negative values
  of intensity. Negative surface brightnesses are unphysical, and are often inferred in a shapelet
  decomposition — for example if the true galaxy has structure that cannot be captured by the shapelet
  basis. Other approaches (MGE, pixelization) can force positive-only intensities on the solution.

- Computationally slower than a single linear `SersicCore` because each shapelet must be NUFFT'd to the
  uv-plane per likelihood. With `nufftax` the per-NUFFT cost is small enough that this is no longer a
  blocker; for many science cases an MGE source is still faster and gives higher quality results.

__NUFFT (nufftax)__

The image-to-visibilities Fourier transform is performed by a Non-Uniform Fast Fourier Transform (NUFFT),
exposed in **PyAutoLens** as `TransformerNUFFT`. The default backend is `nufftax`, a pure-JAX NUFFT:

  https://github.com/GragasLab/nufftax

Because `nufftax` is JAX-native, NUFFT-ing every shapelet basis image happens inside the same compiled
likelihood that does the inversion, mass model ray-tracing, and chi-squared sum. There is no host
round-trip between NUFFT calls, so a model with N shapelets costs only N forward-NUFFTs per iteration on
the GPU — fast enough that shapelet-on-visibilities is now routinely practical.

If `nufftax` is not installed, install it via `pip install nufftax`.

__Positive Negative Solver__

In other examples which use linear algebra to fit the data — linear light profiles, the Multi-Gaussian
Expansion (MGE), and pixelized source reconstructions on CCD imaging — we use a positive-only solver,
which forces all solved-for intensities to be positive. This is physical and sensible because the surface
brightnesses of a galaxy cannot be negative.

Shapelets **cannot** be solved with a positive-only solver. Their ability to decompose the light of a
galaxy relies on being able to use negative intensities — shapelets are not physically motivated light
profiles but a mathematical basis that can represent any light profile, including via cancellations
between positive and negative basis-function amplitudes.

This means shapelet fits may include negative flux in the reconstructed source galaxy, which is
unphysical, and is a known disadvantage of using shapelets. The `Settings` object passed to the analysis
below uses `use_positive_only_solver=False` to allow for negative intensities.

For pixelized source reconstructions on interferometer data this same setting is used for a different
reason: negative visibility-plane noise can pull individual pixels negative without anything being wrong
physically.

__Model__

This script fits an `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data). Interferometer
   convention.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's bulge is a superposition of polar `ShapeletPolar` profiles, with all shapelets
   sharing a centre, elliptical components, and a single `beta` size scale.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.

__Imaging Equivalent__

For the CCD-imaging version of this script, see
`autolens_workspace/*/imaging/features/advanced/shapelets/modeling.py`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

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
and compare directly to the visibilities. We use `TransformerNUFFT`, the JAX-native NUFFT backed by
`nufftax`, which is required for fast shapelet modeling and scales efficiently from a few hundred
visibilities to tens of millions.
"""
dataset_name = "simple"
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

 - The source galaxy's bulge is a superposition of linear `ShapeletPolar` profiles [3 parameters total].
   - All shapelets share a centre, elliptical components, and a single `beta` size scale.
   - The shapelet (n, m) quantum numbers are assigned procedurally.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source: polar shapelet basis with shared centre/ell_comps/beta.
total_n = 10
total_m = sum(range(2, total_n + 1)) + 1

shapelets_bulge_list = af.Collection(
    af.Model(al.lp_linear.ShapeletPolar) for _ in range(total_n + total_m + 1)
)

n_count = 1
m_count = -1

for i, shapelet in enumerate(shapelets_bulge_list):
    if i == 0:
        shapelet.n = 0
        shapelet.m = 0
    else:
        shapelet.n = n_count
        shapelet.m = m_count

        m_count += 2

        if m_count > n_count:
            n_count += 1
            m_count = -n_count

    shapelet.centre = shapelets_bulge_list[0].centre
    shapelet.ell_comps = shapelets_bulge_list[0].ell_comps
    shapelet.beta = shapelets_bulge_list[0].beta

source_bulge = af.Model(al.lp_basis.Basis, profile_list=shapelets_bulge_list)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.

This confirms the source galaxy is made of many `ShapeletPolar` profiles whose centres, elliptical
components, and beta are all shared.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start_here.py` for a
full description).
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features" / "advanced",
    name="shapelets",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisInterferometer` object defining how Nautilus fits the model to the data.

Note `use_positive_only_solver=False` is set on the `Settings` — shapelets require the positive-negative
solver, as discussed above.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    settings=al.Settings(use_positive_only_solver=False),
    use_jax=True,
)

"""
__VRAM__

The `interferometer/modeling.py` example explains how VRAM is used during GPU-based fitting and how to
print the estimated VRAM required by a model.

For each linear shapelet, extra VRAM is used to store its NUFFT'd mapping matrix column. For around 30
shapelets this typically requires a modest amount of VRAM (e.g. 10-50 MB per batched likelihood). Models
that use hundreds of shapelets, especially in combination with a large batch size, may therefore exceed
GBs of VRAM and require you to adjust the batch size to fit within your GPU's VRAM.

VRAM on interferometer datasets is driven primarily by the visibility count and the real-space mask size,
not the number of shapelets in the basis.

__Run Time__

The likelihood evaluation time for a shapelet basis is slower than a single linear `SersicCore` source,
because the image of every shapelet must be evaluated and NUFFT'd to the uv-plane. With `nufftax`, the
per-NUFFT cost is small enough that the total slow-down per likelihood is typically 2-5x for a 30-shapelet
basis compared to a one-component source — paid back in fewer iterations because the parameter space is
simpler (only N=3 free non-linear parameters for the source).

Because the shapelet basis has no free `intensity` or `beta`-per-shapelet parameters (only the shared
beta) and the source intensities are solved by the inversion, Nautilus converges significantly faster
than for a free-intensity Sersic source.

If shapelets are too slow for your science case, consider the MGE source feature
(`interferometer/features/multi_gaussian_expansion/modeling.py`), which uses an even simpler basis (no
quantum-number indexing, just log-spaced sigmas) and is often faster.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the
output folder for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script has illustrated how to use shapelets to model the source galaxy of an interferometer-observed
strong lens. Thanks to nufftax, the per-shapelet NUFFT cost is amortised on the GPU and shapelet fits to
visibility data are practical at any visibility count.

For most science cases an MGE source (see `features/multi_gaussian_expansion/`) will be faster and give
higher quality results. Shapelets may perform better for disky / star-forming source morphologies that
the smoother MGE basis struggles with, but this is not guaranteed — try both and compare.
"""

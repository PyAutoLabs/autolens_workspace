"""
Modeling Features: Linear Light Profiles (Interferometer)
==========================================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood) given
the light profile's other parameters.

Linear light profiles have been a standard tool for fitting CCD imaging data for a long time. For interferometer
data they used to be impractical, because every likelihood evaluation has to Fourier-transform each basis component
into the uv-plane, and prior NUFFT backends were not JAX-friendly. With `nufftax`
(https://github.com/GragasLab/nufftax) — a JAX-native Non-Uniform Fast Fourier Transform — the image-to-uv
transform now runs inside the same jit/vmap pipeline as the rest of the model, so the per-iteration overhead of
NUFFT-ing each basis component is amortised on the GPU. Linear light profile fits are therefore practical for
interferometer data at any visibility count, including ALMA-class datasets with tens of millions of visibilities.

Based on the advantages below, we recommend you use linear light profiles whenever fitting light profiles to
interferometer data.

__Contents__

- **Advantages & Disadvantages:** Benefits and drawbacks of linear light profiles, and how they apply to
  interferometer data specifically.
- **NUFFT (nufftax):** Why linear light profile fits to visibilities are now practical thanks to nufftax.
- **Positive Only Solver:** Ensuring positive-only solutions for linear light profile intensities.
- **Model:** Compose the lens model fitted to the data — `Isothermal` + `ExternalShear` lens mass and a
  linear `SersicCore` source. The lens light is omitted (interferometer convention).
- **Mask:** Define the `real_space_mask` which sets the grid the strong lens is evaluated on.
- **Dataset:** Load the strong lens `Interferometer` dataset, using `TransformerNUFFT` (backed by `nufftax`).
- **Over Sampling:** Interferometer modeling does not use over-sampling (covered briefly here for users
  familiar with imaging).
- **Search:** Configure the non-linear search (Nautilus).
- **Analysis:** Create the `AnalysisInterferometer` object.
- **VRAM:** Linear light profiles add negligible VRAM compared to standard light profiles.
- **Run Time:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Intensities:** How to extract solved-for `intensity` values from the result.
- **Visualization:** Visualising fits with linear light profiles requires the
  `model_obj_linear_light_profiles_to_light_profiles` helper.
- **Max Likelihood Inversion:** Access the `Inversion` object from the result.
- **Linear Objects (Internal Source Code):** The internal `linear_obj_list` representation used by the
  inversion.
- **Wrap Up:** Summary of the script and next steps.

__Advantages__

The source galaxy's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by one. The lens light is already omitted for interferometer data,
so the saving is smaller than the imaging case (where lens and source both contribute) — but the inversion still
removes the degeneracies between `intensity` and the source's shape parameters (e.g. `effective_radius`,
`sersic_index`), which are difficult degeneracies for the non-linear search to map out accurately. This produces
more reliable lens model results and the fit converges in fewer iterations.

The inversion has a relatively small computational cost on top of the NUFFT, so we reduce the model complexity
without much slow-down.

__Disadvantages__

Although the computation time of the inversion is small, it is not non-negligible. It is approximately 3-4x
slower per likelihood than using a standard light profile with a fixed `intensity`.

The gains in run times from the simpler parameter space therefore broadly balance the slower per-likelihood
evaluation. The headline benefit is reliability, not raw speed.

__NUFFT (nufftax)__

The image-to-visibilities Fourier transform is performed by a Non-Uniform Fast Fourier Transform (NUFFT),
exposed in **PyAutoLens** as `TransformerNUFFT`. The default backend is `nufftax`, a pure-JAX NUFFT that
jit-compiles and vmap-batches like the rest of the library:

  https://github.com/GragasLab/nufftax

Because `nufftax` is JAX-native, NUFFT-ing each linear basis image happens inside the same compiled likelihood
that does the inversion, mass model ray-tracing, and chi-squared sum. There is no host round-trip between
NUFFT calls, so a model with N linear light profiles costs only N forward-NUFFTs per iteration on the GPU —
fast enough that linear inversions in the visibility plane are now routinely practical.

If `nufftax` is not installed, install it via `pip install nufftax`. A legacy pynufft-backed transformer
(`TransformerNUFFTPyNUFFT`) is available as a non-JAX fallback but is not recommended for linear light profiles.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algebra solver which allows for positive and
negative values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is
as fast as positive-negative solvers. This ensures that all light profile intensities are positive and therefore
physical.

For pixelized source reconstructions on interferometer data this solver is often disabled because negative
visibility-plane noise can pull individual pixels negative without anything being wrong physically. For linear
*light profiles*, the intensity is a single physical normalisation of an extended profile, so we keep the
positive-only solver enabled.

__Model__

This script fits an `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data). This is the standard
   convention for interferometer modeling, as the lens galaxy's optical/IR emission is typically below the
   detection threshold of mm/sub-mm interferometers.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.

__Imaging Equivalent__

For the CCD-imaging version of this script, which also fits a linear `Sersic` for the lens light, see
`autolens_workspace/*/imaging/features/linear_light_profiles/modeling.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

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

This includes the method used to Fourier transform the real-space image of the strong lens to the uv-plane and
compare directly to the visibilities. We use `TransformerNUFFT`, the JAX-native Non-Uniform Fast Fourier
Transform backed by `nufftax`, which is the required choice for fast linear light profile modeling and
scales efficiently from a few hundred visibilities to tens of millions.
"""
dataset_name = "simple"
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

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling
is used, which evaluates light profiles on a higher resolution grid than the image data to ensure the
calculation is accurate.

Interferometer data does not observe galaxies in a way where over sampling is necessary, therefore all
interferometer calculations are performed without over sampling.

__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is a linear `SersicCore` [5 parameters — `intensity` is solved for analytically].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

Note how the source galaxy uses a linear light profile, meaning that its `intensity` parameter is no longer a
free parameter in the fit. There is no lens-light component (interferometer convention).

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen
refer to `start_here.ipynb` for a description of how to fix this).

This confirms that the source galaxy's light profile does not include an `intensity` parameter.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start_here.py` for a
full description).

In the `interferometer/modeling.py` example 75 live points (`n_live=75`) were used to sample parameter space.
For this linear light profile fit we keep `n_live=75` — the saving from one fewer free parameter is modest, and
the run-time benefit on interferometer data comes mostly from the reliability of the linear inversion rather
than a reduction in live points.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features",
    name="linear_light_profiles",
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

The `interferometer/modeling.py` example explains how VRAM is used during GPU-based fitting and how to print
the estimated VRAM required by a model.

For each linear light profile in the model a small additional amount of VRAM is used to store its NUFFT'd
mapping matrix column. For 1-10 linear light profiles this is a tiny amount of VRAM (e.g. < 10MB per batched
likelihood). Even for large batch sizes you almost certainly will not use enough VRAM to require monitoring.

VRAM on interferometer datasets is driven primarily by the visibility count and the real-space mask size, not
the number of linear light profiles in the model.

__Run Time__

For standard light profiles fitting interferometer data, the log likelihood evaluation time is dominated by the
NUFFT step.

For linear light profiles, the per-evaluation cost is the NUFFT plus a small additional cost from the linear
inversion. The inversion adds approximately 3-4x the cost of the inversion-only term compared to the
fixed-intensity case, but because the NUFFT typically dominates the total cost, the overall slow-down per
likelihood is usually closer to 1.1-1.5x for a model with a single linear source profile.

Because one free parameter has been removed from the model (the source `intensity`) and the parameter-space
degeneracy between `intensity` and shape parameters is broken, the total number of likelihood evaluations needed
for convergence is usually reduced. Fits using standard light profiles and linear light profiles therefore take
roughly the same wall-clock time to run. The simpler parameter space of linear light profiles means the
model-fit is more reliable, less susceptible to converging to a local maximum, and scales better if more linear
light profiles are added (e.g. an MGE source).

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output
folder for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen
refer to `start_here.ipynb` for a description of how to fix this).

This confirms that `intensity` parameters are not inferred by the model-fit.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

The source galaxy appears similar to that in the data, confirming that the `intensity` value inferred by the
inversion process is accurate.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Intensities__

The intensity of a linear light profile is not part of the model parameterization, and is therefore not
displayed in the `model.results` file.

To extract the `intensity` value of a specific component in the model, we use the `max_log_likelihood_tracer`,
which has already performed the inversion and therefore the galaxy light profiles have their solved-for
`intensity` values associated with them.
"""
tracer = result.max_log_likelihood_tracer

# The source is the only galaxy with a light profile in the interferometer model — index -1 grabs it
# regardless of tracer ordering.
print(tracer.galaxies[-1].bulge.intensity)

"""
The `Tracer` contained in the `max_log_likelihood_fit` also has the solved for `intensity` value:
"""
fit = result.max_log_likelihood_fit

tracer = fit.tracer

print(tracer.galaxies[-1].bulge.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies, a tracer) cannot be plotted because they do
not have an `intensity` value.

Therefore, a helper produces an equivalent tracer in which every linear light profile has been replaced with an
ordinary light profile carrying its solved-for `intensity`. That helper-tracer can then be visualised:
"""
tracer = result.max_log_likelihood_tracer

aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Tracer Image")


"""
__Wrap Up__

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

__Result (Advanced)__

The code below shows additional results that can be computed from a `Result` object following a fit with a
linear light profile.

__Max Likelihood Inversion__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit`, which contains the
`Inversion` object we need.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
This `Inversion` is what handled the linear algebra that produced the source `intensity` value above.

__Linear Objects (Internal Source Code)__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`.

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: Holds a list of linear light profiles and the functionality used to
   reconstruct data in an inversion. It may contain a single light profile (e.g. `lp_linear.SersicCore`) or
   many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

 - `Mapper`: The linear object used by a `Pixelization` to reconstruct data via an `Inversion`. The `Mapper`
   is specific to the `Pixelization`'s `Mesh` (e.g. a `RectangularMapper` is used for a `RectangularAdaptDensity`
   mesh).

In this example, the model has one linear `SersicCore` for the source galaxy's bulge and no lens-light
component. The inversion therefore has a single `LightProfileLinearObjFuncList` entry, which holds the source
bulge.
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities come in lists or require us to specify the linear object
we wish to use. Knowing what linear objects are in the `linear_obj_list`, and what indexes they correspond to,
is therefore important.

The single entry in this example is the source bulge.
"""
print(
    f"LightProfileLinearObjFuncList (Source SersicCore) = {inversion.linear_obj_list[0]}"
)

"""
The `LightProfileLinearObjFuncList` contains a `light_profile_list`. In this example the list has a single
light profile.
"""
print(
    f"Linear Light Profile list (Source SersicCore) = {inversion.linear_obj_list[0].light_profile_list}"
)

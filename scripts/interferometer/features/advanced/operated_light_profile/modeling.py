"""
Modeling Features: Operated Light Profiles (Interferometer)
===========================================================

It is common for galaxies to have point-source emission, for example bright emission right at their centre due
to an active galactic nuclei or a compact knot of star formation.

For CCD imaging data this emission is blurred by the telescope's Point Spread Function, and the
`imaging/features/advanced/operated_light_profile` example explains how operated light profiles fit it by
assuming the profile has already been convolved with the PSF.

Interferometer data has no PSF: the visibilities are the Fourier transform of the sky emission, and the
synthesized beam only enters when a dirty image is formed. An operated light profile therefore takes on a
simpler meaning — it is a light profile whose image-plane shape directly represents the compact emission,
with no convolution step to bypass. Its image is Fourier transformed to the visibility plane like every other
light profile.

Using operated light profiles for this compact emission keeps a lens model consistent across datasets: the
same `lp_operated` / `lp_linear_operated` component fitted to imaging data can be fitted to interferometer
data of the same lens, with the PSF-bypass behaviour applying only where a PSF exists.

__Advanced: Visibility-Space Overrides__

Internally, linear operated light profiles use the inversion's `operated_mapping_matrix_override` API to
bypass PSF convolution for imaging data. Interferometer inversions also support this override for custom
linear objects: an override supplied to an interferometer inversion bypasses the NUFFT entirely and must
therefore be a complex matrix in visibility space, of shape [total_visibilities, params] (e.g. computed via
an analytic Fourier transform). See the `LinearObj.operated_mapping_matrix_override` docstring in PyAutoArray
for the full contract. The linear operated light profiles fitted in this example do not use an override for
interferometer data — their images are NUFFT'd like any other profile.

__Model__

This script fits an `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a linear `Sersic` bulge.
 - The lens galaxy includes a linear operated `Gaussian` representing its compact nuclear emission.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear `SersicCore`.

__Fit__

For operated light profiles, there is no `fit.py` example found for standard light profiles, linear light
profiles and other examples.

This is done purely to keep the number of examples in the workspace manageable. To perform a fit with operated
light profiles, simply follow one of the other `interferometer/fit.py` examples and replace the light profiles
with operated light profiles using the API described below.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/start_here.ipynb` notebook.

__Imaging Equivalent__

For the CCD-imaging version of this script, see
`autolens_workspace/*/imaging/features/advanced/operated_light_profile/modeling.py`.

__Contents__

- **Mask:** Define the `real_space_mask` which sets the grid the strong lens is evaluated on.
- **Dataset:** Loading the operated light profile interferometer dataset from FITS files.
- **Dataset Auto-Simulation:** Automatically simulating the dataset if it does not already exist.
- **Model:** Composing the lens model with a linear Sersic bulge and operated Gaussian point source.
- **Search:** Configuring the Nautilus nested sampling non-linear search.
- **Analysis:** Creating the AnalysisInterferometer object for likelihood evaluation.
- **Run Time:** Discussion of computational run times for operated light profiles.
- **Model-Fit:** Running the model-fit and monitoring output.
- **Result:** Inspecting the result object and best-fit model.
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
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `light_operated` from .fits files, using
`TransformerNUFFT` backed by `nufftax`.
"""
dataset_name = "light_operated"
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
        [
            sys.executable,
            "scripts/interferometer/features/advanced/operated_light_profile/simulator.py",
        ],
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
__Model__

We compose a lens model where:

 - The lens galaxy's light is a linear `Sersic` bulge [6 parameters].

 - The lens galaxy's point source emission is a linear operated `Gaussian` centred on the bulge [3 parameters].

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is a linear `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

The prior on the operated `Gaussian`'s `sigma` value is very important, as it is often the case that this is a
very small value (e.g. ~0.1).

By default, **PyAutoLens** assumes a `UniformPrior` from 0.0 to 5.0, but the scale of this value depends on
the resolution of the data. I therefore recommend you set it manually below, using your knowledge of the
compact emission's angular size.

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
bulge = af.Model(al.lp_linear.Sersic)
psf = af.Model(al.lp_linear_operated.Gaussian)

psf.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)

bulge.centre = psf.centre

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    psf=psf,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling
algorithm Nautilus (https://nautilus.readthedocs.io/en/latest/).

A full description of the settings below is given in the beginner modeling scripts, if anything is unclear.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features",
    name="operated_light_profiles",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,  # GPU lens model fits are batched and run simultaneously.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisInterferometer` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

"""
__Run Time__

For interferometer data the likelihood evaluation time of an operated light profile is the same as that of an
ordinary light profile — both are evaluated in real space and NUFFT'd to the visibility plane (the PSF
convolution that operated profiles bypass for imaging data does not exist here).

The overall run-time may be a little slower than a model without the point source component though, because
the `psf` component adds a few extra parameters.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output
folder for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format:
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

The lens galaxy's bulge and compact nuclear emission appear similar to those in the data, confirming that the
`intensity` values inferred by the inversion process are accurate.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""

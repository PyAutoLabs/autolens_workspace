"""
Start Here: Interferometer
==========================

Strong lenses are observed with radio/mm interferometers (e.g. ALMA), which measure
complex visibilities in the uv-plane instead of CCD images.

This script shows you how to model such a lens system using **PyAutoLens** with as little setup
as possible. In about 15 minutes you’ll be able to point the code at your own FITS files and
fit your first lens.

We focus on a *galaxy-scale* lens (a single lens galaxy). If you have multiple lens galaxies,
see the `start_here_group.ipynb` and `start_here_cluster.ipynb` examples.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.

We also show how to simulate interferometer datasets. This is useful for building machine
learning training datasets, or for investigating lensing effects in a controlled way.

__Number of Visibilities__

This example fits a **low-resolution interferometric dataset** with a small number of visibilities (273). The
dataset is intentionally minimal so that the example runs quickly and allows you to become familiar with the API
and modeling workflow. The code demonstrated in this example can feasible fit datasets with up to around 10000
visibilities, above which computational time and VRAM use become significant for this modeling approach.

High-resolution datasets with many visibilities (e.g. high-quality ALMA observations
with **millions hundreds of millions of visibilities**) can be modeled efficiently. However, this requires
using the more advanced **pixelized source reconstructions** modeling approach. These large datasets fully
exploit **JAX acceleration**, enable lens modeling to run in **hours on a modern GPU**.

If your dataset contains many visibilities, you should start by working through this example and the other examples
in the `interferometer` folder. Once you are comfortable with the API, the `feature/pixelization` package provides a
guided path toward efficiently modeling large interferometric datasets.

The threshold between a dataset having many visibilities and therefore requiring pixelized source reconstructions, or
being small enough to be modeled with light profiles, is around **10,000 visibilities**.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
"""

import subprocess
import sys

try:
    import google.colab

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autoconf", "--no-deps"]
    )
except ImportError:
    pass

from autoconf import setup_colab

setup_colab.for_autolens(
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import numpy as np

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Mask (Real Space)__

Interferometer modeling evaluates the lens image on a *real-space grid* and Fourier transforms
to the uv-plane to compare with visibilities. 

We therefore define a circular real-space mask, which sets the pixel grid size and pixel-to-arcsecond 
pixel scale in real space.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

We begin by loading an `Interferometer` dataset from FITS, three ingredients are needed for lens modeling:

- `data.fits`: complex visibilities (shape: n_vis)
- `noise_map.fits`: per-visibility complex RMS
- `uv_wavelengths.fits`: (u, v) sampling of the interferometer in wavelengths

We must also choose a transformer:

- `TransformerDFT`: exact Discrete FT (robust, slower for large n_vis).
- `TransformerNUFFT`: approximate Non-Uniform FFT (fast, accurate for large n_vis) using the pynufft library.

We load a low resolution Square Mile Array (SMA) dataset for this example, which has just
273 visibilities. This has so few visibilities that we can use the exact DFT transformer without
the computation being too slow (for larger datasets with many visibilities the NUFFT transformer is recommended).
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()  # quick look at dirty image / PSF

"""
__Model__

To perform lens modeling we must define a lens model, describing the light profiles of 
the source galaxy, and the mass profile of the lens galaxy.

A brilliant lens model to start with is one which uses a Multi Gaussian Expansion (MGE) 
to model the lens and source light, and a Singular Isothermal Ellipsoid (SIE) plus 
shear to model the lens mass. 

Full details of why this models is so good are provided in the main workspace docs, 
but in a nutshell it  provides an excellent balance of being fast to fit, flexible 
enough to capture complex galaxy morphologies and providing accurate fits to the vast 
majority of strong lenses.

The MGE model composition API is quite long and technical, so we simply load the MGE 
models for the source below via a utility function `mge_model_from` which 
hides the API to make the code in this introduction example ready to read. We then 
use the PyAutoLens Model API to compose the over lens model.
"""
# Lens galaxy

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source galaxy
source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Compose model
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
We can print the model to show the parameters that the model is composed of, which shows many of the MGE's fixed
parameter values the API above hided the composition of.
"""
print(model.info)

"""
__Model Fit__

We now fit the data with the lens model using the non-linear fitting method and nested sampling algorithm Nautilus.

We fit the visibilities with `AnalysisInterferometer`, which defines the `log_likelihood_function` used by 
Nautilus to fit the model to the interferometer data.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce 
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer"),  # The path where results and output are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=75,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see modeling examples for details.
    iterations_per_quick_update=10000,  # Every N iterations the max likelihood model is visualized in the Jupter Notebook and output to hard-disk.
)

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
The code below begins the model-fit. This will take around 10 minutes with a GPU, or 20-30 minutes with a CPU.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce 
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

Now this is running you should checkout the `autolens_workspace/output` folder, where many results of the fit
are written in a human readable format (e.g. .json files) and .fits and .png images of the fit are stored.

When the fit is complex, we can print the results by printing `result.info`.
"""
print(result.info)

"""
The result also contains the maximum likelihood lens model which can be used to plot the best-fit lensing information
and fit to the data.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer,
    grid=result.grids.lp,  # model evaluation grid
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()  # residuals, chi^2, dirty image, etc.

"""
The result object contains pretty much everything you need to do science with your own strong lens, but details
of all the information it contains are beyond the scope of this introductory script. The `guides` and `result` 
packages of the workspace contains all the information you need to analyze your results yourself.

__Model Your Own Lens__

If you have your own strong lens interferometer data, and it has less than ~10000 visibilities, you are now ready to 
model it yourself by adapting the code above and simply inputting the path to your own .fits files into 
the `Interferometer.from_fits()` function.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- Supply your own visibilities, noise-map and uv-wavelengths .fits files.
- Ensure the lens galaxy is roughly centered in the image.
- Double-check `pixel_scales` for the real space mask of your interferometer.
- Adjust the mask radius to include all relevant light.
- Start with the default model — it works very well for pretty much all galaxy scale lenses!

__Simulator__

Let’s now switch gears and simulate our own strong lens interferometer. This is a great way to:

- Practice lens modeling before using real data.
- Build large training sets (e.g. for machine learning).
- Test lensing theory in a controlled environment.

To do this we need to define a 2D grid of (y,x) coordinates in the image-plane. This grid is
where we’ll evaluate the light from the lens and source galaxies.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

"""
We now define a `Tracer` — this is the key object that combines all galaxies in the system
and computes how light rays are deflected.

- The lens galaxy has mass (an isothermal profile + shear).
- The source galaxy has its own light (a SersicCore profile).

Together they define a strong lens system. The tracer will “ray-trace” our grid through
this mass distribution and generate a lensed image.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Plotting the tracer’s image gives us a “perfect” view of the strong lens system, before
adding telescope effects.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
The image cna be saved to .fits for later use.
"""
image = tracer.image_2d_from(grid=grid)

al.output_to_fits(
    values=image.native,
    file_path=Path("image.fits"),
    overwrite=True,
)

"""
__Simulator__

The images above do not represent real interferometer data, as they do not include the transform of the data 
to visibilities or any noise. 

The `SimulatorInterferometer` class simulates these two key properties of real interferometer data, which we use below to 
create realistic interferometer data of the strong lens system.

The units of the image are arbitrary, with the workspace providing guides on how to convert to physical units for lens
simulations.

The code below performs the simulation, plots the simulated interferometer data and outputs it to .fits files with .png
files included for easy visualization.
"""
# You could put your own uv_wavelengths.fits file here to simulate your own interferometer.
uv_wavelengths = dataset.uv_wavelengths

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,  # Length of observation in seconds, higher time = higher S/N
    noise_sigma=1000.0,  # RMS of the complex Gaussian noise added to the visibilities
    transformer_class=al.TransformerDFT,  # keep consistent with your modeling choice
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(dirty_image=True)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

# Save simulated visibilities/uv-wavelengths/noise to FITS (same format as real)

dataset_path = Path("dataset") / "interferometer" / "simulated_lens"

dataset.output_to_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

"""
__Sample__

Often we want to simulate *many* strong lenses — for example, to train a neural network
or to explore population-level statistics.

This uses the model composition API to define the distribution of the light and mass profiles
of the lens and source galaxies we draw from. The model composition is a little too complex for
the first example, thus we use a helper function to create a simple lens and source model.

We then generate 3 lenses for speed, and plot their images so you can see the variety of lenses
we create.

Each lens is simulated as if it were observed with an interferometer, therefore with a PSF and noise-map.
"""
lens_model, source_model = al.model_util.simulator_start_here_model_from(
    include_lens_light=False
)

print(lens_model.info)
print(source_model.info)

"""
We now simulate a sample of strong lenses, we just do 3 for efficiency here but you can increase this to any number.
"""
total_datasets = 3

for sample_index in range(total_datasets):

    lens_galaxy = lens_model.random_instance()
    source_galaxy = source_model.random_instance()

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
    dataset_plotter.subplot_dirty_images()


"""
__Wrap Up__

This script has shown how to model interferometer data of strong lenses, and simulate your own strong lens datasets.

Details of the **PyAutoLens** API and how lens modeling and simulations actually work were omitted for simplicity,
but everything you need to know is described throughout the main workspace documentation. You should check it out,
but maybe you want to try and model your own lens first!

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/*/interferometer/modeling`: A full description of the lens modeling API and how to customize your model-fits.
- `autolens_workspace/*/interferometer/simulators`: A full description of the lens simulation API and how to customize your simulations.
- `autolens_workspace/*/interferometer/data_preparation`: How to load and prepare your own interferometer data for lens modeling.
- `autolens_workspace/guides/results`: How to load and analyze the results of your lens model fits, including tools for large samples.
- `autolens_workspace/guides`: A complete description of the API and information on lensing calculations and units.
- `autolens_workspace/interferometer/features`: A description of advanced features for lens modeling, for example pixelized source reconstructions, read this once you're confident with the basics!
"""

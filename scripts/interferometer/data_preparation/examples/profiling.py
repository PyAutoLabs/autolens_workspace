"""
Profiling: Interferometer Inversion
===================================

For performing an `Inversion` to interferometer data, there are a variety of settings that can be chosen which produce
numerically equivalent solutions but change how the calculation is performed.

The run time of the `Inversion` varies considerably depending on what settings are used, and the fastest settings
depend on the number of visibilities in the interferometer dataset as well as its `uv_wavelengths`.

This script allows you to load an interferometer dataset, define the `real_space_mask` and fit it for all combinations
of different settings to determine which settings give the fastest results for your dataset.

To fit the dataset a lens mass model is omitted, given we have not modeled the dataset yet. Whilst the solution we use
is therefore not an actual strong lens model, it is appropriate for determining the fastest settings.

Some of the settings will use extremely large amounts of memory (e.g. > 100GB) for large visibility datasets
(e.g. > 100000) and may crash your computer. To prevent this, their profiling function is commented out below. However,
these settings may give the fastest performance for low visibility datasets (e.g. < 1000). If your dataset has
low numbers of visibilities you should comment these lines of code out to compare their run times.

__Linear Algebra Formalism__

There are two ways the linear algebra can be calculated for an `Inversion`:

 - **Matrices:** Use a numerically more accurate matrix formalism to perform the linear algebra. For datasets
 of < 100 0000 visibilities this approach is computationally feasible, and if your dataset is this small we recommend
 that you use this option because it is faster (by setting `use_linear_operators=False`. However, larger visibility
 datasets these matrices require excessive amounts of memory (> 16 GB) to store, making this approach unfeasible.

 - **Linear Operators (default)**: These are slightly less accurate, but do not require excessive amounts of memory to
 store the linear algebra calculations. For any dataset with > 1 million visibilities this is the only viable approach
 to perform lens modeling efficiently.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import time

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Fit Time__

This function is used throughout this script to time how long a fit takes for each combination of settings.
"""


def print_fit_time_from(
    interferometer, transformer_class, use_w_tilde, use_linear_operators, repeats=1
):
    settings_dataset = al.SettingsInterferometer(transformer_class=transformer_class)
    dataset = dataset.apply_settings(settings=settings_dataset)

    """
    __Numba Caching__

    Call FitImaging once to get all numba functions initialized.
    """
    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_w_tilde=use_w_tilde, use_linear_operators=use_linear_operators
        ),
    )
    print(fit.figure_of_merit)

    """
    __Fit Time__

    Time FitImaging by itself, to compare to profiling dict call.
    """
    start = time.time()
    for i in range(repeats):
        fit = al.FitInterferometer(
            dataset=dataset,
            tracer=tracer,
            settings_inversion=al.SettingsInversion(
                use_w_tilde=use_w_tilde, use_linear_operators=use_linear_operators
            ),
        )
        fit.figure_of_merit

    fit_time = (time.time() - start) / repeats
    print(f"Fit Time = {fit_time} \n")


"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.2, radius=3.0, sub_size=1
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files , which we will fit 
with the lens model.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Tracer__

Set up the `Tracer` used to profile each method, which:
 
 - Does not implement mass or light profiles for the lens galaxy.
 - Uses a `DelaunayMagnification` mesh with `Constant` regularization to fit the data and thus profile the 
  `Inversion` run time.
"""
lens_galaxy = al.Galaxy(redshift=0.5)

pixelization = al.Pixelization(
    mesh=al.mesh.DelaunayMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
__DFT + Matrices (Mapping)__

Compute the run-time using:

 - `TransformerDFT`: The Discrete Fourier Transform. ,
  - `use_linear_operators=False`: this uses the `Inversion` matrix formalism (as opposed to the linear_operator formalism).
 
These settings are fastest for interferometer datasets with < 1000 visibilities. 

They scale poorly to datasets with > 10000 visibilities which will use large quantities of memory, thus the
code below is commented out by default.
"""
print_fit_time_from(
    dataset=interferometer,
    transformer_class=al.TransformerDFT,
    use_w_tilde=False,
    use_linear_operators=False,
)

"""
__NUFFT + Matrices (Mapping)__

Compute the run-time using:

 - `TransformerNUFFT`: The Non-Uniform Fast Fourier Transform. ,
 - `use_linear_operators=False`: this uses the `Inversion` matrix formalism (as opposed to the linear_operator formalism).

These settingsare fastest for interferometer datasets with ~ 10000 visibilities. 

They scale poorly to datasets with < 1000 and > 10000 visibilities which will use large quantities of memory, thus the
code below is commented out by default.
"""
print_fit_time_from(
    dataset=interferometer,
    transformer_class=al.TransformerNUFFT,
    use_w_tilde=False,
    use_linear_operators=False,
)

"""
__NUFFT + Linear Operators__

Compute the run-time using:

 - `TransformerNUFFT`: The Non-Uniform Fast Fourier Transform. ,
  - `use_linear_operators=True`: this uses the `Inversion` linear operator formalism (as opposed to the matrix 
  formalism).

These settings are fastest for interferometer datasets with > 100000 visibilities. 

They scale poorly to datasets with < 10000 visibilities.
"""
# print_fit_time_from(
#     dataset=interferometer,
#     transformer_class=al.TransformerNUFFT,
#     use_w_tilde=False,
#     use_linear_operators=True
# )

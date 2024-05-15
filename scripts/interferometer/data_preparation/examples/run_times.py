"""
Data Preparation: Run Times
===========================

The run times of an interferometer analysis depend significantly on how many visibilities are in the dataset. The
settings of an interferometer analysis must therefore be chosen based on the dataset being fitted.

Analyses which perform an interferometer pixelization reconstruction (called an `Inversion`) also depend on the number
of visibilities, with additional settings that can be chosen to improve the run time.

This script allows you to load an interferometer dataset, define the `real_space_mask` and fit it with different
settings to determine which give the fastest results for your dataset.

To fit the dataset a lens mass model is omitted, because we have not modeled the dataset yet. Whilst the solution we use
is therefore a poor fit, it wills till give representaitive run times.

Some settings may use extremely large amounts of memory (e.g. > 100GB), if your datasset have many visibilities
(e.g. > 1 000 000). This may crash your computer.

To prevent this, functions which provide run times are commented out below, and you will need to uncomment them
depending on whether they are suitable for your data (e.g. they typically give the best performance for datasets
with less visibilitlies, around < 100 000).

__Preloading Time__

Some functionality takes longer the first time it is run, as it is preloading in memory certain quantities that are
reused many times when lens modeling is performed.

This means that when profiling settings below it may appear that the function is very slow, but actually it is
performing this preloading. The run times provided by the functions below do not include this preloading time (as
this is representative of the run time of a lens model analysis).

You therefore should not cancel the script if it appears to be running slowly, as it could be this preloading time
that is the cause.
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
__Transformer Time__

This function is used to time how long a transformer takes to map the real-space image of a strong lens to its
visibilities. This is used to determine which transformer is optimal for your dataset.
"""


def print_transformer_time_from(dataset, transformer_class, repeats=1):
    """
    __Numba Caching__

    Perform a single transformer call to ensure all numba functions are initialized.
    """
    image = tracer.image_2d_from(grid=dataset.grid)

    dataset.transformer.visibilities_from(image=image)

    """
    __Fit Time__

    Now profile the overall run-time of the transformer.
    """
    start = time.time()
    dataset.transformer.visibilities_from(image=image)

    transformer_time = (time.time() - start) / repeats
    print(f"Transformer Time = {transformer_time} \n")


"""
__Fit Time__

This function is used throughout this script to time how long a fit takes for each combination of settings.
"""


def print_fit_time_from(dataset, transformer_class, use_linear_operators, repeats=1):
    """
    __Numba Caching__

    Call FitImaging once to get all numba functions initialized.
    """
    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(
            use_linear_operators=use_linear_operators
        ),
    )
    print(fit.figure_of_merit)

    """
    __Fit Time__

    Time FitImaging by itself, to compare to run_times dict call.
    """
    start = time.time()
    for i in range(repeats):
        fit = al.FitInterferometer(
            dataset=dataset,
            tracer=tracer,
            settings_inversion=al.SettingsInversion(
                use_linear_operators=use_linear_operators
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
    shape_native=(800, 800),
    pixel_scales=0.2,
    radius=3.0,
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
 - Uses an `Overlay` image-mesh, `Delaunay` mesh with `Constant` regularization to fit the data and thus profile the 
  pixelized source reconstruction `Inversion` run time.
"""
lens_galaxy = al.Galaxy(redshift=0.5)

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Transformer__

The transformer maps the inversion's image from real-space to Fourier space, with two options available that have
optimal run-times depending on the number of visibilities in the dataset:

- `TransformerDFT`: A discrete Fourier transform which is most efficient for < ~100 000 visibilities.

- `TransformerNUFFT`: A non-uniform fast Fourier transform which is most efficient for > ~100 000 visibilities.

If your dataset has < ~100 000 visibilities, you should confirm whether the DFT is faster than the NUFFT for your
specific dataset and use that setting in your modeling scripts.

For datasets with > ~100 000 visibilities, the DFT uses a lot of memory and has very long run times. You may still 
wish to profile it below, but it can use a lot of memory so proceed with caution!
"""
dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

print_transformer_time_from(
    dataset=dataset, transformer_class=al.TransformerDFT, repeats=1
)

"""
__Linear Algebra__

The linear algebra describes how the linear system of equations used to reconstruct a source via a pixelization is
solved. 

The optimal transformer does not depend on the linear algebra settings, thus because we have found the optimal
transformer above we can now easily choose the optimal linear algebra settings.

There are with three options available that again have run-times that are optimal for datasets of different sizes 
(do not worry if you do not understand how the linear algebra works, all you need to do is ensure you choose the
setting most appropriate for the size of your dataset):

- `use_linear_operators`:  If `False`, the matrices in the linear system are computed via a `mapping_matrix`, which 
  is optimal for datasets with < ~10 000 visibilities.
  
- `use_linear_operators`: If `True`, a different formalism is used entirely where matrices are not computed and 
   linear operators  are used instead. This is optimal for datasets with > ~1 000 000 visibilities. Note that 
   the `TransformerNUFFT` must be used with this setting.

If your dataset has > 1 000 000 visibilities, you should be cautious that using `use_linear_operations=False` 
 will use significant amounts of memory and take a long time to run. 

You should now vary the settings below to determine the optimal settings for your dataset, making sure to use the
optimal transformer determined above.
"""
dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

print_fit_time_from(
    dataset=dataset,
    transformer_class=al.TransformerNUFFT,
    use_linear_operators=False,
)

"""
Fin.
"""

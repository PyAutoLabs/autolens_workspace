"""
Plots: Interferometer
=====================

This example shows how to plot an `Interferometer` dataset and a `FitInterferometer` fit, figure by figure and
via multi-panel subplots.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autolens.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`.

The final section documents the `Visualizer`, which outputs all of these figures automatically
during a model-fit via the `Analysis` object.

__Contents__

- **Dataset:** Load the interferometer dataset used throughout this example.
- **Dataset Subplot:** Plot the dataset's uv-plane quantities in one multi-panel subplot.
- **Dirty Images Subplot:** Plot the dataset's dirty images in one multi-panel subplot.
- **Fit:** Set up a tracer and fit the dataset with a `FitInterferometer` object.
- **Fit Figures:** Plot the fit's dirty image, dirty model image, dirty residuals, dirty normalized residuals and dirty chi-squared maps individually.
- **Fit Subplot:** Plot all fit quantities in one multi-panel subplot.
- **Fit Dirty Images Subplot:** Plot the fit's dirty images in one multi-panel subplot.
- **Real Space Subplot:** Plot the fit's real-space image and source reconstruction.
- **Tracer Subplot:** Plot the fit's tracer quantities (convergence, potential, deflections).
- **Outputting to FITS:** Write the dataset to a FITS file instead of an image.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens dataset `simple` from .fits files, which is the dataset used to demonstrate
plotting.

Interferometer data is evaluated on a `real_space_mask`, which defines the grid the image of the strong lens
is computed on before it is Fourier transformed to the uv-plane. We use `TransformerNUFFT`, the JAX-native
Non-Uniform Fast Fourier Transform backed by `nufftax`, which scales efficiently from a few hundred visibilities
to tens of millions.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

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

"""
__Dataset Subplot__

`aplt.subplot_interferometer_dataset()` produces a multi-panel subplot of the dataset in the
uv-plane — the visibilities, their noise-map and the uv-wavelengths sampled by the interferometer.
"""
aplt.subplot_interferometer_dataset(dataset=dataset)

"""
__Dirty Images Subplot__

Visibility data is in uv-space, making it hard to interpret by eye. A multi-panel subplot of the dataset's
dirty images — the data, noise-map and other quantities mapped back to real-space via the dataset's
transformer — is produced with `aplt.subplot_interferometer_dirty_images()`.
"""
aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Fit__

To plot a fit, we fit the dataset with a tracer whose galaxies match the true simulated values, via the
`FitInterferometer` object.
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
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

"""
__Fit Figures__

The fit's quantities are visibilities in the uv-plane, which are hard to interpret by eye. Instead, the fit's
`dirty` variants — its image-space representations, computed by Fourier transforming each quantity back to
real-space via the dataset's transformer — are accessed as attributes and plotted individually with
`aplt.plot_array()`.
"""
aplt.plot_array(array=fit.dirty_image, title="Dirty Image")
aplt.plot_array(array=fit.dirty_model_image, title="Dirty Model Image")
aplt.plot_array(array=fit.dirty_residual_map, title="Dirty Residual Map")
aplt.plot_array(
    array=fit.dirty_normalized_residual_map, title="Dirty Normalized Residual Map"
)
aplt.plot_array(array=fit.dirty_chi_squared_map, title="Dirty Chi-Squared Map")

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_interferometer()`, combining the dirty image,
dirty model image, dirty residual-map and dirty chi-squared map in one figure, as well as other information
contained in the tracer such as the source-plane image.
"""
aplt.subplot_fit_interferometer(fit=fit)

"""
__Fit Dirty Images Subplot__

`aplt.subplot_fit_dirty_images()` collects the fit's dirty images — data, model data, residuals and
chi-squared — into a single subplot, the fit-level counterpart of the dataset subplot above.
"""
aplt.subplot_fit_dirty_images(fit=fit)

"""
__Real Space Subplot__

`aplt.subplot_fit_interferometer_real_space()` plots the fit's real-space quantities: the image of
the tracer evaluated on the `real_space_mask` and the source-plane reconstruction. These are the
quantities being Fourier transformed to the uv-plane, so they show what the model actually looks
like on the sky.
"""
aplt.subplot_fit_interferometer_real_space(fit=fit)

"""
__Tracer Subplot__

`aplt.subplot_fit_interferometer_tracer()` plots the tracer quantities of the fit — image,
convergence, potential and deflection angles — without needing to build the tracer and grid
separately.
"""
aplt.subplot_fit_interferometer_tracer(fit=fit)

"""
__Outputting to FITS__

The dataset itself can be written to a FITS file with `aplt.fits_interferometer()`. Passing
`file_path` writes a single multi-HDU file; passing `data_path` / `noise_map_path` /
`uv_wavelengths_path` instead writes each component to its own file.
"""
output_path = Path("output") / "plot" / "interferometer"
output_path.mkdir(parents=True, exist_ok=True)

aplt.fits_interferometer(
    dataset=dataset,
    file_path=output_path / "dataset.fits",
    overwrite=True,
)

"""
__Visualizer__

During a model-fit (e.g. `search.fit(model=model, analysis=analysis)` in `modeling.py`), all of
the figures above are output to hard-disk automatically — you do not need to call the plotting
functions yourself to inspect a fit's progress or results.

This is performed by the `Visualizer` attached to the `Analysis` class:
"""
print(al.AnalysisInterferometer.Visualizer)

"""
At regular intervals during the non-linear search, and again once it finishes, the `Visualizer`
computes the maximum likelihood fit and outputs its figures to the fit's output folder, under
`image/` (e.g. `output/<path_prefix>/<name>/image/`).

Which figures are output is controlled by the config file `config/visualize/plots.yaml`, for
example:

 - `dataset` -> `subplot_dataset`: The multi-panel dataset subplot shown above.
 - `fit` -> `subplot_fit`: The multi-panel fit subplot shown above.
 - `fit_interferometer` -> `subplot_fit_dirty_images`: A further subplot of the dirty images of the data,
   model data, residuals and chi-squared.
 - `tracer` -> `subplot_tracer`: A subplot of the tracer's image, convergence and potential.

Setting an entry to `true` or `false` in `plots.yaml` therefore switches that figure on or off
for every model-fit, without changing code.
"""

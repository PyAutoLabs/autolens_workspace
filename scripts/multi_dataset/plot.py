"""
Plots: Multi
============

This example shows how to plot multiple datasets — and fits to multiple datasets — together,
with every dataset appearing in one combined subplot.

This uses the same functions the source code's `Visualizer` uses when it outputs figures during a
multi-dataset model-fit:

 - `aplt.subplot_imaging_dataset_list()` — all datasets in one subplot (one row per dataset).
 - `aplt.subplot_fit_combined()` — all fits in one subplot (one row per fit).
 - `aplt.subplot_fit_interferometer_combined()` — all interferometer fits in one subplot (one row
   per fit, which for a datacube means one row per channel).

The example works through two multi-dataset settings in turn. It first loads a multi-wavelength
imaging dataset and plots the g-band and r-band data and fits together, then loads a multi-channel
interferometer datacube and plots every channel's fit together. For an introduction to the plotting
API refer to `guides/plot/start_here.py`; for single-dataset fit plotting refer to
`scripts/imaging/plot.py` and `scripts/interferometer/plot.py`.

__Contents__

- **Dataset:** Load the multi-wavelength strong lens datasets.
- **Single Dataset Subplots:** Plot the subplot overview of each dataset one-by-one.
- **Combined Dataset Subplot:** Plot all datasets in one subplot with `aplt.subplot_imaging_dataset_list()`.
- **Fits:** Fit each waveband's dataset with a tracer using its true simulated values.
- **Combined Fit Subplot:** Plot all fits in one subplot with `aplt.subplot_fit_combined()`.
- **Multi Fits:** Output a list of figures to a single `.fits` file, where each image goes in each HDU.
- **Interferometer Datasets:** Load a multi-channel interferometer datacube as a list of `Interferometer` objects.
- **Interferometer Fits:** Fit each channel with its true simulated tracer.
- **Combined Interferometer Fit Subplot:** Plot all interferometer fits in one subplot with `aplt.subplot_fit_interferometer_combined()`.
- **Visualizer:** How combined figures are output automatically during a multi-dataset model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the multi-wavelength `lens_sersic` datasets.
"""
waveband_list = ["g", "r"]

pixel_scales_list = [0.08, 0.12]

dataset_type = "multi_dataset"
dataset_label = "imaging"
dataset_name = "lens_sersic"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_dataset/simulator.py"],
        check=True,
    )

dataset_list = [
    al.Imaging.from_fits(
        data_path=Path(dataset_path) / f"{waveband}_data.fits",
        psf_path=Path(dataset_path) / f"{waveband}_psf.fits",
        noise_map_path=Path(dataset_path) / f"{waveband}_noise_map.fits",
        pixel_scales=pixel_scales,
    )
    for waveband, pixel_scales in zip(waveband_list, pixel_scales_list)
]

"""
__Single Dataset Subplots__

Each dataset's subplot overview can be plotted one-by-one with `aplt.subplot_imaging_dataset()`.
"""
for dataset in dataset_list:
    aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Combined Dataset Subplot__

To compare the datasets it is more useful to see them in a single figure. The
`aplt.subplot_imaging_dataset_list()` function plots every dataset in one subplot, with one row
per dataset showing its data, noise-map and signal-to-noise map.
"""
aplt.subplot_imaging_dataset_list(dataset_list=dataset_list)

"""
__Fits__

To plot fits to every dataset, we mask each dataset and fit it with a tracer using the true
simulated values.

The lens, source and extra galaxy have a different `intensity` at each wavelength (see
`scripts/multi_dataset/simulator.py`), so a separate tracer is composed per waveband; the mass model is
the same at all wavelengths.
"""
dataset_list = [
    dataset.apply_mask(
        mask=al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=3.0,
        )
    )
    for dataset in dataset_list
]

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

extra_galaxy_centre = (2.2, 1.6)

lens_intensity_list = [0.05, 1.5]
source_intensity_list = [0.5, 0.7]
extra_intensity_list = [0.4, 1.0]

tracer_list = [
    al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                bulge=al.lp.Sersic(
                    centre=(0.0, 0.0),
                    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
                    intensity=lens_intensity,
                    effective_radius=0.8,
                    sersic_index=4.0,
                ),
                mass=mass,
            ),
            al.Galaxy(
                redshift=0.5,
                light=al.lp.ExponentialSph(
                    centre=extra_galaxy_centre,
                    intensity=extra_intensity,
                    effective_radius=0.3,
                ),
            ),
            al.Galaxy(
                redshift=1.0,
                bulge=al.lp.Sersic(
                    centre=(0.0, 0.0),
                    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
                    intensity=source_intensity,
                    effective_radius=0.1,
                    sersic_index=1.0,
                ),
            ),
        ]
    )
    for lens_intensity, source_intensity, extra_intensity in zip(
        lens_intensity_list, source_intensity_list, extra_intensity_list
    )
]

fit_list = [
    al.FitImaging(dataset=dataset, tracer=tracer)
    for dataset, tracer in zip(dataset_list, tracer_list)
]

"""
__Combined Fit Subplot__

The `aplt.subplot_fit_combined()` function plots every fit in one subplot, with one row per fit
showing its data, lens-subtracted image, model images, source plane and normalized residuals.

This is the figure to inspect when checking that a multi-wavelength model fits all datasets well
simultaneously.
"""
aplt.subplot_fit_combined(fit_list=fit_list)

"""
A log10 version highlights the fainter regions of each fit.
"""
aplt.subplot_fit_combined_log10(fit_list=fit_list)

"""
__Multi Fits__

We can also output a list of figures to a single `.fits` file, where each image goes in
each HDU extension.
"""
from autolens import hdu_list_for_output_from

dataset = dataset_list[-1]

image_list = [dataset.data, dataset.noise_map]

hdu_list = hdu_list_for_output_from(
    values_list=[image_list[0].mask.astype("float")] + image_list,
    ext_name_list=["mask"] + ["data", "noise_map"],
    header_dict=dataset.mask.header_dict,
)

hdu_list.writeto("dataset.fits", overwrite=True)

"""
__Interferometer Datasets__

Interferometer data has its own multi-dataset setting: the datacube. A datacube is a list of
`Interferometer` objects observing the same strong lens, one per spectral channel, and it is the
interferometer analogue of the multi-wavelength imaging list above — the same lens seen through a
different slice of the spectrum in each entry of the list.

We therefore reach into the dataset simulated by `scripts/interferometer/features/datacube/`, which
is the workspace's list-of-`Interferometer` example and exactly the case the combined interferometer
plotter was written for. The cube has four channels of a lensed emission line: the lens mass is
identical in every channel, whereas the source's `intensity` follows a Gaussian emission-line
profile across the cube and its `centre` drifts along the y axis to mimic a kinematic gradient.

Every channel is masked with the same `real_space_mask`, the grid the lensed image is evaluated on
before it is Fourier transformed to the uv-plane. The lens and source do not move with frequency, so
masking once and reusing the mask for every channel is correct.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=3.5,
)

dataset_path = Path("dataset") / "interferometer" / "datacube" / "sim_simple"

"""
__Dataset Auto-Simulation__

As with the imaging datasets above, if the cube is not already on your system it is created by
running the corresponding simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/features/datacube/simulator.py"],
        check=True,
    )

"""
The cube is stored as one folder per channel (`channel_000/`, `channel_001/`, ...), each holding
that channel's `data.fits`, `noise_map.fits` and `uv_wavelengths.fits`. We discover the channels by
sorted directory listing and load each one as an `Interferometer` object, giving a plain Python list
— there is no special datacube class.

We use `al.TransformerDFT`, the direct Fourier transform, because this cube has only a few hundred
visibilities and the DFT is the cheaper choice at that size. For real data with many more
visibilities use `al.TransformerNUFFT`, as
`scripts/interferometer/features/datacube/modeling.py` does.
"""
channel_paths = sorted(
    p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("channel_")
)

dataset_list = [
    al.Interferometer.from_fits(
        data_path=channel_path / "data.fits",
        noise_map_path=channel_path / "noise_map.fits",
        uv_wavelengths_path=channel_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )
    for channel_path in channel_paths
]

"""
Each channel's dirty images can be plotted one-by-one with
`aplt.subplot_interferometer_dirty_images()`, the interferometer counterpart of the per-dataset
subplots plotted at the top of this example.

The source is brightest in the central channels, where the emission line peaks, and fainter in the
outer channels, so the lensed signal visibly strengthens and fades as you step through the cube.
"""
for dataset in dataset_list:
    aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Interferometer Fits__

To fit each channel we load the true tracer that the simulator wrote alongside that channel's data,
in `tracer.json`, and pair it with the channel's dataset in a `FitInterferometer` object.

These fits are genuinely different from one another: every channel has its own visibilities and its
own noise realisation, and its tracer carries that channel's own source `intensity` and source
`centre`. That distinction matters, because a `fit_list` exists to hold fits that differ — it should
never be the same fit repeated, which would produce a subplot of identical rows that says nothing
about the data.
"""
tracer_list = [
    al.from_json(file_path=channel_path / "tracer.json")
    for channel_path in channel_paths
]

fit_list = [
    al.FitInterferometer(dataset=dataset, tracer=tracer)
    for dataset, tracer in zip(dataset_list, tracer_list)
]

"""
__Combined Interferometer Fit Subplot__

The `aplt.subplot_fit_interferometer_combined()` function plots every interferometer fit in one
subplot, with one row per fit — which for a datacube means one row per channel, in channel order.

Each row has four panels: the dirty image (the data), the dirty model image with the critical curves
overlaid, the source plane (the source-plane reconstruction), and the dirty normalized residual map.
This is a different panel choice to the six-panel imaging layout of `aplt.subplot_fit_combined()`
above, for a physical reason: an interferometer measures visibilities in the uv-plane, so there is
no image to look at until the visibilities are transformed back to real space. The dirty images are
that view, and they are where a poor fit shows itself.

This is the figure to inspect when checking that one shared lens model fits every channel of a cube
simultaneously — the residuals should be featureless in every row, while the source plane changes
row-to-row as the emission line brightens, fades and drifts.

As with the imaging equivalent, `title_prefix=` prepends a label to every panel title and
`colormap=` sets the colormap used for all image panels. Unlike the imaging case, there is no
`_log10` variant of this function.
"""
aplt.subplot_fit_interferometer_combined(fit_list=fit_list)

"""
__Visualizer__

During a multi-dataset model-fit (e.g. combining analyses with `af.AnalysisFactor` as in
`scripts/multi_dataset/modeling.py`), the `Visualizer` attached to the `Analysis` class outputs the
combined figures above automatically:

 - Before the fit begins, all datasets are output together via `subplot_imaging_dataset_list`.
 - During and after the fit, the maximum likelihood fit to every dataset is output together via
   `subplot_fit_combined`.

These appear in the fit's output folder under `image/` (e.g. `dataset_combined.png`,
`fit_combined.png`), alongside the per-dataset figures described in `scripts/imaging/plot.py`.

Which figures are output is controlled by `config/visualize/plots.yaml`, e.g. the
`dataset` -> `subplot_dataset` and `fit` -> `subplot_fit` entries.

The same holds for a multi-dataset interferometer fit, for example the datacube `FactorGraphModel`
fit in `scripts/interferometer/features/datacube/modeling.py`. There the `Visualizer` writes
`fit_combined.png` into the fit's `image/` folder via `subplot_fit_interferometer_combined`, giving
you the row-per-channel figure above for the maximum likelihood model as the fit proceeds. It is
controlled by the same `fit` -> `subplot_fit` entry of `config/visualize/plots.yaml`.
"""

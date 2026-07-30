"""
Plots: Multi
============

This example shows how to plot multiple datasets — and fits to multiple datasets — together,
with every dataset appearing in one combined subplot.

This uses the same functions the source code's `Visualizer` uses when it outputs figures during a
multi-dataset model-fit:

 - `aplt.subplot_imaging_dataset_list()` — all datasets in one subplot (one row per dataset).
 - `aplt.subplot_fit_combined()` — all fits in one subplot (one row per fit).

The specific example loads a multi-wavelength imaging dataset and plots the g-band and r-band
data and fits together. For an introduction to the plotting API refer to
`guides/plot/start_here.py`; for single-dataset fit plotting refer to `scripts/imaging/plot.py`.

__Contents__

- **Dataset:** Load the multi-wavelength strong lens datasets.
- **Single Dataset Subplots:** Plot the subplot overview of each dataset one-by-one.
- **Combined Dataset Subplot:** Plot all datasets in one subplot with `aplt.subplot_imaging_dataset_list()`.
- **Fits:** Fit each waveband's dataset with a tracer using its true simulated values.
- **Combined Fit Subplot:** Plot all fits in one subplot with `aplt.subplot_fit_combined()`.
- **Multi Fits:** Output a list of figures to a single `.fits` file, where each image goes in each HDU.
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

dataset_type = "multi"
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
        [sys.executable, "scripts/multi/simulator.py"],
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
`scripts/multi/simulator.py`), so a separate tracer is composed per waveband; the mass model is
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
__Visualizer__

During a multi-dataset model-fit (e.g. combining analyses with `af.AnalysisFactor` as in
`scripts/multi/modeling.py`), the `Visualizer` attached to the `Analysis` class outputs the
combined figures above automatically:

 - Before the fit begins, all datasets are output together via `subplot_imaging_dataset_list`.
 - During and after the fit, the maximum likelihood fit to every dataset is output together via
   `subplot_fit_combined`.

These appear in the fit's output folder under `image/` (e.g. `dataset_combined.png`,
`fit_combined.png`), alongside the per-dataset figures described in `scripts/imaging/plot.py`.

Which figures are output is controlled by `config/visualize/plots.yaml`, e.g. the
`dataset` -> `subplot_dataset` and `fit` -> `subplot_fit` entries.
"""

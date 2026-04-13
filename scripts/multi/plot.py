"""
Plots: Multi Figure Plotter
============================

This example shows how to plot the same figure from multiple datasets on a single subplot.

The specific example loads multi-wavelength imaging datasets and plots the data image from
each dataset side-by-side.

In the old API, this was done using a `MultiFigurePlotter` object with a list of `Imaging`
objects. Both `MultiFigurePlotter` and `Imaging` have been removed.

In the new API, we load each dataset and use matplotlib subplots directly.

The dedicated `aplt.subplot_imaging_dataset()` function is also shown for single-dataset plots.

__Contents__

**Dataset:** Load and plot the strong lens dataset.
**Single Dataset Subplots:** Plot the full subplot overview of each dataset using `aplt.subplot_imaging_dataset()`.
**Multi Dataset Plot:** Plot the data image from each dataset side-by-side on the same matplotlib figure.
**Multi Dataset Array Plot:** We can also call `aplt.plot_array()` for each dataset separately.
**Multi Fits:** We can also output a list of figures to a single `.fits` file, where each image goes in each HDU.
**Wrap Up:** Summary of the script and next steps.

__Start Here Notebook__

If any code in this script is unclear, refer to `plot/start_here.ipynb`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
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
if not dataset_path.exists():
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

Plot the full subplot overview of each dataset using `aplt.subplot_imaging_dataset()`.
"""
for dataset in dataset_list:
    aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Multi Dataset Plot__

Plot the data image from each dataset side-by-side on the same matplotlib figure.
"""
fig, axes = plt.subplots(1, len(dataset_list), figsize=(12, 5))

for ax, dataset, waveband in zip(axes, dataset_list, waveband_list):
    im = ax.imshow(dataset.data.native, origin="upper", cmap="gray")
    ax.set_title(f"Data ({waveband}-band)", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Multi-Wavelength Data", fontsize=14)
plt.tight_layout()
plt.show()
plt.close()

"""
__Multi Dataset Array Plot__

We can also call `aplt.plot_array()` for each dataset separately.
"""
for dataset, waveband in zip(dataset_list, waveband_list):
    aplt.plot_array(array=dataset.data, title=f"Data ({waveband}-band)")

"""
__Multi Fits__

We can also output a list of figures to a single `.fits` file, where each image goes in
each HDU extension.
"""
from autoconf.fitsable import hdu_list_for_output_from

dataset = dataset_list[-1]

image_list = [dataset.data, dataset.noise_map]

hdu_list = hdu_list_for_output_from(
    values_list=[image_list[0].mask.astype("float")] + image_list,
    ext_name_list=["mask"] + ["data", "noise_map"],
    header_dict=dataset.mask.header_dict,
)

hdu_list.writeto("dataset.fits", overwrite=True)

"""
__Wrap Up__

The new API uses direct `aplt.plot_array()` calls and matplotlib subplots for combining
multiple figures from different datasets or objects.
"""

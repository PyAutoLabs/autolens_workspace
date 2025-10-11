"""
Modeling: Mass Total + Source Parametric
========================================

In this script, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `Sersic`.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files, which we will fit with the lens model.
"""
dataset_name = "slacs1430+4105"
dataset_path = path.join("dataset", "slacs", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.05,
)

dataset.data = dataset.data.resized_from(new_shape=(151, 151))
dataset.noise_map = dataset.noise_map.resized_from(new_shape=(151, 151))
dataset.psf = dataset.psf.resized_from(new_shape=(11, 11))

dataset.data.output_to_fits(
    file_path=path.join(dataset_path, "data.fits"), overwrite=True
)
dataset.noise_map.output_to_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), overwrite=True
)
dataset.psf.output_to_fits(
    file_path=path.join(dataset_path, "psf.fits"), overwrite=True
)

# dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
# dataset_plotter.subplot_dataset()
#
#
# np.save(file="image.npy", arr=dataset.data.native)
# np.save(file="noise_map.npy", arr=dataset.noise_map.native)
# np.save(file="psf.npy", arr=dataset.psf.native)

"""
Overview: Interferometer
------------------------

**PyAutoLens** supports modeling of interferometer data from submillimeter and radio observatories such as ALMA
or LOFAR.

The visibilities of the interferometer dataset are fitted directly in the uv-plane, circumventing issues that arise
when fitting a dirty image produced via the visibilities.

The most important issue this addresses is removing correlated noise from impacting the fit.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autolens as al
import autolens.plot as aplt
from os import path
import numpy as np

"""
__Real Space Mask__

To begin, we define a real-space mask. Although interferometer lens modeling is performed in the uv-plane and 
therefore Fourier space, we still need to define the grid of coordinates in real-space from which the lensed source's 
images are computed. It is this image that is mapped to Fourier space to compare to the uv-plane data.

The size and resolution of this mask depend on the baselines of your interferometer dataset. datasets with longer
baselines (i.e. higher resolution data) require higher resolution and larger masks.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(400, 400), pixel_scales=0.025, radius=3.0
)

"""
__Dataset__

We next load an interferometer dataset from fits files, which follows the same API that we have seen for an `Imaging`
object.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
The **PyAutoLens** plot module has tools for plotting interferometer datasets, including the visibilities, noise-map
and uv wavelength which represent the interferometer`s baselines. 

The data used in this tutorial contains only ~300 visibilities and is representative of a low resolution 
Square-Mile Array (SMA) dataset. 

We made this choice so the script runs fast, and we discuss below how **PyAutoLens** can scale up to large visibilities
datasets from an instrument like ALMA.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True, uv_wavelengths=True)

"""
It can also plot dirty images of the dataset in real-space, using the fast Fourier transforms described below.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(dirty_image=True, dirty_signal_to_noise_map=True)

"""
__Tracer__

To perform uv-plane modeling, **PyAutoLens** generates an image of the strong lens system in real-space via a tracer.

Lets quickly set up the `Tracer` we'll use in this example.
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
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=real_space_mask.derive_grid.unmasked_sub_1
)
tracer_plotter.figures_2d(image=True)

"""
__UV-Plane__

To perform uv-plane modeling, **PyAutoLens** next Fourier transforms this image from real-space to the uv-plane.

This operation uses a `Transformer` object, of which there are multiple available in **PyAutoLens**. This includes
a direct Fourier transform which performs the exact Fourier transform without approximation.
"""
transformer_class = al.TransformerDFT

"""
However, the direct Fourier transform is inefficient. For ~10 million visibilities, it requires thousands of seconds
to perform a single transform. This approach is therefore unfeasible for high quality ALMA and radio datasets.

For this reason, **PyAutoLens** supports the non-uniform fast fourier transform algorithm
**PyNUFFT** (https://github.com/jyhmiinlin/pynufft), which is significantly faster, being able to perform a Fourier
transform of ~10 million in less than a second!
"""
transformer_class = al.TransformerNUFFT

"""
To use this transformer in a fit, we use the `apply_settings` method.
"""
dataset = dataset.apply_settings(
    settings=al.SettingsInterferometer(transformer_class=transformer_class)
)

"""
__Fitting__

The interferometer can now be used with a `FitInterferometer` object to fit it to a dataset:
"""
fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)

"""
Visualization of the fit is provided both in the uv-plane and in real-space. 

Note that the fit is not performed in real-space, but plotting it in real-space is often more informative.
"""
fit_plotter.subplot_fit()

"""
__Pixelized Sources__

Interferometer data can also be modeled using pixelized source's, which again performs the source reconstruction by
directly fitting the visibilities in the uv-plane.
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(
    dataset=dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_linear_operators=True),
)

"""
The source reconstruction is visualized in real space:
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
__Efficiency__

Computing this source reconstruction would be extremely inefficient if **PyAutoLens** used a traditional approach to
linear algebra which explicitly stored in memory the values required to solve for the source fluxes. In fact, for an
interferometer dataset of ~10 million visibilities this would require **hundreds of GB of memory**!

**PyAutoLens** uses the library **PyLops** (https://pylops.readthedocs.io/en/latest/) to represent this calculation as
a sequence of memory-light linear operators.

The combination of **PyNUFFT** and **PyLops** makes the analysis of ~10 million visibilities from observatories such as
ALMA and JVLA feasible in **PyAutoLens**.
"
__Lens Modeling__

It is straight forward to fit a lens model to an interferometer dataset, using the same API that we saw for imaging
data.

We first compose the model, omitted the lens light components given that most strong lenses observed at submm /
radio wavelengths do not have visible lens galaxy emission.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

disk = af.Model(al.lp.Exponential)

source = af.Model(al.Galaxy, redshift=1.0, disk=disk)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Non-linear Search__

We again choose the non-linear search `Nautilus` (https://github.com/joshspeagle/Nautilus).
"""
search = af.Nautilus(path_prefix="overview", name="interferometer")

"""
__Analysis__

Whereas we previously used an `AnalysisImaging` object, we instead use an `AnalysisInterferometer` object which fits 
the lens model in the correct way for an interferometer dataset. 

This includes mapping the lens model from real-space to the uv-plane via the Fourier transform discussed above.
"""
analysis = al.AnalysisInterferometer(dataset=dataset)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

The results can be found in the `output/overview_interferometer` folder in the `autolens_workspace`.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The **PyAutoLens** visualization library and `FitInterferometer` object includes specific methods for plotting the 
results, for example the maximum log likelihood fit:
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Simulation__

Simulated interferometer datasets can be generated using the ``SimulatorInterferometer`` object, which includes adding
Gaussian noise to the visibilities:
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=dataset.uv_wavelengths, exposure_time=300.0, noise_sigma=0.01
)

real_space_grid = al.Grid2D.uniform(
    shape_native=real_space_mask.shape_native,
    pixel_scales=real_space_mask.pixel_scales,
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=real_space_grid)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Wrap Up__

The `interferometer` package of the `autolens_workspace` contains numerous example scripts for performing 
interferometer modeling and simulating strong lens interferometer datasets.
"""

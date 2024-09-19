"""
Modeling: Mass Total + Source Parametric
========================================

This script fits an `Interferometer` and `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is an `Sersic` (but is invisible in the interferometer data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric `SersicCore`.

__Benefits__

 A number of benefits are apparently if we combine the analysis of both datasets at both wavelengths:

 - The lens galaxy is invisible at sub-mm wavelengths, making it straight-forward to infer a lens mass model by
 fitting the source at submm wavelengths.

 - The source galaxy appears completely different in the g-band and at sub-millimeter wavelengths, providing a lot
 more information with which to constrain the lens galaxy mass model.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Interferometer Masking__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0
)

"""
__Interferometer Dataset__

Load and plot the strong lens `Interferometer` dataset `simple__no_lens_light` from .fits files, which we will fit 
with the lens model.
"""
dataset_type = "multi"
dataset_label = "interferometer"
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

interferometer = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

interferometer_plotter = aplt.InterferometerPlotter(dataset=interferometer)
interferometer_plotter.subplot_dataset()
interferometer_plotter.subplot_dirty_images()

"""
__Imaging Dataset__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files, which we will fit with the lens model.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "lens_sersic"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

imaging = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "g_data.fits"),
    psf_path=path.join(dataset_path, "g_psf.fits"),
    noise_map_path=path.join(dataset_path, "g_noise_map.fits"),
    pixel_scales=0.08,
)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Imaging Masking__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 - An `Sersic` `LightProfile` for the source galaxy's light, which is complete different for each 
 waveband. [14 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=21.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp_linear.Sersic,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__

We create analysis objects for both datasets.
"""
analysis_imaging = al.AnalysisImaging(dataset=imaging)
analysis_interferometer = al.AnalysisInterferometer(dataset=interferometer)

"""
Sum the analyses to create an overall analysis object, which sums the `log_likelihood_function` of each dataset
and returns the overall likelihood of the model fit to the dataset.
"""
analysis = analysis_imaging + analysis_interferometer

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation is performed on a 
different CPU.
"""
analysis.n_cores = 1

"""
Imaging and interferometer datasets observe completely different properties of the lens and source galaxy, where:

 - The lens galaxy is invisible at sub-mm wavelengths, meaning the lens light model should have zero `intensity`
 for the interferometer data fit.
 
 - The source galaxy appears completely different in the imaging data (e.g. optical emission) and sub-millimeter 
 wavelengths, meaning a completely different source model should be used for each dataset.

We therefore fix the lens galaxy intensity in the interferometer fit to zero and make every source parameter a free 
parameter across the two analysis objects.
"""
analysis = analysis.with_free_parameters(model.galaxies.source)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("multi", "modeling"),
    name="imaging_and_interferometer",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The lens model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitInterferometer` objects.
  - Information on the posterior as estimated by the `Nautilus` non-linear search.
"""
print(result_list[0].max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result_list[0].max_log_likelihood_tracer,
    grid=real_space_mask.derive_grid.unmasked,
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result_list[0].max_log_likelihood_fit)
fit_plotter.subplot_fit()

fit_plotter = aplt.FitInterferometerPlotter(fit=result_list[1].max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

plotter = aplt.NestPlotter(samples=result_list.samples)
plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""

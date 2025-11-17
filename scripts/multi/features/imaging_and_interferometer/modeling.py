"""
Modeling: Mass Total + Source Parametric
========================================

This script fits an `Interferometer` and `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is an MGE (but is invisible in the interferometer data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an MGE.

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

from pathlib import Path
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
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

interferometer = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
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
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

imaging = al.Imaging.from_fits(
    data_path=Path(dataset_path, "g_data.fits"),
    psf_path=Path(dataset_path, "g_psf.fits"),
    noise_map_path=Path(dataset_path, "g_noise_map.fits"),
    pixel_scales=0.08,
)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Imaging Masking__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    radius=mask_radius,
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Analysis__

We create analysis objects for both datasets.
"""
analysis_imaging = al.AnalysisImaging(dataset=imaging)
analysis_interferometer = al.AnalysisInterferometer(dataset=interferometer)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 -  with 1 x 20 Gaussians for the source galaxy's light, which is complete different for each waveband. [8 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=21.
"""
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
We now combine them using the factor analysis class, which allows us to fit the two datasets simultaneously.

Imaging and interferometer datasets observe completely different properties of the, such that the galaxy appears 
completely different in the imaging data (e.g. optical emission) and sub-millimeter wavelengths, meaning a completely 
different source model should be used for each dataset.

For this reason, we move lens light and source model composition to the `AnalysisFactor` class, which allows us to fit 
the two datasets simultaneously but with different models.

The benefit of fitting them simultaneously is that the mass model is inferred from both datasets.
"""
analysis_factor_list = []

for analysis in [analysis_imaging, analysis_interferometer]:

    model_analysis = model.copy()

    model_analysis.galaxies.lens.bulge = af.Model(al.lp_linear.Sersic)

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    model_analysis.galaxies.source.bulge = bulge

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
The `info` of the model shows us there are two models, one for the imaging dataset and one for the interferometer
dataset. 
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=Path("multi", "modeling"),
    name="imaging_and_interferometer",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

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
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results in **PyAutoLens**.
"""

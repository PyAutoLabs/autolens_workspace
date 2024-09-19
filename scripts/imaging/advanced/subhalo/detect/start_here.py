"""
Subhalo Detection: Start Here
=============================

Strong gravitational lenses can be used to detect the presence of small-scale dark matter (DM) subhalos. This occurs
when the DM subhalo overlaps the lensed source emission, and therefore gravitationally perturbs the observed image of 
the lensed source galaxy. 

When a DM subhalo is not included in the lens model, residuals will be present in the fit to the data in the lensed
source regions near the subhalo. By adding a DM subhalo to the lens model, these residuals can be reduced. Bayesian
model comparison can then be used to quantify whether or not the improvement to the fit is significant enough to
claim the detection of a DM subhalo.

The example illustrates DM subhalo detection with **PyAutoLens**.

__SLaM Pipelines__

The Source, (lens) Light and Mass (SLaM) pipelines are advanced lens modeling pipelines which automate the fitting
of complex lens models. The SLaM pipelines are used for all DM subhalo detection analyses in **PyAutoLens**. Therefore
you should be familiar with the SLaM pipelines before performing DM subhalo detection yourself. If you are unfamiliar
with the SLaM pipelines, checkout the 
example `autolens_workspace/notebooks/imaging/advanced/chaining/slam/start_here.ipynb`.

Dark matter subhalo detection runs the standard SLaM pipelines, and then extends them with a SUBHALO PIPELINE which
performs the following three chained non-linear searches:

 1) Fits the lens model fitted in the MASS PIPELINE again, without a DM subhalo, to estimate the Bayesian evidence
    of the model without a DM subhalo.

 2) Performs a grid-search of non-linear searches, where each grid cell includes a DM subhalo whose (y,x) centre is 
    confined to a small 2D section of the image plane via uniform priors (we explain this in more detail below).
    
 3) Fit the lens model again, including a DM subhalo whose (y,x) centre is initialized from the highest log evidence
    grid cell of the grid-search. The Bayesian evidence estimated in this model-fit is compared to the model-fit
    which did not include a DM subhalo, to determine whether or not a DM subhalo was detected.

__Grid Search__

The second stage of the SUBHALO PIPELINE uses a grid-search of non-linear searches to determine the highest log
evidence model with a DM subhalo. This grid search confines each DM subhalo in the lens model to a small 2D section
of the image plane via priors on its (y,x) centre. The reasons for this are as follows:

 - Lens models including a DM subhalo often have a multi-model parameter space. This means there are multiple lens 
   models with high likelihood solutions, each of which place the DM subhalo in different (y,x) image-plane location. 
   Multi-modal parameter spaces are synonomously difficult for non-linear searches to fit, and often produce 
   incorrect or inefficient fitting. The grid search breaks the multi-modal parameter space into many single-peaked 
   parameter spaces, making the model-fitting faster and more reliable.

 - By inferring how placing a DM subhalo at different locations in the image-plane changes the Bayesian evidence, we
   map out spatial information on where a DM subhalo is detected. This can help our interpretation of the DM subhalo
   detection.

__Pixelized Source__

Detecting a DM subhalo requires the lens model to be sufficiently accurate that the residuals of the source's light
are at a level where the subhalo's perturbing lensing effects can be detected. 

This requires the source reconstruction to be performed using a pixelized source, as this provides a more detailed 
reconstruction of the source's light than fits using light profiles.

This example therefore using a pixelized source and the corresponding SLaM pipelines.

The `subhalo/detection/examples` folder contains an example using light profile sources, if you have a use-case where
using light profile source is feasible (e.g. fitting simple simulated datasets).

__Model__

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS TOTAL PIPELINE and SUBHALO PIPELINE this SLaM script
fits `Imaging` of a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge with a linear parametric `Sersic` light profile.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "dark_matter_subhalo"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=path.join("subhalo_detect"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=1,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0


"""
__SOURCE LP PIPELINE__

This is the standard SOURCE LP PIPELINE described in the `slam/start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(dataset=dataset)

bulge = af.Model(al.lp_linear.Sersic)

source_lp_result = slam.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp_linear.SersicCore),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE PIX PIPELINE__

This is the standard SOURCE PIX PIPELINE described in the `slam/start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_lp_result),
    positions_likelihood=source_lp_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

source_pix_result_1 = slam.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay,
)

adapt_image_maker = al.AdaptImageMaker(result=source_pix_result_1)
adapt_image = adapt_image_maker.adapt_images.galaxy_name_image_dict[
    "('galaxies', 'source')"
]

over_sampling = al.OverSamplingUniform.from_adapt(
    data=adapt_image,
    noise_map=dataset.noise_map,
)

dataset = dataset.apply_over_sampling(
    over_sampling=al.OverSamplingDataset(pixelization=over_sampling)
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

source_pix_result_2 = slam.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
)

"""
__LIGHT LP PIPELINE__

This is the standard LIGHT LP PIPELINE described in the `slam/start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1)
)

bulge = af.Model(al.lp_linear.Sersic)

light_result = slam.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE__

This is the standard MASS TOTAL PIPELINE described in the `slam/start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
    positions_likelihood=source_pix_result_2.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

mass_result = slam.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
)

"""
__SUBHALO PIPELINE (single plane detection)__

The SUBHALO PIPELINE (single plane detection) consists of the following searches:
 
 1) Refit the lens and source model, to refine the model evidence for comparing to the models fitted which include a 
 subhalo. This uses the same model as fitted in the MASS TOTAL PIPELINE. 
 2) Performs a grid-search of non-linear searches to attempt to detect a dark matter subhalo. 
 3) If there is a successful detection a final search is performed to refine its parameters.
 
For this modeling script the SUBHALO PIPELINE customizes:

 - The [number_of_steps x number_of_steps] size of the grid-search, as well as the dimensions it spans in arc-seconds.
 - The `number_of_cores` used for the gridsearch, where `number_of_cores > 1` performs the model-fits in parallel using
 the Python multiprocessing module.
 
A full description of the SUBHALO PIPELINE can be found in the SLaM pipeline itself, which is located at 
`autolens_workspace/slam/subhalo/detection.py`. You should read this now.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=mass_result.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2, use_resample=True
    ),
    adapt_image_maker=al.AdaptImageMaker(result=source_pix_result_1),
)

result_no_subhalo = slam.subhalo.detection.run_1_no_subhalo(
    settings_search=settings_search,
    analysis=analysis,
    mass_result=mass_result,
)

result_subhalo_grid_search = slam.subhalo.detection.run_2_grid_search(
    settings_search=settings_search,
    analysis=analysis,
    mass_result=mass_result,
    subhalo_result_1=result_no_subhalo,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
)

result_with_subhalo = slam.subhalo.detection.run_3_subhalo(
    settings_search=settings_search,
    analysis=analysis,
    subhalo_result_1=result_no_subhalo,
    subhalo_grid_search_result_2=result_subhalo_grid_search,
    subhalo_mass=af.Model(al.mp.NFWMCRLudlowSph),
)

"""
__Bayesian Evidence__

To determine if a DM subhalo was detected by the pipeline, we can compare the log of the  Bayesian evidences of the 
model-fits performed with and without a subhalo. 

The following scale describes how different log evidence increases correspond to difference detection significances:

 - Negative log evidence increase: No detection.
 - Log evidence increase between 0 and 3: No detection.
 - Log evidence increase between 3 and 5: Weak evidence, should consider it a non-detection.
 - Log evidence increase between 5 and 10: Medium evidence, but still inconclusive.
 - Log evidence increase between 10 and 20: Strong evidence, consider it a detection.
 - Log evidence increase > 20: Very strong evidence, definitive detection.

Lets inspect the log evidence ncrease for the model-fit performed in this example:
"""
evidence_no_subhalo = result_no_subhalo.samples.log_evidence
evidence_with_subhalo = result_with_subhalo.samples.log_evidence

log_evidence_increase = evidence_with_subhalo - evidence_no_subhalo

print("Evidence Increase: ", log_evidence_increase)

"""
__Log Likelihood__

Different metrics can be used to inspect whether a DM subhalo was detected.

The Bayesian evidence is the most rigorous because it penalizes models based on their complexity. If a model is more
complex (e.g. it can fit the data in more ways) than another model, the evidence will decrease. The Bayesian evidence
therefore favours simpler models over more complex models, unless the more complex model provides a much better fit to
the data. This is called the Occam's Razor principle.

An alternative goodness of fit is the `log_likelihood`. This is directly related to the residuals of the model or
the chi-squared value. The log likelihood does not change when a model is made more or less complex, and as such it 
will nearly always favour the more complex model because this must provide a better fit to the data one way or another.

The benefit of the log likelihood is it is a straight forward value indicating how well a model fitted the data. It
can provide useful sanity checks, for example the `log_likelihood` of the lens model without a subhalo must always be
less than the model with a subhalo (because the model with a subhalo can simple reduce its mass and recover the model
without a subhalo). If this is not the case, something must have gone wrong with one of the model-fits, for example
the non-linear search failed to find the highest likelihood regions of parameter space.
"""
log_likelihood_no_subhalo = result_no_subhalo.samples.log_likelihood
log_likelihood_with_subhalo = result_with_subhalo.samples.log_likelihood

log_likelihood_increase = log_likelihood_with_subhalo - log_likelihood_no_subhalo

print("Log Likelihood Increase: ", log_likelihood_increase)

"""
__Visualization__

There are DM subhalo specific visualization tools which can be used to inspect the results of DM subhalo detection.

The `SubhaloPlotter` takes as input `FitImaging` objects of the no subhalo and with subhalo model-fits, which will
allow us to plot their images alongside one another and therefore inspect how the residuals change when a DM
subhalo is included in the model.

We also input the `result_subhalo_grid_search`, which we will use below to show visualization of the grid search.
"""
subhalo_plotter = al.subhalo.SubhaloPlotter(
    fit_imaging_with_subhalo=result_with_subhalo.max_log_likelihood_fit,
    fit_imaging_no_subhalo=result_no_subhalo.max_log_likelihood_fit,
)

"""
The following subplot compares the fits with and without a DM subhalo.

It plots the normalized residuals, chi-squared map and source reconstructions of both fits alongside one another.
The improvement to the fit that including a subhalo provides is therefore clearly visible.
"""
subhalo_plotter.subplot_detection_fits()

"""
__Grid Search Result__

The grid search results have attributes which can be used to inspect the results of the DM subhalo grid-search.

For example, we can produce a 2D array of the log evidence values computed for each grid cell of the grid-search.

We compute these values relative to the `log_evidence` of the model-fit which did not include a subhalo, such that
positive values indicate that including a subhalo increases the Bayesian evidence.
"""
result_subhalo_grid_search = al.subhalo.SubhaloGridSearchResult(
    result=result_subhalo_grid_search
)

log_evidence_array = result_subhalo_grid_search.figure_of_merit_array(
    use_log_evidences=True,
    relative_to_value=result_no_subhalo.samples.log_evidence,
)

print("Log Evidence Array: \n")
print(log_evidence_array)

"""
We can plot this array to get a visualiuzation of where including a subhalo in the model increases the Bayesian
evidence.
"""
plotter = aplt.Array2DPlotter(
    array=log_evidence_array,
)
plotter.figure_2d()

"""
The grid search result also contained arrays with the inferred masses for each grid cell fit and the inferred
DM subhalo centres.
"""
mass_array = result_subhalo_grid_search.subhalo_mass_array

print("Mass Array: \n")
print(mass_array)

subhalo_centres_grid = result_subhalo_grid_search.subhalo_centres_grid

print("Subhalo Centres Grid: \n")
print(subhalo_centres_grid)

"""
An array with the inferred parameters for any lens model parameter can be computed as follows:
"""
einstein_radius_array = result_subhalo_grid_search.attribute_grid(
    "galaxies.lens.mass.einstein_radius"
)

"""
__Grid Search Visualization__

The `SubhaloPlotter` can also plot the results of the grid search, which provides spatial information on where in
the image plane including a DM subhalo provides a better fit to the data.

The plot below shows the increase in `log_evidence` of each grid cell, laid over an image of the lensed source
so one can easily see which source features produce a DM subhalo detection.

The input `remove_zeros` removes all grid-cells which have a log evidence value below zero, which provides more
clarity in the figure where including a DM subhalo makes a difference to the Bayesian evidence.
"""
subhalo_plotter = al.subhalo.SubhaloPlotter(
    result=result_subhalo_grid_search,
    fit_imaging_with_subhalo=result_with_subhalo.max_log_likelihood_fit,
    fit_imaging_no_subhalo=result_no_subhalo.max_log_likelihood_fit,
)

subhalo_plotter.figure_figures_of_merit_grid(
    use_log_evidences=True,
    relative_to_value=result_no_subhalo.samples.log_evidence,
    remove_zeros=True,
)

"""
A grid of inferred DM subhalo masses can be overlaid instead:
"""
subhalo_plotter.figure_mass_grid()

"""
A subplot of all these quantities can be plotted:
"""
subhalo_plotter.subplot_detection_imaging(
    use_log_evidences=True,
    relative_to_value=result_no_subhalo.samples.log_evidence,
    remove_zeros=True,
)

"""
Finish.
"""

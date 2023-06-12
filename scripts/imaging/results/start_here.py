"""
Results: Start Here
===================

This script is the starting point for investigating the results of lens modeling and it provides
an overview of the lens modeling API.

After reading this script, the `examples` folder provides more detailed examples for analysing the different aspects of
performing  modeling results outlined here.

__Model__

We begin by fitting a quick lens model to a simple lens dataset, which we will use to illustrate the lens modeling
results API.

If you are not familiar with the lens modeling API and process, checkout the `autolens_workspace/examples/modeling`
folder for examples.
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

"""
__Model Fit__

The code below (which we have omitted comments from for brevity) performs a lens model-fit using dynesty. You should
be familiar enough with lens modeling to understand this, if not you should go over the beginner model-fit script again!
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Info__

As seen throughout the workspace, the `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
__Samples__

The result's `Samples` object contains the complete set of non-linear search dynesty samples, where each sample 
corresponds to a set of a model parameters that were evaluated and accepted. 

The examples script `autolens_workspace/*/imaging/results/examples/samples.py` provides a detailed description of 
this object, including:

 - Extracting the maximum likelihood lens model.
 - Using marginalized PDFs to estimate errors on the lens model parameters.
 - Deriving errors on derived quantities, such as the Einstein radius.

Below, is an example of how to use the `Samples` object to estimate the lens mass model parameters which are 
the median of the probability distribution function and its errors at 3 sigma confidence intervals.
"""
samples = result.samples

median_pdf_instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(median_pdf_instance.galaxies.lens.mass)
print()

ue3_instance = samples.values_at_upper_sigma(sigma=3.0)
le3_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(ue3_instance.galaxies.lens.mass, "\n")
print(le3_instance.galaxies.lens.mass, "\n")

"""
__Tracer__

The result's maximum likelihood `Tracer` object contains everything necessary to perform ray-tracing and other
calculations with the lens model.

The examples script `autolens_workspace/*/imaging/results/examples/tracer.py` provides a detailed 
description of this object, including:

 - Producing individual images of the lens and source galaxies in the image and source plane.
 - Inspecting mass model components like the convergence, potential and deflection angles.
 - Other lensing quantities like the critical curve and caustics.

Below, is an example of how to use the `Tracer` object to calculate the image of the lens and source galaxies.
"""
tracer = result.max_log_likelihood_tracer

image = tracer.image_2d_from(grid=dataset.grid)

"""
__Fits__

The result's maximum likelihood `FitImaging` object contains everything necessary to inspect the lens model fit to the 
data.

The examples script `autolens_workspace/*/imaging/results/examples/fits.py` provides a detailed description of this 
object, including:

 - How to inspect the residuals, chi-squared, likelihood and other quantities.
 - Outputting resulting images (e.g. the source reconstruction) to hard-disk.
 - Refitting the data with other lens models from the `Samples` object, to investigate how sensitive the fit is to
   different lens models.

Below, is an example of how to use the `FitImaging` object to output the source reconstruction to print the 
chi-squared and log likelihood values.
"""
fit = result.max_log_likelihood_fit

print(fit.chi_squared)
print(fit.log_likelihood)

"""
__Galaxies__

The result's maximum likelihood `Galaxy` objects contain everything necessary to inspect the individual properties of
the lens and source galaxies.

The examples script `autolens_workspace/*/imaging/results/examples/galaxies.py` provides a detailed description 
of this object, including:

 - How to plot individual galaxy images, such as the source galaxy's image-plane and source-plane images.
 - Plotting the individual light profiles and mass profiles of the galaxies.
 - Making one dimensional profiles of the galaxies, such as their light and mass profiles as a function of radius.
 
Below, is an example of how to use the `Galaxy` objects to plot the source galaxy's source-plane image.
"""
tracer = result.max_log_likelihood_tracer

source = tracer.planes[1].galaxies[0]
galaxy_plotter = aplt.GalaxyPlotter(galaxy=source, grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Cosmological Quantities__

The maximum likelihood model includes cosmological quantities, which can be computed via the result.

The examples script `autolens_workspace/*/imaging/results/examples/cosmological_quantities.py` provides a detailed 
description of this object, including:

 - Calculating the Einstein radius of the lens galaxy.
 - Converting quantities like the Einstein radius or effective radius from arcseconds to kiloparsecs.
 - Computing the Einstein mass of the lens galaxy in solar masses.
 
Below, is an example of how to convert the effective radius of the source galaxy from arcseconds to kiloparsecs.
"""
tracer = result.max_log_likelihood_tracer

cosmology = al.cosmo.Planck15()

source = tracer.planes[1].galaxies[0]
source_kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=source.redshift)
source_effective_radius_kpc = source.bulge.effective_radius * source_kpc_per_arcsec

"""
__Linear Light Profiles / Basis Objects__

A lens model can be fitted using a linear light profile, which is a light profile whose `intensity` parameter is 
sovled for via linear algebra.

This includes Basis objects such as a Multi-Gaussian expansion of Shapelets.

These objects mostly behave identically to ordinary light profiles, but due to the linear algebra have their own
specific functionality.

The example script `autolens_workspace/*/imaging/results/examples/linear.py` provides a detailed description of 
this functionality, including:

 - Extracting individual quantities from the linear light profile, such as the coefficients of the basis functions.
 - Extracting the intensity of the linear light profiles after they have been computed via linear algebra.
 - Plotting the linear light profiles.
 
The fit above did not use a pixelization, so we omit a example of the API below.

__Pixelization__

The lens model can reconstruct the source galaxy using a pixelization, for example on a Voronoi mesh.

The example script `autolens_workspace/*/imaging/results/examples/pixelizations.py` provides a detailed description 
of inspecting the results of a fit using a pixelization, including:

 - Producing source reconstructions using the Voronoi mesh, Delaunay triangulation or whichever mesh is used.
 - Inspecting the evidence terms of the fit, which quantify how well the pixelization reconstructs fits the data whilst
   accounting for the complexity of the pixelization.
 - Estimating the magnification of the source galaxy's image using the pixelization.

The fit above did not use a pixelization, so we omit a example of the API below.
"""

"""
Misc: Custom Analysis
=====================

Users familiar with **PyAutoLens** will have seen that `Analysis` classes are used to performed lens modeling of
different datasets. For example, the `AnalysisImaging` class fits imaging datasets, the `AnalysisInterferometer` class
fits interferometer datasets.

You may have a dataset which you want to use **PyAutoLenss**'s lensing capabilities to model, but which does not fit
into one of the standard `Analysis` classes.

A good example (at the time of writing this script) is fitting a weak lensing shear catalogue with a model of the lens
galaxy's mass. **PyAutoLens** as the lensing capabilities to produce the shears of a mass model, but does not have an
`Analysis` class to fit these shears to a dataset.

This example demonstrates how you can write your own `Analysis` class to fit a dataset with **PyAutoLens**.

__PyAutoFit__

The `Analysis` class is the interface between the data and model, whereby a `log_likelihood_function` is defined
and called by the non-linear search to fit the model.

You may have performed a similar task yourself, for example by taking a fitting library (e.g. an MCMC method like
Emcee or nested sampler like Dynesty) and writing a likelihood function that calls it to fit a model to a dataset.
If you haven't done this, this script will explain how!

**PyAutoLens** uses a library called **PyAutoFit** to set up this interfacebetween the data, model,
`log_likelihood_function` and non-linear search. **PyAutoFit** is a general purpose library for model fitting,
and we will see that it has a lot of powerful tools that we can use to customize our `Analysis` class.

You can checkout the **PyAutoFit** readthedocs here:

 https://pyautofit.readthedocs.io/en/latest/

The following analysis cookbook provides a concise reference guide to `Analysis` objects, and once you have completed
this example will be a useful resource for writing your own `Analysis` class:

 https://pyautofit.readthedocs.io/en/latest/cookbooks/analysis.html

__Source Code__

This example contains URLs to the locations of the source code of the classes used when creating light and mass
profiles.

The example itself is standalone and should by the end allow you to implement a custom profile without diving into
the **PyAutoLens** source code.

We still recommend you take a look to see how things are structured!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Example Analysis Class__

The `Analysis` classes available in **PyAutoLens** are actually located in both **PyAutoLens** and its parent 
package, **PyAutoGalaxy**: 

 https://github.com/Jammy2211/PyAutoGalaxy
 https://github.com/Jammy2211/PyAutoLens

All classes used for lens modeling are found in the following packages:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/imaging/model
 https://github.com/Jammy2211/PyAutoLens/tree/main/autolens/imaging/model

The `AnalysisImaging` classes are found in the following modules:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/imaging/model/analysis.py
 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/imaging/model/analysis.py

__Lens Model__

To illustrate how to write a custom `Analysis` class, we require an example lens model that we will use to fit
the dataset.

We compose a simple lens model with an `IsothermalSph` mass model for the lens and an `Sersic` for the source.
"""
# Lens:
mass = af.Model(al.mp.IsothermalSph)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.ExponentialCoreSph)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Instances__

Instances of the model above can be created, where an input `vector` of parameters is mapped to create an instance of 
the Python class of the model.

This is used internally by the `Analysis` class we are about to write, and will be used in 
our `log_likelihood_function`. Therefore, we are quickly highlighting it here.

We first need to know the order of parameters in the model, so we know how to define the input `vector`. This
information is contained in the models `paths` attribute:
"""
print(model.paths)

"""
We input values for the parameters of our model following the order of paths above.

This creates an `instance` of the lens model.
"""
instance = model.instance_from_vector(vector=[0.0, 0.0, 1.6, 0.1, 0.1, 0.01, 2.0])

"""
This `instance` contains each of the model components we defined above. 

The argument names input into each `Collection` define the attribute names of the `instance`. 

For example, when composing the `model` above, we used a `Collection` called `galaxies` which had a `lens` and `source` 
attribute. These `lens` and `source` attributes each contained components called `mass` and `bulge` respectively.
"""
print(f"Lens Centre = {instance.galaxies.lens.mass.centre}")
print(f"Lens Einstein Radius = {instance.galaxies.lens.mass.einstein_radius}")
print(f"Source Centre = {instance.galaxies.source.bulge.centre}")
print(f"Source Intensity = {instance.galaxies.source.bulge.intensity}")
print(f"Source Effective Radius = {instance.galaxies.source.bulge.effective_radius}")

"""
__Simple Analysis Example__

For simplicity, a shortened version of an `AnalysisImaging` class is shown below where certain functions have been 
edited to make them easy to read and understand. 

This has docstrings updated to focus on the key aspects of implementing a new `Analysis` class and simplifies the 
inheritance structure of the profile.
"""


class AnalysisImaging(af.Analysis):
    def __init__(
        self,
        dataset: al.Imaging,
        cosmology: al.cosmo.LensingCosmology = al.cosmo.Planck15(),
    ):
        """
        Fits a lens model to an imaging dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit strong lenses composed via a `Tracer` to an imaging dataset.
        Parameters
        ----------
        dataset
            The `Imaging` dataset that the model is fitted to.
        cosmology
            The Cosmology assumed for this analysis.
        """
        self.dataset = dataset
        self.cosmology = cosmology

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extracts all galaxies from the model instance and set up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        2) Use the `Tracer` and other attributes to create a `FitImaging` object, which performs steps such as creating
           model images of every galaxy in the tracer, blurring them with the imaging dataset's PSF and computing
           residuals, a chi-squared statistic and the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """

        """
        The `instance` that comes into this method is an instance of the lens model above, which we illustrated
        via print statements how it is structured.

        The parameter values are chosen by the non-linear search, based on where it thinks the high likelihood regions 
        of parameter space are.

        The lines of Python code are commented out below to prevent excessive print statements when we run the
        non-linear search, but feel free to uncomment them and run the search to see the parameters of every instance
        that it fits.
        """

        # print(f"Lens Centre = {instance.galaxies.lens.mass.centre}")
        # print(f"Lens Einstein Radius = {instance.galaxies.lens.mass.einstein_radius}")
        # print(f"Source Centre = {instance.galaxies.source.bulge.centre}")
        # print(f"Source Intensity = {instance.galaxies.source.bulge.intensity}")
        # print(f"Source Effective Radius = {instance.galaxies.source.bulge.effective_radius}")

        """
        You should be familiar with the `Tracer` object, given a list of galaxies it provides all the functionality
        necessary to perform ray-tracing and strong lensing calculations.
        
        One aspect of its design you may not have considered is that the input galaxies can be any size, and it does
        not matter what the galaxies, light or mass profiles are called (e.g. it does not depend on the lens mass
        have the path `galaxies.lens.mass`).
        
        This means that a user can compose a lens model using any combination of light and mass profiles, and the
        `log_likelihood_function` below will still work. You should ensure your `Analysis` class is written generically
        like this.        
        """

        tracer = al.Tracer(
            galaxies=instance.galaxies,
            cosmology=self.cosmology,
        )

        """
        You should also be familiar with the `FitImaging` object, which given a tracer and imaging dataset fits the
        tracer's model image to the data, using a chi-squared map to compute the residuals and likelihood.
        """

        fit = al.FitImaging(
            dataset=self.dataset,
            tracer=tracer,
        )

        """
        To get your custom analysis class, running quickly, you may not want to define your own `Fit` class but
        instead just write out manually how the `log_likelihood` is computed. 
        
        The commented out code below shows the simplest way to do this, and it is probably suitable for most 
        use-cases.
        
        At step-by-step description of what the code is doing is as follows:
        
         1) Creates an image of the lens and source galaxies from the tracer using its `image_2d_from()` method.

         2) Blurs the tracer`s image with the data's PSF, ensuring the telescope optics are included in the fit. This 
         creates what is called the `model_image`.
        
         3) Computes the difference between this model-image and the observed image, creating the fit`s `residual_map`.
        
         4) Divides the residual-map by the noise-map, creating the fit`s `normalized_residual_map`.
        
         5) Squares every value in the normalized residual-map, creating the fit's `chi_squared_map`.
        
         6) Sums up these chi-squared values and converts them to a `log_likelihood`, which quantifies how good 
         this tracer`s fit to the data was (higher log_likelihood = better fit).
         
        Quantities like the `chi_squared_map` and `log_likelihood` are standard quantities used by all model-fitting
        approaches.
        """
        # model_data = tracer.blurred_image_2d_from(
        #    grid=self.dataset.grid,
        #    convolver=self.dataset.convolver,
        #    blurring_grid=self.dataset.grids.blurring,
        # )

        # residual_map = self.dataset.data - model_data
        # chi_squared_map = (residual_map / self.dataset.noise_map) ** 2.0
        # chi_squared = sum(chi_squared_map)
        # noise_normalization = np.sum(np.log(2 * np.pi * self.dataset.noise_map**2.0))
        # log_likelihood = -0.5 * (chi_squared + noise_normalization)

        """
        The `log_likelihood` is returned to the non-linear search, informing it how good a fit this lens model
        was and whether to continue sampling this region of parameter space.
        """

        return fit.log_likelihood


"""
__Analysis Class Considerations__

Lets quickly think about the design of an `Analysis` class and how this can help us to set up any model-fit we can
imagine:

 - The `__init__` method can be extended to include any data structures needed to perform the analysis. For example, 
   the `AnalysisImaging` object in the autolens source code has a `settings_inversion` object that customize 
   how fits using a `Pixelization` are performed.
   
 - The `log_likelihood_function` can be written in any way that is desired to fit the data. The example above uses
   the `FitImaging` object, but this is not necessary. Furthermore, you could customize this function to assume a 
   likelihood function defined by Poisson statistics (the example above assumes Gaussian statistics) or to include
   additional constraints on the model that are specific to your dataset.

__Model Fit__

The standard API for choosing a non-linear search and performing a model-fit can now be used with this `Analysis`
class.
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

search = af.Nautilus(
    path_prefix=path.join("custom_analysis"),
    name="strong_lensing_example",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
    iterations_per_update=10000,
)

# We are using the Analysis class above here!

analysis = AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Weak Lensing Example__

Now lets consider how to write our own custom `Analysis` class, for the example of performing a weak lensing analysis.

If you are unfamiliar with weak lensing, a brief summary is as follows:

 - Weak lensing is the small lensing signal induced into galaxies by lensing due to large-scale structure in the
   universe. 
   
- This signal is much smaller than the strong lensing regime and is often summarized as the small change in the 
  ellipticity of a source galaxy's light. 
  
- This change in ellipticity can be measured and is called the `shear`, with the dataset our `Analysis` class will
  fit called a shear catalogue.

 - In strong lensing, we typically use the deflection angles of a mass profile to fit the data. For weak lensing
 analysis we compute its shear (via the function `shear_yx_2d_from`) and compare this to the observed shear in the
 shear catalogue data.

__Lens Model__

We first compose our lens model for weak lensing analysis.

This can reuse the **PyAutoLens** API for model composition, but does not require a source galaxy to be included as
we are simply comparing the mass model shears.`
"""
# Lens:
mass = af.Model(al.mp.IsothermalSph)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens))

"""
Here is our example `Analysis` class:
"""


class AnalysisShearCatalogue(af.Analysis):
    def __init__(
        self,
        data,  # You may wish to group these into a `ShearCatalogue` dataset.
        noise_map,
        grid,
        cosmology: al.cosmo.LensingCosmology = al.cosmo.Planck15(),
    ):
        """
        Fits a lens model to a shear catalogue dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit strong lenses composed via a `Tracer` to a weak lensing
        shear catalogue dataset.

        Parameters
        ----------
        data
            The shear catalogue data.
        noise_map
            An array describing the RMS standard deviation error in each shear measurement point (e.g. the noise-map).
        grid
            The (y,x) coordinates defining where the shears are measured and evaluated (e.g. the locations of the
            galaxies in the shear catalogue).
        cosmology
            The Cosmology assumed for this analysis.
        """
        self.data = data
        self.noise_map = noise_map
        self.grid = grid
        self.cosmology = cosmology

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extracts all galaxies from the model instance and set up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        2) Use the `Tracer` to compute the model shear field of the entire strong lensing system.

        3) Compute the shear residuals, a chi-squared statistic and the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """

        """
        For this example, its very easy to compute the model shear field as the `Tracer` object already has this
        functionality built in.   
        """
        tracer = al.Tracer(
            galaxies=instance.galaxies,
            cosmology=self.cosmology,
        )

        model_data = tracer.shear_yx_2d_via_hessian_from(grid=self.grid)

        """
        We then use this model data and the data itself to compute the residuals, chi-squared and log likelihood.
        """
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood


"""
__Model Fit__

The standard API for choosing a non-linear search and performing a model-fit can now be used with this `Analysis`
class.

NOTE: Felix can you send me an example shear catalogue so I can get this to run :)
"""
dataset_name = "example_shear_catalogue"
dataset_path = path.join("dataset", "weak_lensing", dataset_name)

# data = load_shear()
# noise_map = load_noise_map()
# grid = load_grid()

search = af.Nautilus(
    path_prefix=path.join("custom_analysis"),
    name="weak_lensing_example",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
    iterations_per_update=10000,
)

# We are using the Analysis class above here!

# analysis = AnalysisShearCatalogue(
#     data=data,
#     noise_map=noise_map,
#     grid=grid
# )

"""
If you are used to using **PyAutoLens**, you'll know that when we run the fit below lots of information about the
model fit is output to hard-disk (e.g. the best-fit model, error estimates, the model info).

By writing our own `Analysis` class, this is output for free without us having to do anything - pretty cool, huh?

Below, we'll show you how to customize the `Analysis` class even more, to output additional information to hard-disk
such as visualization and results which you can load elsewhere via the **PyAutoLens** database functionality.
"""
# result = search.fit(model=model, analysis=analysis)

"""
__Result__

If you're familiar with **PyAutoLens**'s API, you'll know that the `Result` object returned by the non-linear search
contains lots of information about the fit. 

This includes parameter estimates and errors, details of the non-linear search, etc. 

By writing our own `Analysis` class we get all of this information for free, without having to change our code!
Therefore you should be good to inspect and interpret the results as normal.

The results `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
The result contains the maximum log likelihood instance, which we can use to inspect the result or make plots.
"""
instance = result.instance

print(f"Lens Centre = {instance.galaxies.lens.mass.centre}")
print(f"Lens Einstein Radius = {instance.galaxies.lens.mass.einstein_radius}")

"""
It also contains information on the posterior as estimated by the non-linear search (in this example `Nautilus`). 

Below, we make a corner plot of the "Probability Density Function" of every parameter in the model-fit.
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

__To Do List__

The following `Analysis` cookbook from the **PyAutoFit** readthedocs should help you get started customizing your
own `Analysis` class: 

https://pyautofit.readthedocs.io/en/latest/cookbooks/analysis.html

I will extend this guide to include the following in the next few days:

 - How to output your own custom visualization.
 - How to extend the `Result` class to include additional information about the model-fit specific to weak lensing 
   (e.g. the maximum likelihood shear map).
 - Add methods which output model-specific results to hard-disk in the files folder (e.g. as .json files) to aid in 
 the interpretation of results.
 - How to output results to hard-disk in a format that can be loaded into the **PyAutoLens** database.
 
"""

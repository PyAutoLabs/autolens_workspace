"""
Tutorial 1:  Individual Models
==============================

The example scripts throughout the workspace have focused on fitting a lens model to one dataset. You will have
inspected the results of those individual model-fits and used them to estimate properties of the lens, like its
Einstein radius.

You may even have analysed a sample consisting of tens of objects, and combined the results to make more general
statements about galaxy formation or cosmology. That is, deduce  'global' trends of many models fits a lens sample.

These tutorials show you how to compose and fit graphical models to a large datasets. A graphical model fits many
individual models to each dataset in your sample, but it also links parameters in these models together to
enable global inference on the model over the full dataset.

To illustrate this, these tutorials will use graphical models to infer the slope across a sample of lenses.
Graphical models will be used to determine the global distribution from which the slope are drawn, which uses
specific type of graphical model called a hierarchical model.

The first two tutorials will begin by simplifying the problem. We are going to fit a sample of 3 lenses whose mass
profiles are `IsothermalSph` profiles which all have the same `slope` value. We can therefore consider the `slope`
the global parameter we seek to estimate.

The data that we fit is going to be low resolution, meaning that our estimate of each `slope` has large errors.
To estimate the global slope of the sample, this tutorial does not use graphical models, but instead estimates
the `slope` by fitting each dataset one-by-one and combining the results post model-fitting. This will act as a
point of comparison to tutorial 2, where we will fit for the slope using graphical models.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/simulators/imaging/samples/simple__no_lens_light.py`.

__PyAutoFit Tutorials__

**PyAutoFit** has dedicated tutorials describing graphical models, which users not familiar with graphical
modeling may benefit from reading -- https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_graphical_models.html.

__Realism__

For an realistic lens sample, one would not expect that each lens galaxy has the same value of `slope`, which is
assumed in tutorials 1, 2 and 3. We make this assumption here to simplify the problem and make it easier to
illustrate graphical models. Later tutorials fit more realistic graphical models where each lens galaxy has its own
value of slope!

One can easily imagine datasets where the shared parameter is the same across the full sample. For example, studies
where cosmological parameters (e.g. the Hubble constant, H0) are included in the graphical mode. The tools introduced
in tutorials 1 and 2 could therefore be used for many science cases!
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
__Dataset__

For each dataset in our sample we set up the correct path and load it by iterating over a for loop.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/simulators/imaging/samples/simple__no_lens_light.py`.  
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "simple__no_lens_light__mass_sis"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 3

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = path.join(dataset_path, f"dataset_{dataset_index}")

    dataset_list.append(
        al.Imaging.from_fits(
            data_path=path.join(dataset_sample_path, "data.fits"),
            psf_path=path.join(dataset_sample_path, "psf.fits"),
            noise_map_path=path.join(dataset_sample_path, "noise_map.fits"),
            pixel_scales=0.1,
        )
    )

"""
__Mask__

We now mask each lens in our dataset, using the imaging list we created above.

We will assume a 3.0" mask for every lens in the dataset is appropriate.
"""
masked_imaging_list = []

for dataset in dataset_list:
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    masked_imaging_list.append(dataset.apply_mask(mask=mask))

"""
__Paths__

The path the results of all model-fits are output:
"""
path_prefix = path.join("imaging", "graphical", "tutorial_1_individual_models")

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `PowerLawSph` with its centre fixed to its true value of 
 (0.0, 0.0) [2 parameter].
 
 - The source galaxy's light is a parametric `ExponentialSph` [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6.

To make graphical model fits run fast, the model above is simple compared to a lot of models fitted throughout the 
workspace (for example, both galaxies are spherical).
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawSph)
lens.mass.centre = (0.0, 0.0)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.ExponentialSph)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit__

For each dataset we now create a non-linear search, analysis and perform the model-fit using this model.

Results are output to a unique folder named using the `dataset_index`.
"""
result_list = []

for dataset_index, masked_dataset in enumerate(masked_imaging_list):
    dataset_name_with_index = f"dataset_{dataset_index}"
    path_prefix_with_index = path.join(path_prefix, dataset_name_with_index)

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="search__simple__no_lens_light",
        unique_tag=dataset_name_with_index,
        nlive=50,
    )

    analysis = al.AnalysisImaging(dataset=masked_dataset)

    result = search.fit(model=model, analysis=analysis)
    result_list.append(result)

"""
__Results__

In the `model.results` file of each fit, it will be clear that the `slope` value of every fit (and the other 
parameters) have much larger errors than other **PyAutoGalaxy** examples due to the low signal to noise of the data.

The `result_list` allows us to plot the median PDF value and 3.0 confidence intervals of the `slope` estimate from
the model-fit to each dataset.
"""
import matplotlib.pyplot as plt

samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf() for samps in samples_list]
ue3_instances = [samp.errors_at_upper_sigma(sigma=3.0) for samp in samples_list]
le3_instances = [samp.errors_at_lower_sigma(sigma=3.0) for samp in samples_list]

mp_slopees = [instance.galaxies.lens.mass.slope for instance in mp_instances]
ue3_slopees = [instance.galaxies.lens.mass.slope for instance in ue3_instances]
le3_slopees = [instance.galaxies.lens.mass.slope for instance in le3_instances]

print(mp_slopees)

plt.errorbar(
    x=["galaxy 1", "galaxy 2", "galaxy 3"],
    y=mp_slopees,
    marker=".",
    linestyle="",
    yerr=[le3_slopees, ue3_slopees],
)
plt.show()
plt.close()

"""
These model-fits are consistent with the input `slope` values of 2.0 (the isothermal profile used to simulate
them). 

We can show this by plotting the 1D and 2D PDF's of each model fit
"""

for samples in samples_list:
    search_plotter = aplt.DynestyPlotter(samples=samples)
    search_plotter.cornerplot()


"""
We can also print the values of each slope estimate, including their estimates at 3.0 sigma.

Note that above we used the samples to estimate the size of the errors on the parameters. Below, we use the samples to 
get the value of the parameter at these sigma confidence intervals.
"""
u1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
l1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

u1_slopees = [instance.galaxies.lens.mass.slope for instance in u1_instances]
l1_slopees = [instance.galaxies.lens.mass.slope for instance in l1_instances]

u3_instances = [samp.values_at_upper_sigma(sigma=3.0) for samp in samples_list]
l3_instances = [samp.values_at_lower_sigma(sigma=3.0) for samp in samples_list]

u3_slopees = [instance.galaxies.lens.mass.slope for instance in u3_instances]
l3_slopees = [instance.galaxies.lens.mass.slope for instance in l3_instances]

for index in range(total_datasets):
    print(f"slope estimate of galaxy dataset {index}:\n")
    print(
        f"{mp_slopees[index]} ({l1_slopees[index]} {u1_slopees[index]}) [1.0 sigma confidence interval]"
    )
    print(
        f"{mp_slopees[index]} ({l3_slopees[index]} {u3_slopees[index]}) [3.0 sigma confidence interval] \n"
    )


"""
__Estimating the slope__

So how might we estimate the global `slope`, that is the value of slope we know all 3 lenses were 
simulated using? 

A simple approach takes the weighted average of the value inferred by all fits above.
"""
ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_slopees = [instance.galaxies.lens.mass.slope for instance in ue1_instances]
le1_slopees = [instance.galaxies.lens.mass.slope for instance in le1_instances]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_slopees, le1_slopees)]

values = np.asarray(mp_slopees)
sigmas = np.asarray(error_list)

weights = 1 / sigmas**2.0
weight_averaged = np.sum(1.0 / sigmas**2)

weighted_slope = np.sum(values * weights) / np.sum(weights, axis=0)
weighted_error = 1.0 / np.sqrt(weight_averaged)

print(
    f"Weighted Average slope Estimate = {weighted_slope} ({weighted_error}) [1.0 sigma confidence intervals]"
)

"""
__Posterior Multiplication__

An alternative and more accurate way to combine each individual inferred slope is multiply their posteriors 
together.

In order to do this, a smooth 1D profile must be fit to the posteriors via a Kernel Density Estimator (KDE).

[**PyAutoGalaxy** does not currently support posterior multiplication and an example illustrating this is currently
missing from this tutorial. However, I will discuss KDE multiplication throughout these tutorials to give the
reader context for how this approach to parameter estimation compares to graphical models.]

__Wrap Up__

Lets wrap up the tutorial. The methods used above combine the results of different fits and estimate a global 
value of `slope` alongside estimates of its error. 

In this tutorial, we fitted just 5 datasets. Of course, we could easily fit more datasets, and we would find that
as we added more datasets our estimate of the global slope would become more precise.

In the next tutorial, we will compare this result to one inferred via a graphical model. 
"""

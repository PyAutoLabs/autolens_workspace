"""
Modeling Features: Clumps
=========================

Certain lenses have small galaxies within their Einstein radius, or nearby the lensed source emission. The emission
of these galaxies may overlap the lensed source emission, and their mass may contribute to the lensing of the source.

We may therefore wish to include these additional galaxies in the lens model, as:

 - Light profiles which fit and subtract the emission of these nearby galaxies.
 - Mass profiles which account for their lensing effects via ray-tracing.

The **PyAutoLens** clump API makes it straight forward to include these galaxies, referred to as "clumps", which is
illustrated in this tutorial.

__Data Preparation__

The clump API optionally requires the user to input the centre of each clump in order to set up their light and mass
profile.

The `data_preparation` tutorial `autolens_workspace/*/imaging/data_preparation/optional/clump_centres.py` 
describes how to create these centres and output them to a `.json` file, which are loaded in this example.

__Start Here Notebook__

If any code in this script is unclear, refer to the modeling `start_here.ipynb` notebook for more detailed comments.
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
__Dataset__

Load and plot the strong lens dataset `clumps` via .fits files.
"""
dataset_name = "clumps"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
Visualization of this dataset shows two galaxies outside by nearby the lensed source. 
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We define a bigger circular mask of 6.0" than the 3.0" masks used in other tutorials, to ensure the clump's 
emission is included.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

"""
Lets plot the masked imaging to make sure the clumps are included in the fit.
"""
visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.subplot_dataset()

"""
__Clump Centres__

To set up a lens model including each clump we input the clump centres. 

In principle, a lens model including the clumps could be composed without these centres. For example, if there were
two clumps in the data, we could simply add two additional light and mass profiles into the model we compose. The 
clump API does support this, but we will not use it in this example.

This is because models with clumps with free centres are often too complex to fit. It is likely the fit will infer 
an inaccurate lens model and local maxima. 

For example, a common problem is that one of the clump light profiles intended to model a nearby galaxy instead fit 
one of the lensed source's multiple images. Alternatively, a clump's mass profile may act as the main lens galaxy's.

Therefore, via the clump API we input the centre of each clump, which fixes their light and mass profile centres.

The `data_preparation` tutorial `autolens_workspace/*/imaging/data_preparation/examples/optional/clump_centres.py` 
describes how to create these centres. Using this script they have been output to the `.json` file we load below.
"""
clump_centres = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "clump_centres.json")
)

"""
__Model__ 

Performs the normal steps to set up the main model of the lens galaxy and source.

__Model Cookbook__

A full description of model composition, including lens model customization, is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

mass = af.Model(al.mp.Isothermal)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

bulge = af.Model(al.lp.Sersic)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

"""
__Clump Model__ 

We now use the `ClumpModel` object to create the model for the clumps, which is passed the following:

 - `redshift`: the redshift of every clump, which is the same as the lens galaxy redshift.
 - `clump_centres`: he centre of every clump light and mass profile in the lens model.
 - `light_cls`: the light model used for every clump, which in this example is a `SersicSph`.
 - `mass_cls`: the mass model used for every clump, which in this example is a `IsothermalSph`.
 - `einstein_radius_upper_limit`: the upper limit on the `UniformPrior` assigned to every `einstein_radius` 
 parameter in the mass profiles.

__Notes__

If we passed `light_cls=None` or `mass_cls=None` a clump model can still be composed, however the clumps would
omit either the light or mass profiles.

Clump mass profiles often to go unphysically high `einstein_radius` values, degrading the fit. The 
`einstein_radius_upper_limit` parameter is used to set an upper limit on the `einstein_radius` of every clump mass
to prevent this. A physical upper limit depends on the exact size of the clump, but roughly speaking a stelar
mass of 10^9 solar masses corresponds to an `einstein_radius` of below 0.1", therefore, we set the upper limit to 0.1".
"""
clump_model = al.ClumpModel(
    redshift=0.5,
    centres=clump_centres,
    light_cls=al.lp.SersicSph,
    mass_cls=al.mp.IsothermalSph,
    einstein_radius_upper_limit=0.1,
)

"""
The `ClumpModel.clumps` property makes it straight forward to compose the overall lens model.

This property is a `af.Collection()` object, which we have used to compose models throughout **PyAutoLens**.

It contains `Model` `Galaxy` objects, all of which are at the input redshift of the `ClumpModel` above and which 
contain model light and mass profiles whose centres are fixed to the input `clump_centres` but have their 
remaining parameters free.
"""
print(clump_model.clumps)
print(clump_model.clumps.clump_0.light.centre)
print(clump_model.clumps.clump_1.mass.centre)

"""
Currently, the clump API require that the centres of the light and mass profiles are fixed to the input centres
(but the other parameters of the light and mass profiles remain free). 

A future version of **PyAutoLens** will add more flexibility to the `CLumpModel` object.

Therefore, in this example fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].

 - The source galaxy's light is a parametric `Sersic` [7 parameters].

 - Each clump's light is a parametric `SersicSph` profile with fixed centre [2 clumps x 3 parameters = 6 parameters].

 - Each clump's total mass distribution is a `IsothermalSph` profile with fixed 
 centre [2 clumps x 1 parameters = 2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=20.
"""
# Clumps:

clumps = clump_model.clumps

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source), clumps=clumps)

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms the model includes additional clump galaxies that we defined above.
"""
print(model.info)

"""
__Search + Analysis__ 

The code below performs the normal steps to set up a model-fit.

Given the extra model parameters due to the clumps, we increase the number of live points from the default of
50 to 100 and make the random walk length 10.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="clumps",
    unique_tag=dataset_name,
    n_live=150,
)

analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Time__

Adding clumps to the model increases the likelihood evaluation, because their light profiles need their images 
evaluated and their mass profiles need their deflection angles computed.

However, these calculations are pretty fast for profiles like `SersicSph` and `IsothermalSph`, so only a small
increase in time is expected.

The bigger hit on run time is due to the extra free parameters, which increases the dimensionality of non-linear
parameter space/ This means Nautilus takes longer to converge on the highest likelihood regions of parameter space.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

By plotting the maximum log likelihood `FitImaging` object we can confirm the clumps contribute to the fit.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

These examples show how the results API can be extended to investigate clumps in the results.

__Wrap Up__

The clump API makes it straight forward for us to model galaxy-scale strong lenses with additional components for
the light and mass of nearby objects.

The `autolens_workspace` includes a `group` package, for modeling group scale strong lenses which have multiple lens 
galaxies. When you should use the clump API as shown here, and when you should use the group package? 

The distinction is as follows:

 - A galaxy scale lens is a system which can be modeled to a high level of accuracy using a single light and mass 
 distribution for the main lens galaxy. Including additional galaxies in the model via the clump API makes small 
 improvements on the lens model, but a good fit is possible without them. 
 
 - A group scale lens is a system which cannot be modeled to a high level of accuracy using a single light and mass 
 distribution. Defining a 'main' lens galaxy is unclear and two or more main lens galaxies are required to fit an 
 accurate model. 
 
The `group` package also uses the clump API for model composition, but does so to compose and fit more complex lens 
models.
"""

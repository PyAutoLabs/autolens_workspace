"""
Chaining: SIE to Power-law
==========================

This script chains two searches to fit `PointDict` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source `Galaxy` is a point source `Point`.

The two searches break down as follows:

 1) Models the lens galaxy's mass as an `Isothermal` and the source galaxy's as a point `Point`.

 2) Models the lens galaxy's mass an an `PowerLaw` and the source galaxy's as a point `Point`.

__Why Chain?__

The `EllPower` is a general form of the `Isothermal` which has one additional parameter, the `slope`,
which controls the inner mass distribution as follows:

 - A higher slope concentrates more mass in the central regions of the mass profile relative to the outskirts.
 - A lower slope shallows the inner mass distribution reducing its density relative to the outskirts. 

By allowing the lens model to vary the mass profile's inner distribution, its non-linear parameter space becomes
significantly more complex and a notable degeneracy appears between the mass model`s mass normalization, elliptical
components and slope. This is challenging to sample in an efficient and robust manner, especially when the non-linear
search's initial samples use broad uniform priors on the lens and source parameters.

Search chaining allows us to begin by fitting an `Isothermal` model and therefore estimate the lens's mass
model and the source parameters via a non-linear parameter space that does not have a strong of a parameter degeneracy
present. This makes the model-fit more efficient and reliable.

The second search then fits the `PowerLaw`, using prior passing to initialize the mass and elliptical
components of the lens galaxy as well as the source galaxy's point-source (y,x) centre.
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

Load and plot the `Imaging` of the point-source dataset, purely for visualization of the strong lens.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "point_source", dataset_name)

data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.05
)

"""
__PointDict__

Load and plot the `PointDict` dataset, which is the dataset used to perform lens modeling.
"""
point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

print("Point Source Multiple Image (y,x) Arc-second Coordinates:")
print(point_dict["point_0"].positions.in_list)

visuals = aplt.Visuals2D(positions=point_dict.positions_list)

point_dict_plotter = aplt.PointDictPlotter(point_dict=point_dict)
point_dict_plotter.subplot_positions()
point_dict_plotter.subplot_fluxes()

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("point_source", "chaining", "sie_to_power_law")

"""
__MultipleImageSolver__

Setup the `MultipleImageSolver`.
"""
grid = al.Grid2D.uniform(shape_native=data.shape_native, pixel_scales=data.pixel_scales)

solver = al.MultipleImageSolver(grid=grid, pixel_scale_precision=0.025)

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The source galaxy's is a point `Point` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=9.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)
source = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.Point)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix, name="search[1]__sie", unique_tag=dataset_name, n_live=100
)

analysis_1 = al.AnalysisPoint(point_dict=point_dict, solver=solver)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_1.info)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `PowerLaw` [6 parameters: priors initialized from search 1].
 - The source galaxy's light is again a point `Point` [2 parameters: priors initialized from search 1].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

The term `model` below passes the source model as model-components that are to be fitted for by the  non-linear search. 
Because the source model does not change we can pass its priors by simply using the`model` attribute of the result:
"""
source = result_1.model.galaxies.source

"""
However, we cannot use this to pass the lens galaxy, because its mass model must change from an `Isothermal` 
to an `PowerLaw`. The following code would not change the mass model to an `PowerLaw`:
 
 `lens = result.model.galaxies.lens`
 
We can instead use the `take_attributes` method to pass the priors. Below, we pass the lens of the result above to a
new `PowerLaw`, which will find all parameters in the `Isothermal` model that share the same name
as parameters in the `PowerLaw` and pass their priors (in this case, the `centre`, `ell_comps` 
and `einstein_radius`).

This leaves the `slope` parameter of the `PowerLaw` with its default `UniformPrior` which has a 
`lower_limit=1.5` and `upper_limit=3.0`.
"""
mass = af.Model(al.mp.PowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the `model.info` file of the search 2 model-fit to ensure the priors were passed correctly, as 
well as the checkout the results to ensure an accurate power-law mass model is inferred.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__power_law",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_2 = al.AnalysisPoint(point_dict=point_dict, solver=solver)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

The final results can be summarized via printing `info`.
"""
print(result_2.info)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a lens mass model as an `Isothermal` and 
passed its priors to then fit the more complex `PowerLaw` model. 

This removed difficult-to-fit degeneracies from the non-linear parameter space in search 1, providing a more robust 
and efficient model-fit.
"""

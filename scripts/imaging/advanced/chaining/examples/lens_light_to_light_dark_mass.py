"""
Chaining: Chaining Lens Light To Mass
=====================================

This script chains two searches to fit `Imaging` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a bulge with a linear parametric `Sersic` light profile.
 - The lens galaxy's stellar mass distribution is a bulge tied to the light model above.
 - The lens galaxy's dark matter mass distribution is a `NFWSph`.
 - The source galaxy's light is an `Sersic`.

The two searches break down as follows:

 1) Models the lens galaxy's light using an `Sersic` bulge. The source is present in the image, but modeling it is
 omitted.

 2) Models the lens galaxy's mass using a stellar mass distriubtion which is initialized using the bulge light
 models inferred by search 1, alongside a dark matter profile. The source is again modeled using an `Sersic`

__Why Chain?__

For many strong lenses the lens galaxy's light is distinct from the source galaxy's light, and it is therefore a valid
approach to first subtract the lens's light and then focus on fitting the lens mass model and source's light. This
provides the following benefits:

 - The non-linear parameter space defined by a bulge (N=7), stellar and dark mass (N=5) and parametric source (N=7)
 has N=27 dimensions. By splitting the model-fit into two searches, we fit parameter spaces of dimensions N=11 and then
 N=27, with many priors initialized. These are more efficient to sample and less like to infer a local maxima or
 unphysical solution.

 - The lens galaxy's light traces its mass, so we can use the lens light model inferred in search 1 to initialize
 sampling of the stellar mass model in search 2.

 __Preloading__

When certain components of a model are fixed its associated quantities do not change during a model-fit. For
example, for a lens model where all light profiles are fixed, the PSF blurred model-image of those light profiles
is also fixed.

**PyAutoLens** uses _implicit preloading_ to inspect the model and determine what quantities are fixed. It then stores
these in memory before the non-linear search begins such that they are not recomputed for every likelihood evaluation.

In this example no preloading occurs.

__Start Here Notebook__

If any code in this script is unclear, refer to the `chaining/start_here.ipynb` notebook.
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
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "lens_light_to_light_dark_mass")

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's light is a linear parametric `Sersic` bulge [6 parameters].
 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
bulge = af.Model(al.lp_linear.Sersic)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)

model_1 = af.Collection(galaxies=af.Collection(lens=lens))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We use the results of search 1 to create the lens model fitted in search 2, where:

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__lens_light",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_1.info)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's light and stellar mass is a linear parametric `Sersic` bulge  [6 parameters: priors initialized from 
   search 1].
 - The lens galaxy's dark matter mass distribution is a `NFW` whose centre is aligned with the 
 `Sersic` bulge and stellar mass model above [5 parameters].
 - The lens mass model also includes an `ExternalShear` [2 parameters].
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

We use the `take_attributes` method to pass the priors of the bulge. The reason we use this method is because
the bulge above use a `LightProfile` (e.g. via `al.lp`), whereas the model below gives a `LightAndMassProfile` 
(e.g. via `al.lmp`). 

The `take_attributes` method is used when we pass parameters from two different models. In the example below it finds
all parameters in the `Sersic` and `Sersic` light models that share the same names
as parameters in the ``Sersic` and `Sersic` light and mass models and passes their priors 
(in this case, the `centre`, `ell_comps`, `intensity`, `effective_radius` and `sersic_index`).
"""
bulge = af.Model(al.lmp.Sersic)
bulge.take_attributes(source=result_1.model)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, dark=af.Model(al.mp.NFW))

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model_2 = af.Collection(galaxies=af.Collection(lens=lens))

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
    name="search[2]__light_dark_mass",
    unique_tag=dataset_name,
    n_live=100,
)


analysis_2 = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

The final results can be summarized via printing `info`.
"""
print(result_2.info)

"""
__Wrap Up__

In this example, we passed a bulge lens light model to a decomposed stellar + dark matter mass model. Thus, we
use an initial fit of the lens galaxy's light to better constrained our lens mass model! 

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex lens modeling 
in a robust and efficient way. 

The following example pipelines exploit our ability to model separately the lens's light and its stellar mass to
perform model-fits in non-linear parameter spaces of reduced complexity, as shown in this example:

 `autolens_workspace/imaging/chaining/pipelines/light_parametric__mass_light_dark__source_lp_linear.py`
 `autolens_workspace/imaging/chaining/pipelines/light_parametric__mass_light_dark__source_pixelization.py`
 
__SLaM (Source, Light and Mass)__
 
An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling 
processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
mass model. 

The SLaM Mass pipelines include pipelines which specifically decomposed stellar light + dark matter mass models. These
follow the Light pipelines and pass the priors of the light model in an identical fashion to this example.
"""

"""
Chaining: Double Einstein Ring
==============================

This script chains two searches to fit `Imaging` data of a 'galaxy-scale' strong lens which has two source galaxies at
two different redshifts, forming a double Einstein ring system. This fits a model where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The first source galaxy's mass is a `IsothermalSph` and its light a `Sersic`.
 - The second source galaxy's light is an `Sersic`.

The two searches break down as follows:

 1) Model the lens galaxy mass as an `Isothermal`  and first source galaxy using an `Sersic` light profiles.
 2) Model the lens, first and second source galaxies, where the first source's mass is an `IsothermalSph` and second
  source is an `Sersic`.

__Why Chain?__

Systems with two (or more) strongly lensed sources are a great example of the benefits of search chaining. The lens
model can quickly have many parameters (e.g. N > 20), but many of the components being fitted are only mildly covariant
with one another.

Most importantly, ray-tracing of the first source galaxy does not depend on the properties of the second source galaxy
at all, meaning it can be used to initialize the lens mass model before the second source is fitted. For the simulated
data fitted in this example, we'll see that the first search successfully initializes the lens mass model and first
source model without issue, such that fitting of the second source can be done efficiently.

The only problem is that the light of the second source is included in the data we fit in the first search, and thus
could bias or impact its model fit. To circumvent this, the first search uses a smaller mask which removes the light
of the second source from the model-fit. A larger mask included both sources is then used in the second search.

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
__Dataset__ 

Load and plot the `Imaging` data. N

ote that we use different masks for searches 1 and 2.
"""
dataset_name = "double_einstein_ring"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)

dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "double_einstein_ring")

"""
__Masking (Search 1)__

We apply a smaller circular mask, the radius of which is chosen to remove the light of the second source galaxy.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)

dataset_plotter.subplot_dataset()

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The first source galaxy's light is a linear parametric `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

We therefore omit the second source from the model entirely.
"""
mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
source_0 = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)
source_1 = af.Model(al.Galaxy, redshift=2.0)

model_1 = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1),
)

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
    path_prefix=path_prefix,
    name="search[1]__source_0_parametric",
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
__Masking (Search 2)__

We apply a larger circular mask, which includes the second source galaxy now that it is included in the model.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)

dataset_plotter.subplot_dataset()

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `Isothermal` [parameters fixed to results of search 1].
 - The first source galaxy's light is a linear parametric `Sersic` [parameters fixed to results of search 1].
 - The first source galaxy's mass is a `IsothermalSph` [3 parameters].
 - The second source galaxy's light is a linear parametric `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

The galaxies are assigned redshifts of 0.5, 1.0 and 2.0. This ensures the multi-plane ray-tracing necessary for the 
double Einstein ring lens system is performed correctly.

The lens galaxy's mass and first source galaxy's light are passed as an `instance` (as opposed to the `model` which 
was used in the API tutorial). By passing these objects as an `instance` it passes the maximum log likelihood parameters 
inferred by search 1 as fixed values that are not free parameters fitted for by the non-linear search of search 2.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=result_1.instance.galaxies.lens.mass)
source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=result_1.instance.galaxies.source_0.bulge,
    mass=al.mp.IsothermalSph,
)
source_1 = af.Model(al.Galaxy, redshift=2.0, bulge=al.lp_linear.SersicCore)
source_1.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
source_1.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)

model_2 = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1),
)

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
    name="search[2]__source_1_parametric",
    unique_tag=dataset_name,
    n_live=150,
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

In this example, we used prior passing to initialize a model fit to a double Einstein ring. We exploited the fact that 
ray-tracing of the first source is fully independent of the source behind it, such that we could use it to initialize 
the lens model before fitting the second source.

For certain double Einstein ring systems, it is possible that the light of the first and second sources are harder to
deblend than the simple masking we used in this example. Manual masks drawn using a GUI which removes the second 
source's light will nevertheless always be possible, but more care may be required.

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex lens modeling 
in a robust and efficient way. 

There are currently no pipelines written for double Einstein ring systems, albeit one can craft them by learning the
API and concepts from existing template pipelines. We are still figuring out the most effective way to model double
Einstein ring systems, which is why pipeline templates are not yet written.

__SLaM (Source, Light and Mass)__
 
An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling 
processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
mass model. 

The SLaM pipelines begin with a linear parametric Source pipeline, which then switches to an inversion Source pipeline, 
exploiting the chaining technique demonstrated in this example.
"""

"""
Pipeline: Double Einstein Ring
==============================

By chaining together four searches this script fits `Imaging` dataset of a 'galaxy-scale' strong lens, which has two source galaxies
at two different redshifts, forming a double Einstein ring system. In the final model:

 - The lens galaxy's light is an `Sersic`.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The first source galaxy's mass is a `IsothermalSph` and its light is modeled using an `Inversion`.
 - The second source galaxy's light is modeled using an `Inversion`.

The three searches break down as follows:

 1) Model the lens galaxy using a linear parametric `Sersic` to subtract its emission.
 2) Model the lens galaxy mass as an `Isothermal`  and first source galaxy using an `Sersic` light profiles.
 3) Model the lens, first and second source galaxies, where the first source's mass is an `IsothermalSph` and second
  source is an `Sersic`.
 4) Model the first and second source galaxy simultaneously using an `Inversion` and lens galaxy mass as an
 `Isothermal`.

The approach used in this pipeline and benefits of using chaining searching to fit double einstein ring systems are
described in the script `notebooks/imaging/chaining/double_einstein_ring.ipynb`.

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

ote that we use different masks for each search.
"""
dataset_name = "double_einstein_ring"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "double_einstein_ring")

"""
__Masking (Search 1 & 2)__

We apply a smaller circular mask, the radius of which is chosen to remove the light of the second source galaxy.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's light is a linear parametric `Sersic` bulge [6 parameters].
 - The lens galaxy's mass and both source galaxies are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.
"""
bulge = af.Model(al.lp_linear.Sersic)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)

model_1 = af.Collection(galaxies=af.Collection(lens=lens))

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.
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
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's light is an `Sersic` bulge [Parameters fixed to results of search 1].
 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The first source galaxy's light is a linear parametric `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

We therefore omit the second source from the model entirely.
"""
mass = af.Model(al.mp.Isothermal)

mass.centre = result_1.model.galaxies.lens.bulge.centre

lens = af.Model(
    al.Galaxy, redshift=0.5, bulge=result_1.instance.galaxies.lens.bulge, mass=mass
)
source_0 = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model_2 = af.Collection(galaxies=af.Collection(lens=lens, source_0=source_0))

"""
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__parametric_source_0",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Masking (Search 3)__

We apply a larger circular mask, which includes the second source galaxy now that it is included in the model.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)

dataset_plotter.subplot_dataset()

"""
__Model (Search 3)__

We use the results of searches 1 & 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's light is an `Sersic` bulge [Parameters fixed to results of search 1].
 - The lens galaxy's total mass distribution is an `Isothermal` [Parameters fixed to results of search 2].
 - The first source galaxy's light is a linear parametric `Sersic` [Parameters fixed to results of search 2].
 - The first source galaxy's mass is a `IsothermalSph` [3 parameters].
 - The second source galaxy's light is a linear parametric `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

The galaxies are assigned redshifts of 0.5, 1.0 and 2.0. This ensures the multi-plane ray-tracing necessary for the 
double Einstein ring lens system is performed correctly.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_1.instance.galaxies.lens.bulge,
    mass=result_2.model.galaxies.lens.mass,
)
source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=result_2.model.galaxies.source_0.bulge,
    mass=al.mp.IsothermalSph,
)
source_1 = af.Model(al.Galaxy, redshift=2.0, bulge=al.lp_linear.SersicCore)

model_3 = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1),
)

"""
__Search + Analysis + Model-Fit (Search 3)__

We now create the non-linear search, analysis and perform the model-fit using this model.
"""
search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]__source_2_parametric",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_3 = al.AnalysisImaging(dataset=dataset)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Model (Search 4)__

We use the results of searches 1, 2 & 3 to create the lens model fitted in search 4, where:

 - The lens galaxy's light is an `Sersic` bulge [7 Parameters: we do not use the results of search 1 to 
 initialize priors].
 - The lens galaxy's total mass distribution is an `Isothermal` [5 Parameters: priors initialized from search 2].
 - The first source galaxy's light is a linear parametric `Sersic` [6 parameters: priors initialized from search 2].
 - The first source galaxy's mass is a `IsothermalSph` [3 parameters: priors initialized from search 3].
 - The second source galaxy's light is a linear parametric `Sersic` [6 parameters: priors initialized from search 3].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=29.

The galaxies are assigned redshifts of 0.5, 1.0 and 2.0. This ensures the multi-plane ray-tracing necessary for the 
double Einstein ring lens system is performed correctly.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=af.Model(al.lp_linear.Sersic),
    mass=result_2.model.galaxies.lens.mass,
)
source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=result_2.model.galaxies.source_0.bulge,
    mass=result_3.model.galaxies.source_0.mass,
)
source_1 = af.Model(
    al.Galaxy, redshift=2.0, bulge=result_3.model.galaxies.source_1.bulge
)

model_4 = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1),
)

"""
__Search + Analysis + Model-Fit (Search 4)__

We now create the non-linear search, analysis and perform the model-fit using this model.
"""
search_4 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[4]__parametric_all",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_4 = al.AnalysisImaging(dataset=dataset)

result_4 = search_4.fit(model=model_4, analysis=analysis_4)

"""
__Model (Search 5)__

We use the results of search 4 to create the lens model fitted in search 5, where:

 - The lens galaxy's light is an `Sersic` bulge [Parameters fixed to results of search 4].
 - The lens galaxy's total mass distribution is again an `Isothermal` [Parameters fixed to results of search 4].
 - The first source galaxy's mass is a `IsothermalSph` [Parameters fixed to results of search 4].
 - The first source-galaxy's light uses an `Overlay` image-mesh, `Delaunay` mesh and `ConstantSplit` regularization 
 scheme [3 parameters].
 - The second source-galaxy's light uses an `Overlay` image-mesh, `Delaunay` mesh and `ConstantSplit` regularization  
 scheme [3 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6.
"""
lens = result_4.instance.galaxies.lens
source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    mass=result_4.instance.galaxies.source_0.mass,
    pixelization=af.Model(
        al.Pixelization,
        image_mesh=al.image_mesh.Overlay,
        mesh=al.mesh.Delaunay,
        regularization=al.reg.ConstantSplit,
    ),
)
source_1 = af.Model(
    al.Galaxy,
    redshift=2.0,
    pixelization=af.Model(
        al.Pixelization,
        image_mesh=al.image_mesh.Overlay,
        mesh=al.mesh.Delaunay,
        regularization=al.reg.ConstantSplit,
    ),
)
model_5 = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1),
)

"""
__Analysis__
"""

analysis_5 = al.AnalysisImaging(dataset=dataset)

"""
__Search + Model-Fit__

We now create the non-linear search and perform the model-fit using this model.
"""
search_5 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[5]__sources_pixelization",
    unique_tag=dataset_name,
    n_live=100,
)

result_5 = search_5.fit(model=model_5, analysis=analysis_5)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a model fit to a double Einstein ring using 
two `Inversion`'s.

Fitting just the `Inversion` by itself for a double Einstein ring system is practically impossible, due to the 
unphysical solutions which reconstruct its light as a demagnified version of each source. Furthermore, it helped to 
ensure that the model-fit ran efficiently.

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex lens modeling 
in a robust and efficient way. 

There are currently no pipelines written for double Einstein ring systems, albeit one can craft them by learning the
API and concepts from existing template pipelines. We are still figuring out the most effective way to model double
Einstein ring systems, which is why pipeline templates are not yet written.

__SLaM (Source, Light and Mass)__

An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling 
processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
mass model. 

The SLaM pipelines begin with a parametric Source pipeline, which then switches to an inversion Source pipeline, 
exploiting the chaining technique demonstrated in this example.
"""

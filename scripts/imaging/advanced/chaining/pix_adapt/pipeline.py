"""
Chaining: Adaptive Pixelization Example Pipeline
================================================

Non-linear search chaining is an advanced model-fitting approach in **PyAutoLens** which breaks the model-fitting
procedure down into multiple non-linear searches, using the results of the initial searches to initialization parameter
sampling in subsequent searches. This contrasts the `modeling` examples which each compose and fit a single lens
model-fit using one non-linear search.

An overview of search chaining is provided in the `autolens_workspace/*/imaging/chaining/api.py` script, make
sure to read that before reading this script!

The script `adapt_pix.py` introduces **PyAutoLens**'s pixelization adaption features, which passes the results of 
previous model-fits performed by earlier searches to searches performed later in the chain. 

This script gives an example pipeline using these features. It is an adaption of the 
pipeline `chaining/examples/pipeline.py` and it can be used as a template for setting up any pipeline to use these
features.

Adaptive features are also built into the SLaM pipelines by default.

By chaining together five searches this script fits strong lens `Imaging`, where in the final model:

 - The lens galaxy's light is a parametric `Sersic` and `Exponential`.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source galaxy is modeled using the `KMeans` image-mesh, `Delaunay` mesh and `AdaptiveBrightness` regularization
 schemes which use adaptive features.

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

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__source_x2"
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
path_prefix = path.join("imaging", "chaining", "pix_adapt_pipeline")

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Adapt Setup__

The `SetupAdapt` determines the pixelization adaption setup. 

The following options are available:

 - `mesh_pixels_fixed`: Use a fixed number of source pixels in the pixelization's mesh.
 
 - `search_pix_cls`: The non-linear search used to adapt the pixelization's mesh and regularization scheme.
 
 - `search_pix_dict`: The dictionary of search options for the adapt model-fit searches.
 
The mesh and regularization schemes which adapt to the source's properties are not passed into
`SetupAdapt`, but are used in this example script below.

In this example, we only fix the number of source pixels to 1500, which balances computational runtimes with the
resolution of the source reconstruction. The adapt search uses the default settings, including a `Nautilus` 
non-linear search.
"""
setup_adapt = al.SetupAdapt(mesh_pixels_fixed=1000)

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's light is a parametric `Sersic` bulge [7 parameters].

 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
bulge = af.Model(al.lp.Sersic)

model_1 = af.Collection(
    galaxies=af.Collection(lens=af.Model(al.Galaxy, redshift=0.5, bulge=bulge))
)

search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]_light[lp]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's light and stellar mass is an `Sersic` bulge [Parameters fixed to results of search 1].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with the 
 `Sersic` bulge and stellar mass model above [3 parameters].

 - The lens mass model also includes an `ExternalShear` [2 parameters].

 - The source galaxy's light is a parametric `Sersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

NOTES:

 - By using the fixed `bulge` model from the result of search 1, we are assuming this is a sufficiently 
 accurate fit to the lens's light that it can reliably represent the stellar mass.
"""
bulge = result_1.instance.galaxies.lens.bulge

dark = af.Model(al.mp.NFWMCRLudlow)
dark.centre = bulge.centre
dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e15)
dark.redshift_object = redshift_lens
dark.redshift_source = redshift_source

model_2 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=bulge,
            dark=af.Model(al.mp.NFW),
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    )
)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]_light[fixed]_mass[light_dark]_source[lp]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

We use the results of searches 1 and 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's light and stellar mass is a parametric `Sersic` bulge [7 parameters: priors initialized 
   from search 1].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with the 
 `Sersic` bulge and stellar mass model above [3 parameters: priors initialized from search 2].

 - The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 2].

 - The source galaxy's light is a parametric `Sersic` [7 parameters: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

Notes:

 - This search attempts to address any issues there may have been with the bulge's stellar mass model.
"""
bulge = result_1.model.galaxies.lens.bulge

dark = result_2.model.galaxies.lens.dark
dark.centre = bulge.centre

model_3 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=bulge,
            dark=dark,
            shear=result_2.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy, redshift=1.0, bulge=result_2.model.galaxies.source.bulge
        ),
    )
)

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]_light[lp]_mass[light_dark]_source[lp]",
    unique_tag=dataset_name,
    n_live=150,
)

analysis_3 = al.AnalysisImaging(dataset=dataset)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

We use the results of searches 3 to create the lens model fitted in search 4, where:

 - The lens galaxy's light and stellar mass is an `Sersic` bulge [Parameters fixed to results of search 3].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` [Parameters fixed to results of 
 search 3].

 - The lens mass model also includes an `ExternalShear` [Parameters fixed to results of search 3].

 - The source galaxy's light uses an `Overlay` image-mesh [2 parameters].
 
 - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].

 - This pixelization is regularized using a `ConstantSplit` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

NOTES:

 - This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
 of the regularization scheme, before using these models to refit the lens mass model.
"""
pixelization = af.Model(
    al.Pixelization,
    image_mesh=al.image_mesh.Hilbert,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.ConstantSplit,
)

model_4 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_3.instance.galaxies.lens.bulge,
            dark=result_3.instance.galaxies.lens.dark,
            shear=result_3.instance.galaxies.lens.shear,
        ),
        source=af.Model(al.Galaxy, redshift=redshift_source, pixelization=pixelization),
    )
)

search_4 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[4]_light[fixed]_mass[fixed]_source[pix_init]",
    unique_tag=dataset_name,
    n_live=50,
)

analysis_4 = al.AnalysisImaging(dataset=dataset)

result_4 = search_4.fit(model=model_4, analysis=analysis_4)

"""
__Model +  Search (Search 5)__

We use the results of searches 3 and 4 to create the lens model fitted in search 5, where:

 - The lens galaxy's light and stellar mass is an `Sersic` bulge[11 parameters: priors initialized from search 3].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` [8 parameters: priors initialized 
 from search 3].

The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 3].


 - The source-galaxy's light uses an `Overlay` image-mesh [parameters fixed to results of search 4].

 - The source-galaxy's light uses a `Delaunay` mesh [parameters fixed to results of search 4].

 - This pixelization is regularized using a `ConstantSplit` scheme [parameters fixed to results of search 4]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.
"""
model_5 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_3.model.galaxies.lens.bulge,
            dark=result_3.model.galaxies.lens.dark,
            shear=result_3.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=result_4.instance.galaxies.source.pixelization,
        ),
    )
)

search_5 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[5]_light[lp]_mass[light_dark]_source[pix]",
    unique_tag=dataset_name,
    n_live=50,
)

"""
__Positions + Analysis + Model-Fit (Search 5)__

We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis_5 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=result_4.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

result_5 = search_5.fit(model=model_5, analysis=analysis_5)

"""
__Wrap Up__

It took us 7 searches to set up hyper-mode, just so that we could fit a complex lens model in one final search. However,
this is what is unfortunately what is necessary to fit the most complex lens models accurately, as they really are
trying to extract a signal that is contained in the intricate detailed surfaceness brightness of the source itself.

The final search in this example fitting an `PowerLaw`, but it really could have been any of the complex
models that are illustrated throughout the workspace (e.g., decomposed light_dark models, more complex lens light
models, etc.). You may therefore wish to adapt this pipeline to fit the complex model you desire for your science-case,
by simplying swapping out the model used in search 8.
 
However, it may instead be time that you check out the for the SLaM pipelines, which have hyper-mode built in but 
provide a lot more flexibility in customizing the model and fitting procedure to fully exploit the hyper-mode features
whilst fitting many different lens models.
"""

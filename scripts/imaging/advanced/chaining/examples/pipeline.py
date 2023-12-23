"""
Pipelines: Light Parametric + Mass Light Dark + Source Inversion
================================================================

By chaining together five searches this script fits `Imaging` dataset of a 'galaxy-scale' strong lens, where in the
final model:

 - The lens galaxy's light is a parametric bulge with a parametric `Sersic` light profile.
 - The lens galaxy's stellar mass distribution is a bulge tied to the light model above.
 - The source galaxy is modeled using a `Pixelization`.

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
dataset_name = "mass_stellar_dark"
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
path_prefix = path.join("imaging", "pipelines")

"""
__Redshifts__

The redshifts of the lens and source galaxies.

In this analysis, they are used to explicitly set the `mass_at_200` of the elliptical NFW dark matter profile, which is
a model parameter that is fitted for.
"""
redshift_lens = 0.5
redshift_source = 1.0

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

 - The lens galaxy's light and stellar mass is an `Sersic` bulge  [Parameters fixed to results of search 1].

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

 - The lens galaxy's light and stellar mass is a parametric `Sersic` bulge [7 parameters: priors initialized from 
   search 1].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with the 
 `Sersic` bulge and stellar mass model above [3 parameters: priors initialized from search 2].

 - The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 2].

 - The source galaxy's light is a parametric `Sersic` [7 parameters: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.

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
    image_mesh=al.image_mesh.KMeans,
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

 - The lens galaxy's light and stellar mass is an `Sersic` bulge [7 parameters: priors initialized from search 3].

 - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` [8 parameters: priors initialized 
 from search 3].

The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 3].

 - The source-galaxy's light uses an `Overlay` image-mesh [parameters fixed to results of search 4].

 - The source-galaxy's light uses a `Delaunay` mesh [parameters fixed to results of search 4].

 - This pixelization is regularized using a `ConstantSplit` scheme [parameters fixed to results of search 4]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=17.
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
Finish.
"""

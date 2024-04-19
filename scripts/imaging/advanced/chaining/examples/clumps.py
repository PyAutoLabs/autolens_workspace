"""
Chaining: Clumps
================

Certain lenses may have galaxies inside or nearby their lensed source emission, which we may wish to include in the
len model.

This could be as light profiles which fit and subtract their emission and / or mass profiles which account for their
lensing effects.

The **PyAutoLens** clump API makes it straight forward to include these objects, all we need is the centre of each
clump (which we can estimate from its luminous emission).

The `modeling` tutorial `autolens_workspace/*/imaging/modeling/features/clumps.py` describes a simple
example usihng the clump API which does not use search chaining.

The `data_preparation` tutorial `autolens_workspace/*/imaging/data_preparation/examples/optional/clump_centres.py`
describes how to create these centres.

This example shows how the clump API can be used in a pipeline. By chaining together three searches This script
fits an `Imaging` dataset of a 'galaxy-scale' strong lens, where in the final model:

 - The lens galaxy's light is a parametric `Sersic`.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy is a parametric `Sersic`.
 - Two clumps are included in the lens model which have `SersicSph` light profiles and `IsothermalSph` mass profiles.

__Start Here Notebook__

If any code in this script is unclear, refer to the `chaining/start_here.ipynb` notebook.
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


"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.

We define a bigger circular mask of 6.0" than the 3.0" masks used in other tutorials, to ensure the clump's emission is 
included.
"""
dataset_name = "clumps"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
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
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Clump Model__ 

This model includes clumps, which are `Galaxy` objects with light and mass profiles fixed to an input centre which 
model galaxies nearby the strong lens system.

A full description of the clump API is given in the 
script `autolens_workspace/*/imaging/modeling/features/clumps.py`
"""
clump_centres = clump_centres = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "clump_centres.json"))
)

clump_model = al.ClumpModel(
    redshift=0.5,
    centres=clump_centres,
    mass_cls=al.mp.IsothermalSph,
    light_cls=al.lp.SersicSph,
    einstein_radius_upper_limit=1.0,
)

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's light is a parametric `Sersic` bulge [7 parameters].

 - The lens galaxy's mass and source galaxy are omitted.
 
 - The clump light profiles are fitted (but their mass profiles are omitted) [6 parameters]

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.
"""
model_1 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, bulge=af.Model(al.lp.Sersic))
        + clump_model.clumps_light_only
    ),
)

"""
Given the extra parameters in the model due to the clumps, we increase the number of live points from the default of
50 to 100 and make the random walk length 10, both of which lead to a more thorough sampling of parameter space
(see `autolens_workspace/*/howtolens/chapter_optional/tutorial_searches.py`).

We do this for every search below.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]_light[lp]",
    unique_tag=dataset_name,
    n_live=150,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

Search 2 fits a lens model where:

 - The lens galaxy's light is an `Sersic` bulge [Parameters fixed to results of search 1].

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].

 - The source galaxy's light is a parametric `Sersic` [7 parameters].

 - The clump mass profiles are fitted and their light profiles are fixed to the previous search [2 parameters]

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=16.
"""
model_2 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_1.instance.galaxies.lens.bulge,
            mass=al.mp.Isothermal,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=redshift_source, bulge=al.lp.Sersic),
    )
    + clump_model.clumps_mass_only
    + al.util.chaining.clumps_from(result=result_1),
)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]_light[fixed]_mass[sie]_source[lp]",
    unique_tag=dataset_name,
    n_live=150,
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

Search 2 fits a lens model where:

 - The lens galaxy's light is an `Sersic` bulge [7 Parameters: we do not use the results of search 1 to 
 initialize priors].

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters: priors
 initialized from search 2].

 - The source galaxy's light is a parametric `Sersic` [7 parameters: priors initialized from search 2].

 - The clump light and mass profiles are fitted [8 parameters: priors initialized from search 2]

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=25.

The result of search 1 is sufficient for subtracting the lens light, so that search 2 can accurately fit the lens
mass model and source light. However, the lens light model may not be particularly accurate, so we opt not to use
the result of search 1 to initialize the priors.
"""
model_3 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=af.Model(al.lp.Sersic),
            mass=result_2.model.galaxies.lens.mass,
            shear=result_2.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            bulge=result_2.model.galaxies.source.bulge,
        ),
    )
    + clump_model.clumps_light_only
    + al.util.chaining.clumps_from(result=result_2, mass_as_model=True),
)

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]_light[lp]_mass[total]_source[lp]",
    unique_tag=dataset_name,
    n_live=150,
)

analysis_3 = al.AnalysisImaging(dataset=dataset)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

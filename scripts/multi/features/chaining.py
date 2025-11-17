"""
Chaining: Point Source to Imaging
=================================

This script chains three searches to fit `Imaging` data of a strong lens with multiple lens galaxies where:

 - The group consists of three whose light models are MGEs and total mass distributions
 are `IsothermalSph` models.
 - The source galaxy's light is an MGE.

The two searches break down as follows:

 1) Model the lens galaxy masses with a point source galaxy, fitting just the position information in the source.
 2) Model the full surface brightness information in the `Imaging` data using `LightProfile`'s for the lens galaxies
 and lensed source.

__Why Chain?__

There are a number of benefits of chaining a point source fit to an imaging fit, as opposed to doing just one fit:

 - The point source fit is lower dimensionality than a light profile fit and computationally very fast. It can
 therefore provide accurate estimates for the lens and source model parameters. However, the point source fit does
 not extract anywhere near the maximal amount of information from the data.

 - The fit to the imaging data is much higher dimensionality and computationally slower. It would be challenging to
 fit this model without accurate initialization of the lens model parameters. However, it extracts a lot more
 information from the data.

This script therefore initializes the lens model efficiently using a point-source fit and then switches to a full
fit on the imaging data.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax.numpy as jnp
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens dataset `group` point source dataset and imaging, and plot them.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "group", dataset_name)

imaging = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

dataset = al.from_json(
    file_path=Path(dataset_path, "point_dataset.json"),
)

visuals = aplt.Visuals2D(positions=dataset.positions)

array_plotter = aplt.Array2DPlotter(array=imaging.data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = Path("group") / "chaining" / "point_to_imaging"

"""
__PointSolver__

Define the position solver used for the point source fitting.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
    xp=jnp,
)

"""
__Model (Search 1)__

Compose the lens model by loading it from a .json file made in the file `group/model_maker/simple.py`:

 - There are three lens galaxy's with `IsothermalSph` total mass distributions, with the prior on the centre of each 
 profile informed by its observed centre of light [9 parameters].
 - The source galaxy's light is a point `PointFlux` [3 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""
model_path = Path("dataset", "group", dataset_name)

lenses_file = Path(model_path, "lenses.json")
lenses = af.Collection.from_json(file=lenses_file)

sources_file = Path(model_path, "sources.json")
sources = af.Collection.from_json(file=sources_file)

galaxies = lenses + sources

model_1 = af.Collection(galaxies=galaxies)

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
    name="search[1]_point_source",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisPoint(dataset=dataset, solver=solver)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarised in the `info` attribute.
"""
print(result_1.info)

"""
__Masking (Search 2)__

The model-fit to imaging data requires a `Mask2D`. 

Note how this has a radius of 9.0", much larger than most example lenses!
"""
mask_radius = 9.0

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    radius=mask_radius,
)

imaging = imaging.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
imaging_plotter.subplot_dataset()

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - There are again three lens galaxy's with MGEs [6 parameters: centres initialized from model].
 - The three lens galaxy's have `IsothermalSph` mass distributions [9 parameters: priors initialized from search 1].
 - The source-galaxy's light uses a MGE with 1 x 20 Gaussians light profile [4 parameters: centre initialized from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=31.

The term `model` below passes the source model as model-components that are to be fitted for by the 
non-linear search. We pass the `lens` as a `model`, so that we can use the mass model inferred by search 1. The source
does not use any priors from the result of search 1.
"""
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

lens_0 = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=al.mp.IsothermalSph)
lens_0.bulge.centre = model_1.galaxies.lens_0.mass.centre
lens_0.mass = result_1.model.galaxies.lens_0.mass

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

lens_1 = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=al.mp.IsothermalSph)
lens_1.bulge.centre = model_1.galaxies.lens_1.mass.centre
lens_1.mass = result_1.model.galaxies.lens_1.mass

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=True,
)

lens_2 = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=al.mp.IsothermalSph)
lens_2.bulge.centre = model_1.galaxies.lens_2.mass.centre
lens_2.mass = result_1.model.galaxies.lens_2.mass

source_0 = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)
source_0.bulge.centre = result_1.model.galaxies.source_0.point_0.centre

model_2 = af.Collection(
    galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, lens_2=lens_2, source_0=source_0
    ),
)

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_2.info)

"""
__Analysis + Position Likelihood__

We add a penalty term ot the likelihood function, which penalizes models where the brightest multiple images of
the lensed source galaxy do not trace close to one another in the source plane. This removes "demagnified source
solutions" from the source pixelization, which one is likely to infer without this penalty.

A comprehensive description of why we do this is given at the following readthedocs page. I strongly recommend you 
read this page in full if you are not familiar with the positions likelihood penalty and demagnified source reconstructions:

 https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

__Brief Description__

In this example we update the positions between searches, where the positions correspond to the (y,x) locations of the 
lensed source's multiple images. When a model-fit uses positions, it requires them to trace within a threshold value of 
one another for every mass model sampled by the non-linear search. If they do not, a penalty term is added to the
likelihood penalizing that solution 

Below, we use the results of the first search to compute the lensed source positions that are input into search 2. The
code below uses the maximum log likelihood model mass model and source galaxy centre, to determine where the source
positions are located in the image-plane. 

We also use this result to set the `threshold`, whereby the threshold value is based on how close these positions 
trace to one another in the source-plane (using the best-fit mass model again). This threshold is multiplied by 
a `factor` to ensure it is not too small (and thus does not remove plausible mass  models). If, after this 
multiplication, the threshold is below the `minimum_threshold`, it is rounded up to this minimum value.
"""
positions_likelihood = al.PositionsLH(threshold=1.0, positions=dataset.positions)

"""
__Analysis + Positions__

In this example we update the positions between searches, where the positions correspond to the (y,x) locations of the 
lensed source's multiple images. When a model-fit uses positions, it requires them to trace within a threshold value of 
one another for every mass model sampled by the non-linear search. If they do not, the model likelihood is heavily
penalized.

Below, we use the point source dictionary positions and a threshold double the resolution of the data, which should be
sufficient for this analysis.
"""
analysis_2 = al.AnalysisImaging(
    dataset=imaging, positions_likelihood_list=[positions_likelihood]
)

"""
__Search + Model-Fit__

We now create the non-linear search and perform the model-fit using this model.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__imaging",
    unique_tag=dataset_name,
    n_live=150,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

The final results can be summarised via printing `info`.
"""
print(result_2.info)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a lens model using a point source dataset and passed this 
a second fit which fitted the full `Imaging` dataset. 

This circumvented the challenge with initializing a high dimensionality complex lens model to `Imaging` data where
the computational run time is slower.
"""

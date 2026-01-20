"""
Pixelization: Adaptive
======================

Non-linear search chaining is an advanced model-fitting approach which breaks the model-fitting procedure down into
multiple non-linear searches, using the results of the initial searches to initialization parameter sampling in
subsequent searches. This contrasts the `modeling` example which fits a single lens model-fit using one non-linear search.

An overview of search chaining is provided in the `autolens_workspace/*/guides/modeling/chaining` script, make
sure to read that before reading this script!

This script introduces adaptive pixdelizations features, which use search chainig to pass the results of previous
model-fits performed by earlier searches to searches performed later in the chain, in order to adapt the pixelizaiton's
mesh and regularization to the source's unlensed properties. It also uses the results of previous searches to
calculate and pass the multiple image-plane positions of the lensed source to later searches, which resamples
bad mass models removing demagnified source reconstructions.

This script illustrates using the `RectangularAdaptImage` mesh and `AdaptiveBrightness` regularization
scheme to adapt the source reconstruction to the source galaxy's morphology (as opposed to the methods used in other
examplesw hich adapt to the mass model magnification and apply a constant regularization scheme).

This script illustrates the API used for adaptive pixelizations, but does not go into the details of how they
work. This is described in chapter 4 of the **HowToLens** lectures.

__Why Chain?__

There are a number of benefits of chaining a linear source model and  a pixelized source, as opposed to fitting the
pixelization in one search:

 - Parametric sources are computationally faster to fit. Therefore, even though the MGE has more
 parameters for the search to fit than a pixelized source, the model-fit is faster overall.

 - pixelizations often go to unphysical solutions where the mass model goes to high / low normalization_list and the source
 is reconstructed as a demagnified version of the image. (see Chapter 4, tutorial 6 for a complete description of
 this effect). This does not occur for a linear source, therefore the mass model can be initialized using a
 parametric source, which sets up the search which fits a pixelization so as to not sample these unphysical solutions.

 - The positions and positions threshold can be updated to further ensure these unphysical solutions do not bias the
 model-fit. The updated positions use the maximum log likelihood mass model of the first search to determine the
 image-plane position of the lensed source. In the second search, we then require that a mass model must trace these
 positions within a threshold arc-secoond value of one another in the source-plane, removing these unphysical solutions.

__Model__

This script chains three searches to fit `Imaging` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a multi-Gaussian expansion (MGE) in search 1 and a pixelization in searches 2 and 3.

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/chaining.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking + Positions__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = Path("imaging") / "pixelization" / "adaptive"

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 - The source galaxy's light is an MGE with 1 x 20 Gaussians [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.

The benefit of using an MGE in search 1 is that it is computationally fast to fit, allowing the
non-linear search to quickly converge to a reasonable lens model. This lens model is then used 
to set up the adaptive pixelization and multiple image positions in search 2.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

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
    path_prefix=path_prefix,
    name="search[1]__parametric",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__JAX & Preloads__

The `autolens_workspace/*/imaging/features/pixelization/modeling` example describes how JAX required preloads in
advance so it knows the shape of arrays it must compile functions for.
"""
mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = al.Preloads(
    mapper_indices=al.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

"""
__Model (Search 2)__

To use adapt features, we require a model image of the lensed source galaxy, which is what the code will adapt the
analysis too.

When we begin a fit, we do not have such an image, and thus cannot use the adaptive features. This is why search chaining
is required, it allows us to perform an initial model-fit which gives us the source image, which we can then use to
perform a subsequent model-fit which adapts the analysis to the source's properties.

We therefore compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In the first
search our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light uses no image-mesh (only used for Delaunay meshes) [0 parameters].
 
 - The source-galaxy's light uses a 20 x 20 `RectangularAdaptDensity` mesh [0 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.

**Chaining API:** The term `model` below passes the source model as model-components that are to be fitted for by the 
non-linear search. We pass the `lens` as a `model`, so that we can use the mass model inferred by search 1. The source
does not use any priors from the result of search 1.
"""
lens = result_1.model.galaxies.lens

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

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
analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[
        result_1.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__adaptive_pixelization_setup",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_2 = al.AnalysisImaging(dataset=dataset, preloads=preloads)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Adaptive Pixelization__

Search 3 uses two adaptive pixelization classes that have not been used elsewhere in the workspace:

 - `RectangularAdaptImage` mesh: adapts the rectangular source-pixel upsampling to the source's unlensed morphology. This 
 means that more rectangular pixels will be used where the source is located, even if its far away from the caustic
 and therefore in lower magnification regions.

 - `AdaptiveBrightness` regularization: adapts the regularization coefficient to the source's
 unlensed morphology. This means that the source's brightest regions are regularized less than its faintest regions, 
 ensuring that the bright central regions of the source are not over-smoothed.
 
This adaptive mesh and regularization produces a significantly better lens analysis and reconstruction of the source 
galaxy than other schemes used throughout the workspace. Now you are familiar with them, you should
never use anything else!

It is recommend that the parameters governing these features are always fitted using a fixed lens light and
mass model. This ensures the adaptation is performed quickly, and removes degeneracies in the lens model that
are difficult to sample. Extensive testing has shown that this does not reduce the accuracy of the lens model.

For this reason, search 2 fixes the lens galaxy's light and mass model to the best-fit model of search 1. A third
search will then fit for the lens galaxy's light and mass model using these adaptive features.

The details of how the above features work is not provided here, but is given at the end of chapter 4 of the HowToLens
lecture series.

__Model (Search 3)__

We therefore compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In 
the second search our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` with fixed parameters from 
   search 1 [0 parameters].
 
 - The source galaxy's light uses no image-mesh (only used for Delaunay meshes) [0 parameters].
 
 - The source-galaxy's light uses a 20 x 20 `RectangularAdaptImage` mesh [0 parameters].

 - This pixelization is regularized using a `AdaptiveBrightness` scheme [2 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=4.

**Chaining API:** The term `instance` below passes the lens model as fixed model-components that are not to be
fitted for by the non-linear search. We pass the `lens` as an `instance`, so that its parameters are fixed to 
the best-fit values of search 2. The source
"""
lens = result_2.instance.galaxies.lens

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularAdaptImage(shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=pixelization,
)

model_3 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis (Search 2)__

We now create the analysis for the second search.

__Adapt Images__

When we create the analysis, we pass it an `adapt_images`, which contains a dictionary mapping each galaxy name 
(e.g. galaxies.source) to the corresponding lens subtracted image of the source galaxy from the result of search 1. 

This is telling the `Analysis` class to use the lens subtracted images of this fit to guide the `AdaptiveBrightness` 
regularization for the source galaxy. Specifically, it uses the lens subtracted signal to noise map of the lensed 
source in order  to adapt the location of the source-pixels to the source's brightest regions and lower the 
regularization coefficient in  these regions.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result_2)

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

analysis_3 = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
    positions_likelihood_list=[
        result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
)

"""
__Search + Model-Fit (Search 3)__

We now create the non-linear search and perform the model-fit using this model.
"""
search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]__adaptive_pixelization",
    unique_tag=dataset_name,
    n_live=75,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Result (Search 3)__

If you inspect and compare the results of searches 2 and 3, you'll note how the model-fits of search 3 have a much
higher likelihood than search 2 and how the source reconstruction has congregated it pixels to the bright central
regions of the source. This indicates that a much better result has been achieved.

__Model + Search + Analysis + Model-Fit (Search 4)__

We now perform a final search which uses the `AdaptiveBrightness` regularization with their parameter fixed to the 
results of search 2.

The lens mass model is free to vary.

The analysis class still uses the adapt images from search 2, because this is what the adaptive features adapted
to in search 3.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=result_3.instance.galaxies.source.pixelization,
)

model_4 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_4 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[4]__adapt",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_4 = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
)

result_4 = search_4.fit(model=model_4, analysis=analysis_4)

"""
__SLaM Pipelines__

The API above allows you to write modeling code using adaptive features yourself.

However, it is recommend you use the Source, Light and Mass (SLaM) pipeline, whcih are carefully crafted to automate 
lens modeling of large samples whilst ensuring models of the highest complexity can be reliably fitted.

The SLaM pipelines are built around the use of these adaptive pixelization features, with the Source pipeline first 
so that these features are set up robustly before more complex lens light and mass models are fitted.

The example `guides/modeling/slam_start_here` provides a full run through of how to use the SLaM pipelines with 
adaptive pixelizations.
"""

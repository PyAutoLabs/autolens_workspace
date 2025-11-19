"""
Pixelization: Adaptive
======================

Non-linear search chaining is an advanced model-fitting approach which breaks the model-fitting procedure down into
multiple non-linear searches, using the results of the initial searches to initialization parameter sampling in
subsequent searches. This contrasts the `modeling` example which fits a single lens model-fit using one non-linear search.

An overview of search chaining is provided in the `autolens_workspace/*/guides/modeling/chaining` script, make
sure to read that before reading this script!

This script introduces adaptive pixdelizations features, which use the results of previous model-fits performed by
earlier searches to searches performed later in the chain, in order to adapt the pixelizaiton's mesh and regularization
to the source's unlensed properties.

This script illustrates using the `RectangularSource` mesh and `AdaptiveBrightness` regularization
scheme to adapt the source reconstruction to the source galaxy's morphology (as opposed to the methods used in other
examplesw hich adapt to the mass model magnification and apply a constant regularization scheme).

This script illustrates the API used for adaptive pixelizations, but does not go into the details of how they
work. This is described in chapter 4 of the **HowToLens** lectures.

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
path_prefix = Path("imaging") / "chaining" / "pix_adapt"

"""
__JAX & Preloads__

The `autolens_workspace/*/imaging/features/pixelization/modeling` example describes how JAX required preloads in
advance so it knows the shape of arrays it must compile functions for.
"""
image_mesh = None
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
__Model (Search 1)__

To use adapt features, we require a model image of the lensed source galaxy, which is what the code will adapt the
analysis too.

When we begin a fit, we do not have such an image, and thus cannot use the adaptive features. This is why search chaining
is required, it allows us to perform an initial model-fit which gives us the source image, which we can then use to
perform a subsequent model-fit which adapts the analysis to the source's properties.

We therefore compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In the first
search our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light uses no image-mesh (only used for Delaunay meshes) [0 parameters].
 
 - The source-galaxy's light uses a 20 x 20 `RectangularMagnification` mesh [0 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

pixelization = af.Model(
    al.Pixelization,
    image_mesh=None,
    mesh=al.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=al.reg.Constant,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__adapt",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset, preloads=preloads)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Adaptive Pixelization__

Search 2 uses two adaptive pixelization classes that have not been used elsewhere in the workspace:

 - `RectangularSource` mesh: adapts the rectangular source-pixel upsampling to the source's unlensed morphology. This 
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

__Model (Search 2)__

We therefore compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In 
the second search our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` with fixed parameters from 
   search 1 [0 parameters].
 
 - The source galaxy's light uses no image-mesh (only used for Delaunay meshes) [0 parameters].
 
 - The source-galaxy's light uses a 20 x 20 `RectangularSource` mesh [0 parameters].

 - This pixelization is regularized using a `AdaptiveBrightness` scheme [2 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=4.
"""
lens = result_1.instance.galaxies.lens

pixelization = af.Model(
    al.Pixelization,
    image_mesh=None,
    mesh=al.mesh.RectangularSource(shape=mesh_shape),
    regularization=al.reg.AdaptiveBrightness,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=pixelization,
)

model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis (Search 2)__

We now create the analysis for the second search.

__Adapt Images__

When we create the analysis, we pass it an `adapt_image_maker`, which contains the lens subtracted image of the 
source galaxy from the result of search 1. 

This is telling the `Analysis` class to use the lens subtracted images of this fit to guide the `AdaptiveBrightness` 
regularization for the source galaxy. Specifically, it uses the lens subtracted image of the lensed source in order 
to adapt the location of the source-pixels to the source's brightest regions and lower the regularization coefficient in 
these regions.
"""
analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=result_1),
    preloads=preloads,
)

"""
__Search + Model-Fit (Search 2)__

We now create the non-linear search and perform the model-fit using this model.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix, name="search[2]__adapt", unique_tag=dataset_name, n_live=75
)

analysis_2._adapt_images = analysis_2.adapt_images

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

If you inspect and compare the results of searches 1 and 2, you'll note how the model-fits of search 2 have a much
higher likelihood than search 1 and how the source reconstruction has congregated it pixels to the bright central
regions of the source. This indicates that a much better result has been achieved.

__Model + Search + Analysis + Model-Fit (Search 3)__

We now perform a final search which uses the `AdaptiveBrightness` regularization with their parameter fixed to the 
results of search 2.

The lens mass model is free to vary.

The analysis class still uses the adapt images from search 1, because this is what the adaptive features adapted
to in search 2.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=result_2.instance.galaxies.source.pixelization,
)

model_3 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]__adapt",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_3 = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=result_1),
    preloads=preloads,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__SLaM Pipelines__

The API above allows you to write modeling code using adaptive features yourself.

However, it is recommend you use the Source, Light and Mass (SLaM) pipeline. This pipelines has been carefully crafted 
to automate lens modeling of large samples whilst ensuring models of the highest complexity can be reliably fitted 
using adaptive pixelizations.

In fact, the SLaM pipelines are built around the use of adaptive features, with the Source pipeline first so that
these features are set up robustly before more complex lens light and mass models are fitted.

The example `guides/modeling/slam_start_here` provides a full run through of how to use the SLaM pipelines with 
adaptive pixelizations.
"""

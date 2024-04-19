"""
Chaining: Pixelization Adapt
============================

Non-linear search chaining is an advanced model-fitting approach in **PyAutoLens** which breaks the model-fitting
procedure down into multiple non-linear searches, using the results of the initial searches to initialization parameter
sampling in subsequent searches. This contrasts the `modeling` examples which each compose and fit a single lens
model-fit using one non-linear search.

An overview of search chaining is provided in the `autolens_workspace/*/imaging/chaining/api.py` script, make
sure to read that before reading this script!

This script introduces **PyAutoLens**'s pixelization adaption features, which pass the results of previous
model-fits performed by earlier searches to searches performed later in the chain, in order to adapt the pixelizaiton's
mesh and regularization to the source's unlensed properties.

This script illustrates using the `Hilbert` image-mesh, `Delaunay` mesh and `AdaptiveBrightnessSplit` regularization
scheme to adapt the source reconstruction to the source galaxy's morphology (as opposed to schemes introduced
previously which adapt to the mass model magnification or apply a constant regularization pattern).

This script illustrates the API used for pixelization adaptive features, but does not go into the details of how they
work. This is described in chapter 4 of the **HowToLens** lectures.

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
__Dataset + Masking + Positions__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__no_lens_light"
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

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "pix_adapt")

"""
__Model (Search 1)__

To use adapt features, we require a model image of the lensed source galaxy, which is what the code will adapt the
analysis too.

When we begin a fit, we do not have such an image, and thus cannot use the adapt features. This is why search chaining
is important -- it allows us to perform an initial model-fit which gives us the source image, which we can then use to
perform a subsequent model-fit which adapts the analysis to the source's properties.

We therefore compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In the first
search our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light uses an `Overlay` image-mesh with fixed resolution 30 x 30 pixels [0 parameters].
 
 - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].

 - This pixelization is regularized using a `ConstantSplit` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

pixelization = af.Model(
    al.Pixelization,
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.ConstantSplit,
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

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Adaptive Pixelization__

Search 2 is going to use two adaptive pixelization features that have not been used elsewhere in the workspace:

 - The `Hilbert` image-mesh, which adapts the distribution of source-pixels to the source's unlensed morphology. This
 means that the source's brightest regions are reconstructed using significantly more source pixels than seen for
 the `Overlay` image mesh. Conversely, the source's faintest regions are reconstructed using significantly fewer
 source pixels.

 - The `AdaptiveBrightness` regularization scheme, which adapts the regularization coefficient to the source's
 unlensed morphology. This means that the source's brightest regions are regularized less than its faintest regions, 
 ensuring that the bright central regions of the source are not over-smoothed.
 
Both of these features produce a significantly better lens analysis and reconstruction of the source galaxy than
other image-meshs and regularization schemes used throughout the workspace. Now you are familiar with them, you should
never use anything else!

It is recommend that the parameters governing these features are always fitted from using a fixed lens light and
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
 
 - The source galaxy's light uses a `Hilbert` image-mesh with fixed resolution 1000 pixels [2 parameters].
 
 - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].

 - This pixelization is regularized using a `AdaptiveBrightnessSplit` scheme [2 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=4.
"""
lens = result_1.instance.galaxies.lens

pixelization = af.Model(
    al.Pixelization,
    image_mesh=al.image_mesh.Hilbert(pixels=1000),
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
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

When we create the analysis, we pass it a `adapt_images`, which contains the model image of the source galaxy from 
the result of search 1. 

This is telling the `Analysis` class to use the model-images of this fit to aid the fitting of the `Hilbert` 
image-mesh and `AdaptiveBrightness` regularization for the source galaxy. Specifically, it uses the model image 
of the lensed source in order to adapt the location of the source-pixels to the source's brightet regions and lower
the regularization coefficient in these regions.

__Image Mesh Settings__

The `Hilbert` image-mesh may not fully adapt to the data in a satisfactory way. Often, it does not place enough
pixels in the source's brightest regions and it may place too few pixels further out where the source is not observed.
To address this, we use the `settings_inversion` input of the `Analysis` class to specify that we require the following:

- `image_mesh_min_mesh_pixels_per_pixel=3` and `image_mesh_min_mesh_number=5`: the five brightest source image-pixels
   must each have at least 3 source-pixels after the adaptive image mesh has been computed. If this is not the case,
   the model is rejected and the non-linear search samples a new lens model.
 
- `image_mesh_adapt_background_percent_threshold=0.1` and `image_mesh_adapt_background_percent_check=0.8`: the faintest
   80% of image-pixels must have at least 10% of the total source pixels, to ensure the regions of the image with no
   source-flux are reconstructed using sufficient pixels. If this is not the case, the model is rejected and the
   non-linear search samples a new lens model.

These inputs are a bit contrived, but have been tested to ensure they lead to good lens models.
"""
analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=result_1),
    settings_inversion=al.SettingsInversion(
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=5,
        image_mesh_adapt_background_percent_threshold=0.1,
        image_mesh_adapt_background_percent_check=0.8,
    ),
)

"""
__Search + Model-Fit (Search 2)__

We now create the non-linear search and perform the model-fit using this model.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix, name="search[2]__adapt", unique_tag=dataset_name, n_live=75
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

If you inspect and compare the results of searches 1 and 2, you'll note how the model-fits of search 2 have a much
higher likelihood than search 1 and how the source reconstruction has congregated it pixels to the bright central
regions of the source. This indicates that a much better result has been achieved.

__Model + Search + Analysis + Model-Fit (Search 3)__

We now perform a final search which uses the `Hilbert` image-mesh and `AdaptiveBrightness` regularization with their
parameter fixed to the results of search 2.

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
    dataset=dataset, adapt_image_maker=al.AdaptImageMaker(result=result_1)
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__SLaM Pipelines__

The API above allows you to use adaptive features yourself, and you should go ahead an explore them on datasets you
are familiar with.

However, you may also wish to use the Source, Light and Mass (SLaM) pipelines, which are pipelines that
have been carefully crafted to automate lens modeling of large samples whilst ensuring models of the highest
complexity can be reliably fitted.

These pipelines are built around the use of adaptive features -- for example the Source pipeline comes first so that
these features are set up robustly before more complex lens light and mass models are fitted.

Below, we detail a few convenience functions that make using adaptive features in the SLaM pipelines straight forward.
"""

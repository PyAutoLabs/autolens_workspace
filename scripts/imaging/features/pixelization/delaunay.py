"""
Pixelization: Delaunay
======================

Non-linear search chaining is an advanced model-fitting approach which breaks the model-fitting
procedure down into multiple non-linear searches, using the results of the initial searches to initialization parameter
sampling in subsequent searches. This contrasts the `modeling` examples which each compose and fit a single lens
model-fit using one non-linear search.

An overview of search chaining is provided in the `autolens_workspace/*/guides/modeling/chaining` script, make
sure to read that before reading this script!

This script introduces adaptive pixdelizations features, which pass the results of previous
model-fits performed by earlier searches to searches performed later in the chain, in order to adapt the pixelizaiton's
mesh and regularization to the source's unlensed properties.

This script illustrates using the `Hilbert` image-mesh, `Delaunay` mesh and `AdaptiveBrightnessSplit` regularization
scheme to adapt the source reconstruction to the source galaxy's morphology (as opposed to schemes introduced
previously which adapt to the mass model magnification or apply a constant regularization pattern).

This script illustrates the API used for pixelization adaptive features, but does not go into the details of how they
work. This is described in chapter 4 of the **HowToLens** lectures.

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/chaining.ipynb` notebook.
"""

try:
    import numba
except ModuleNotFoundError:
    input(
        "##################\n"
        "##### NUMBA ######\n"
        "##################\n\n"
        """
        Numba is not currently installed.

        Numba is a library which makes PyAutoLens run a lot faster. Certain functionality is disabled without numba
        and will raise an exception if it is used.

        If you have not tried installing numba, I recommend you try and do so now by running the following 
        commands in your command line / bash terminal now:

        pip install --upgrade pip
        pip install numba

        If your numba installation raises an error and fails, you should go ahead and use PyAutoLens without numba to 
        decide if it is the right software for you. If it is, you should then commit time to bug-fixing the numba
        installation. Feel free to raise an issue on GitHub for support with installing numba.

        A warning will crop up throughout your *PyAutoLens** use until you install numba, to remind you to do so.

        [Press Enter to continue]
        """
    )

# No JAX support for Delaunay

from autoconf import conf

use_jax = conf.instance["general"]["jax"]["use_jax"]

if use_jax:
    raise RuntimeError(
        """
        You have enabled JAX in the config file (general.yaml -> jax -> use_jax = true).
        
        For a Delaunay mesh, JAX is not currently supported. Please disable JAX to run this script
        by changing the setting in the general.yaml file to use_jax = false.
        """
    )

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
path_prefix = Path("imaging") / "chaining" / "pix_adapt_delaunay"

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

When we begin a fit, we do not have such an image, and thus cannot use the adapt features. This is why search chaining
is important -- it allows us to perform an initial model-fit which gives us the source image, which we can then use to
perform a subsequent model-fit which adapts the analysis to the source's properties.

We therefore compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In the first
search our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light uses an `Overlay` image-mesh with fixed resolution 30 x 30 pixels [0 parameters].
 
 - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

pixelization = af.Model(
    al.Pixelization,
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
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
    number_of_cores=2,
    preloads=preloads,
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

When we create the analysis, we pass it a `adapt_images`, which contains the lens subtracted image of the source galaxy from 
the result of search 1. 

This is telling the `Analysis` class to use the lens subtracted images of this fit to aid the fitting of the `Hilbert` 
image-mesh and `AdaptiveBrightness` regularization for the source galaxy. Specifically, it uses the model image 
of the lensed source in order to adapt the location of the source-pixels to the source's brightest regions and lower
the regularization coefficient in these regions.

__Image Mesh Settings__

The `Hilbert` image-mesh may not fully adapt to the data in a satisfactory way. Often, it does not place enough
pixels in the source's brightest regions and it may place too few pixels further out where the source is not observed.
To address this, we use the `settings_inversion` input of the `Analysis` class to specify that we require the following:
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
    dataset=dataset,
    adapt_image_maker=al.AdaptImageMaker(result=result_1),
    preloads=preloads,
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

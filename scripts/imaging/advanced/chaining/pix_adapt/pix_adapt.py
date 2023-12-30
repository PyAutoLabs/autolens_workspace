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

This script illustrates using the `KMeans` image-mesh, `Delaunay` mesh and `AdaptiveBrightness` regularization
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

positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "pix_adapt")

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
__Model (Search 1)__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In the first
search our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source-galaxy's light uses a `DelaunayBrightness` pixelization with fixed resolution 30 x 30 pixels (0 parameters).

 - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

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
__Model (Search 2)__

Search 2, our source model now uses the `KMeans` image-mesh, `Delaunay` mesh and `AdaptiveBrightness` regularization
scheme that adapt to the source's unlensed morphology. These use the model-images of search 1, which is passed to the
`Analysis` class below. 

We also use the results of search 1 to create the lens `Model` that we fit in search 2. This is described in the 
`api.py` chaining example.
"""
lens = result_1.model.galaxies.lens

pixelization = af.Model(
    al.Pixelization,
    image_mesh=al.image_mesh.Hilbert,
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
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

When we create the analysis, we pass it a `adapt_result`, which is the result of search 1. This is telling the 
`Analysis` class to use the model-images of this fit to aid the fitting of the `KMeans` image-mesh, `Delaunay` mesh 
and `AdaptiveBrightness` regularization for the source galaxy.

If you inspect and compare the results of searches 1 and 2, you'll note how the model-fits of search 2 have a much
higher likelihood than search 1 and how the source reconstruction has congregated it pixels to the bright central
regions of the source. This indicates that a much better result has been achieved, the reasons for which are discussed
in chapter 5 of the **HowToLens** lectures.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix, name="search[2]__adapt", unique_tag=dataset_name, n_live=75
)

analysis_2 = al.AnalysisImaging(dataset=dataset, adapt_result=result_1)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
Fin.
"""

"""
Pixelization: CPU Fast Modeling
===============================

Pixelization examples have used JAX to speed up computations via GPU acceleration.

However, for pixelizations there are a number of situations where JAX and GPU acceleration will not provide fast
run times:

- `VRAM`: JAX acceleration only provides significant speed up when the GPU VRAM is large, with examples using a
  rectangular   mesh of shape 20 x 20 to keep VRAM requirements low. If you do not have access to a modern GPU with >16GB
  VRAM, JAX GPU calculations will often be slow.

- `Sparsity`: Pixelization calculations use large, but very sparse matrices. JAX does not currently support sparse
  matrix operations, meaning that many calculations are performed on large dense matrices which are slow. In a nutshell,
  this means for high resolution data (e.g. `pixel_scales=0.03` or smaller), JAX often becomes slower than CPU
  computations, even when large VRAM GPUs are used.

This example illustrates how to perform fast pixelization modeling on a CPU without JAX. Instead it combines
`the library `numba` with Python `multiprocessing` to perform fast pixelization modeling on a CPU. If you have access
to a HPC with many CPU cores (e.g. > 10), this method can often outperform JAX GPU acceleration for high resolution
imaging data where exploiting matrix sparsity is important.
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

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import matplotlib.pyplot as plt
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
__W_Tilde__

The linear algebra calculations performed to fit a pixelized source to imaging data can use an alternative 
mathetical formalism called the `w_tilde` formalism.

The details of how this work are not important, but the key point is that it significantly speeds up calculations
when using a pixelized source by exploiting the sparsity of matrices involved in the calculations. The way
this is achieved does not currently support JAX and therefore does not support GPU acceleration.

To activate the `w_tilde` formalism, we use the `apply_w_tilde()` method of the `Imaging` dataset, which calculates
and stores a matrix called the `w_tilde` matrix in the dataset, which is used when a pixelized source is fitted to the 
data. 

Computing this matrix takes between a few seconds to a few minutes depending on the size of the dataset, but once 
computed it reused for every model-fit, meaning that for modeling using a pixelized source it provides a significant 
speed up.
"""
dataset = dataset.apply_w_tilde()

"""
__JAX & Preloads__

The `autolens_workspace/*/imaging/features/pixelization/modeling` example describes how JAX required preloads in
advance so it knows the shape of arrays it must compile functions for.

The code below is the same as in that example, but note how the `mesh_shape` is now (30 x 30) because the exploitation
of matrix sparsity by CPU calculations combined with more abundent CPU RAM means we can now use higher resolution meshes
than when using JAX GPU acceleration.
"""
image_mesh = None
mesh_shape = (30, 30)
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
__Fit__

In the example `imaging/features/pixelization/fit.py`, we illustrate how to use a pixelized source
with a rectangular mesh to fit imaging data.

Below, we illustrate a fit using the same pixelization, but now using fast CPU calculations with numba.
"""
mesh = al.mesh.RectangularMagnification(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()
ffff

"""
__Model__

Using a Del
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

search_1 = af.Nautilus(
    path_prefix=Path("features"),
    name="delaunay",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=2,
    preloads=preloads,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

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

__Likelihood Function__

The example `imaging/features/pixelization/likelihood_function.py` provides a step-by-step description of how
a likelihood evaluation is performed for imaging data using a pixelized source reconstruction with a rectangular
mesh.

We now give the same step-by-step description for a pixelized source reconstruction using a Delaunay mesh and
adaptive features.

We only describe code which is specific to Delaunay meshes and adaptive features -- for all other aspects of the likelihood
evaluation, refer to rectangular mesh example.
"""
dataset_path = Path("dataset", "imaging", "simple")

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=masked_dataset)
dataset_plotter.subplot_dataset()

masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1,
    over_sample_size_pixelization=1,
)

grid_plotter = aplt.Grid2DPlotter(grid=masked_dataset.grids.pixelization)
grid_plotter.figure_2d()

bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=2.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

lens_galaxy = al.Galaxy(redshift=0.5, bulge=bulge, mass=mass, shear=shear)

"""
__Source Galaxy Pixelization and Regularization__

The source galaxy is reconstructed using a pixel-grid, in this example a Delaunay mesh, which accounts for 
irregularities and asymmetries in the source's surface brightness. 

A constant regularization scheme is applied which applies a smoothness prior on the reconstruction. 

One of the biggest differences between a Delaunay mesh and rectangular mesh is how the centres of the mesh pixels
in the source-plane are computed. 

For the rectangular mesh, the pixel centres are computed by overlaying a uniform grid over the source-plane.

For a Delaunay mesh, the uniform grid is instead laid over the image-plane to create a course grid of (y,x) coordinates.
These are then ray-traced to the source-plane and are used as the vertexes of the Delaunay triangles.
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),  # Specific to Delaunay
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Lens Light__
"""
image = lens_galaxy.image_2d_from(grid=masked_dataset.grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=masked_dataset.grid)
galaxy_plotter.figures_2d(image=True)

blurring_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grids.blurring)

galaxy_plotter = aplt.GalaxyPlotter(
    galaxy=lens_galaxy, grid=masked_dataset.grids.blurring
)
galaxy_plotter.figures_2d(image=True)

convolved_image_2d = masked_dataset.psf.convolved_image_from(
    image=image, blurring_image=blurring_image_2d
)

array_2d_plotter = aplt.Array2DPlotter(array=convolved_image_2d)
array_2d_plotter.figure_2d()

lens_subtracted_image_2d = masked_dataset.data - convolved_image_2d

array_2d_plotter = aplt.Array2DPlotter(array=lens_subtracted_image_2d)
array_2d_plotter.figure_2d()

"""
__Source Pixel Centre Calculation__

In order to reconstruct the source galaxy using a Delaunay mesh, we need to determine the centres of the Delaunay
source pixels.

The image-mesh `Overlay` object computes the source-pixel centres in the image-plane (which are ray-traced to the
source-plane below). The source pixelization therefore adapts to the lens model magnification, because more
source pixels will congregate in higher magnification regions.

This calculation is performed by overlaying a uniform regular grid with an `pixelization_shape_2d` over the image
mask and retaining all pixels that fall within the mask. This uses a `Grid2DSparse` object.
"""
image_plane_mesh_grid = pixelization.image_mesh.image_plane_mesh_grid_from(
    mask=masked_dataset.mask,
)

"""
Plotting this grid shows a sparse grid of (y,x) coordinates within the mask, which will form our source pixel centres.
"""
visuals = aplt.Visuals2D(grid=image_plane_mesh_grid)
dataset_plotter = aplt.ImagingPlotter(dataset=masked_dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(data=True)

"""
__Ray Tracing__
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
The source code gets quite complex when handling grids for a pixelization, but it is all handled in
the `TracerToInversion` objects.

The plots at the bottom of this cell show the traced grids used by the source pixelization, showing
how the Delaunay mesh and traced image pixels are constructed.
"""
tracer_to_inversion = al.TracerToInversion(tracer=tracer, dataset=masked_dataset)

# A list of every grid (e.g. image-plane, source-plane) however we only need the source plane grid with index -1.
traced_grid_pixelization = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.pixelization
)[-1]

# This functions a bit weird - it returns a list of lists of ndarrays. Best not to worry about it for now!
traced_mesh_grid = tracer_to_inversion.traced_mesh_grid_pg_list[-1][-1]

mat_plot = aplt.MatPlot2D(axis=aplt.Axis(extent=[-1.5, 1.5, -1.5, 1.5]))

grid_plotter = aplt.Grid2DPlotter(grid=traced_grid_pixelization, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
We have also ray-traced the coarse grid of image-pixel coordinates used to form the source pixelization's
Delaunay mesh, which we can also plot.
"""
grid_plotter = aplt.Grid2DPlotter(grid=traced_mesh_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Border Relocation__

Coordinates that are ray-traced near the mass profile centres are heavily demagnified and may trace to far outskirts of
the source-plane. 

Border relocation is performed on both the traced image-pixel grid and traced mesh pixels, therefore ensuring that
the vertexes of the Delaunay triangles are not at the extreme outskirts of the source-plane.
"""
from autoarray.inversion.pixelization.border_relocator import BorderRelocator

border_relocator = BorderRelocator(mask=masked_dataset.mask, sub_size=1)

relocated_grid = border_relocator.relocated_grid_from(grid=traced_grid_pixelization)

relocated_mesh_grid = border_relocator.relocated_mesh_grid_from(
    grid=traced_mesh_grid, mesh_grid=traced_mesh_grid
)

mat_plot = aplt.MatPlot2D(axis=aplt.Axis(extent=[-1.5, 1.5, -1.5, 1.5]))

grid_plotter = aplt.Grid2DPlotter(grid=relocated_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

grid_plotter = aplt.Grid2DPlotter(grid=relocated_mesh_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Delaunay Mesh__

The relocated mesh grid is used to create the `Pixelization`'s Delaunay mesh using the `scipy.spatial` library.
"""
grid_delaunay = al.Mesh2DDelaunay(
    values=relocated_mesh_grid,
)

"""
Plotting the Delaunay mesh shows that the source-plane and been discretized into a grid of irregular Delaunay pixels.

(To plot the Delaunay mesh, we have to convert it to a `Mapper` object, which is described in the next likelihood step).

Below, we plot the Delaunay mesh without the traced image-grid pixels (for clarity) and with them as black dots in order
to show how each set of image-pixels fall within a Delaunay pixel.
"""
mapper_grids = al.MapperGrids(
    mask=mask,
    source_plane_data_grid=relocated_grid,
    source_plane_mesh_grid=grid_delaunay,
    image_plane_mesh_grid=image_plane_mesh_grid,
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.figure_2d(interpolate_to_uniform=False)

visuals = aplt.Visuals2D(
    grid=mapper_grids.source_plane_data_grid,
)
mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals)
mapper_plotter.figure_2d(interpolate_to_uniform=False)

"""
__Image-Source Mapping__
"""
mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index

print(pix_indexes_for_sub_slim_index[0:9])

visuals = aplt.Visuals2D(indexes=[list(range(2050, 2090))])

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)
mapper_plotter.subplot_image_and_mapper(
    image=lens_subtracted_image_2d, interpolate_to_uniform=False
)

pix_indexes = [[200]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(indexes=indexes)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(
    image=lens_subtracted_image_2d, interpolate_to_uniform=False
)

mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for Delaunay
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for Delaunay
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

indexes_source_pix_200 = np.nonzero(mapping_matrix[:, 200])

print(indexes_source_pix_200[0])

array_2d = al.Array2D(values=mapping_matrix[:, 200], mask=masked_dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

blurred_mapping_matrix = masked_dataset.psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix, mask=masked_dataset.mask
)

plt.imshow(
    blurred_mapping_matrix,
    aspect=(blurred_mapping_matrix.shape[1] / blurred_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

indexes_source_pix_200 = np.nonzero(blurred_mapping_matrix[:, 200])

print(indexes_source_pix_200[0])

array_2d = al.Array2D(values=blurred_mapping_matrix[:, 200], mask=masked_dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

print(f"Mapping between image pixel 0 and source pixel 2 = {mapping_matrix[0, 2]}")

data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(lens_subtracted_image_2d),
    noise_map=np.array(masked_dataset.noise_map),
)

plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

source_pixel_0 = 0
source_pixel_1 = 1

print(curvature_matrix[source_pixel_0, source_pixel_1])

array_2d = al.Array2D(
    values=blurred_mapping_matrix[:, source_pixel_0], mask=masked_dataset.mask
)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

array_2d = al.Array2D(
    values=blurred_mapping_matrix[:, source_pixel_1], mask=masked_dataset.mask
)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=source_galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

plt.imshow(regularization_matrix)
plt.colorbar()
plt.show()
plt.close()

curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction, interpolate_to_uniform=False)

mapped_reconstructed_image_2d = (
    al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
)

mapped_reconstructed_image_2d = al.Array2D(
    values=mapped_reconstructed_image_2d, mask=mask
)

array_2d_plotter = aplt.Array2DPlotter(array=mapped_reconstructed_image_2d)
array_2d_plotter.figure_2d()

"""
__Likelihood Function__
"""
model_image = convolved_image_2d + mapped_reconstructed_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

chi_squared_map = al.Array2D(values=chi_squared_map, mask=mask)

array_2d_plotter = aplt.Array2DPlotter(array=chi_squared_map)
array_2d_plotter.figure_2d()

regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)

print(log_evidence)

"""
__Fit__

This process to perform a likelihood function evaluation is what is performed in the `FitImaging` object.
"""
fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_border_relocator=True),
    preloads=preloads,
)
fit_log_evidence = fit.log_evidence
print(fit_log_evidence)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `Nautilus` (https://github.com/joshspeagle/Nautilus)
but **PyAutoLens** supports multiple MCMC and optimization algorithms. 

__Sub Gridding__

The calculation above uses a `Grid2D` object, with a `sub-size=1`, meaning it does not perform oversampling to
evaluate the light profile flux at every image pixel.

**PyAutoLens** has alternative methods of computing the lens galaxy images above, which uses a grid whose sub-size
adaptively increases depending on a required fractional accuracy of the light profile.

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/two_d/grid_iterate.py

__Sourrce Plane Interpolation__

For the `Delaunay` mesh used in this example, every image-sub pixel maps to a single source Voronoi
pixel. Therefore, the plural use of `pix_indexes` is not required. However, for other pixelizations each sub-pixel
can map to multiple source pixels with an interpolation weight (e.g. `Delaunay` triangulation or a `Voronoi` mesh
which uses natural neighbor interpolation).

`MapperVoronoiNoInterp.pix_index_for_sub_slim_index`:
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/voronoi.py

`pixelization_index_for_voronoi_sub_slim_index_from`:
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/util/mapper_util.py

The number of pixels that each sub-pixel maps too is also stored and extracted. This is used for speeding up
the calculation of the `mapping_matrix` described next.

As discussed above, because for the `VoronoiNoInterp` pixelization where every sub-pixel maps to one source pixel,
every entry of this array will be equal to 1.
"""
# pix_sizes_for_sub_slim_index = mapper.pix_sizes_for_sub_slim_index

"""
When each sub-pixel maps to multiple source pixels, the mappings are described via an interpolation weight. For 
example, for a `Delaunay` triangulation, every sub-pixel maps to 3 Delaunay triangles based on which triangle
it lands in.

For the `VoronoiNoInterp` pixelization where every sub-pixel maps to a single source pixel without inteprolation,
every entry of this weight array is 1.0.
"""
# pix_weights_for_sub_slim_index = mapper.pix_weights_for_sub_slim_index

"""
__Wrap Up__

We have presented a visual step-by-step guide to the **PyAutoLens** likelihood function, which uses a pixelization, 
regularization scheme and inversion to reconstruct the source galaxy.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in this package. In brief, these describe:

 - **Sub-gridding**: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 ray-traced to the source-plane and paired fractionally with each source pixel.

 - **Source-plane Interpolation**: Using a Delaunay triangulation or Delaunay mesh with natural neighbor interpolation
 to pair each image (sub-)pixel to multiple source-plane pixels with interpolation weights.

 - **Source Morphology Pixelization Adaption**: Adapting the pixelization such that is congregates source pixels around
 the source's brightest regions, as opposed to the magnification-based pixelization used here.

 - **Luminosity Weighted Regularization**: Using an adaptive regularization coefficient which adapts the level of 
 regularization applied to the source based on its luminosity.
"""

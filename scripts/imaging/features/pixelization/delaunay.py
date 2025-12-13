"""
Pixelization: Delaunay
======================

The majority of pixelized source reconstructions in the workspace use a rectangular mesh to reconstruct
the source's surface brightness.

This example illustrates an alternative pixelization that uses a Delaunay triangulation mesh to reconstruct the
source.

The approach is distinct from the rectangular mesh and has a number of traits which are unique to it:

- `Adaptive Mesh`: In the source plane, the Delaunay mesh uses irregularly shaped triangles to reconstruct the
  source, as opposed to uniform rectangular pixels. This allows the mesh to better adapt to irregular and
  asymmetric source morphologies and change the distribution of source pixels to better match the source's
  surface brightness.

- `Image Mesh`: The vertexes of the Delaunay triangles are computed by overlaying a coarse uniform grid in the
  image-plane and ray-tracing these coordinates to the source-plane. This is unlike the rectangular mesh, which
  simply overlays a uniform grid in the source-plane. This again helps the Delaunay mesh to better adapt to the
  source's surface brightness.

- `Interpolation`: The Delaunay mesh uses a different interpolation scheme to the rectangular mesh, which is
  barycentric interpolation within each triangle. This is different to the rectangular mesh, which uses bilinear
  interpolation within each rectangular pixel.

- `Regularization`: The Delaunay mesh provides different approaches to regularization, with the default being
  one which uses the barycentric coordinates of the triangles to compute how source pixels are regularized with
  their neighbors.

Currently it is not expected that the Delaunay is better or worse than the rectangular mesh, it is simply a different
approach to pixelization that may work better for certain datasets.

__JAX + GPU__

Generating a Delaunay mesh currently supports JAX and GPU acceleration, however certain operations (e.g.
generating the Delaunay triangulation itself) do not run on the GPU because they cannot be easily
converted to JAX.

Instead, JAX sends them to a CPU, runs them there, and then sends the results back to the GPU. This process is
very efficient, because these operations run very fast on a CPU and the data being sent back and forth is small.
Current benchmarking suggests the Delaunay runs less than twice as long as the same fit using a rectangular mesh,
but scientfically offers better results in many cases.

If you do want to run only on CPU, you can use fast CPU method described in
example `imaging/features/pixelization/cpu_fast_modeling` with the Delaunay mesh.
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
__JAX & Preloads__

The example in `autolens_workspace/*/imaging/features/pixelization/modeling` explains why JAX requires certain
arrays to be **preloaded** before the fit begins. JAX must know the shape of arrays in advance so it can compile
functions for them.

For a Delaunay mesh, the vertices of the triangles are defined by (y, x) coordinates in the image-plane. These
coordinates are then ray-traced into the source-plane for each mass model sampled during the non-linear search.
Because this ray-tracing happens repeatedly, the `image_plane_mesh_grid` must be computed once at the start and
passed into a `Preloads` object.

Below, we compute this `image_plane_mesh_grid` using an **Overlay image-mesh**, which places a regular grid of
(y, x) points across the image-plane. This has a mild adaptive effect: regions of high lens magnification receive
more source pixels once they are ray-traced. Later in this example, we switch to a **Hilbert mesh**, which adapts
the pixel distribution more strongly to the source’s surface brightness.

Unlike regular pixelizations, which define a `mesh_shape` to set the total number of source pixels, Delaunay
meshes instead use an `image_mesh_shape`, because the triangulation comes from the overlaid image-plane grid.

Another feature of pixelizations is that all pixels at the edge of the mesh in the source-plane are forced to
solutions of zero brightness by the linear algebra solver. This prevents unphysical solutions where pixels at the
# edge of the mesh reconstruct bright surface brightnesses, often because they fit residuals from the lens
light subtraction.

This requires us to input the `source_pixel_zeroed_indices` into the `Preloads` object, which for rectangular meshes
was simply the edge pixels of the rectangular grid which could be computed via their 2D indices. 

For an image-plane mesh, we simply add a circle of edge points to the image-plane mesh-grid after it has been computed.
We pass the indices of these edge points to the `Preloads` object so that the linear algebra solver knows to force these
pixels to zero during the fit.
"""
image_mesh = al.image_mesh.Overlay(shape=(26, 26))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

image_plane_mesh_grid_edge_pixels = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=image_plane_mesh_grid_edge_pixels,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

total_linear_light_profiles = 0

mapper_indices = al.mapper_indices_from(
    total_linear_light_profiles=total_linear_light_profiles,
    total_mapper_pixels=total_mapper_pixels,
)

# Extract the last `image_plane_mesh_grid_edge_pixels` indices, which correspond to the circle edge points we added

source_pixel_zeroed_indices = mapper_indices[-image_plane_mesh_grid_edge_pixels:]

preloads = al.Preloads(
    mapper_indices=mapper_indices,
    source_pixel_zeroed_indices=source_pixel_zeroed_indices,
)

"""
__Fit__

In the example `imaging/features/pixelization/fit.py`, we illustrate how to use a pixelized source
with a rectangular mesh to fit imaging data.

Below, we use a Delaunay mesh to perform a fit using the Delaunay source reconstruction.

The API is nearly identical to the rectangular mesh example, noting that the use of 
preloads with an `image_plane_mesh_grid` and the `Delaunay` mesh changes the 
calculation internally.
"""
mesh = al.mesh.Delaunay()
regularization = al.reg.ConstantSplit(coefficient=1.0)

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

adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={source: image_plane_mesh_grid}
)

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
    adapt_images=adapt_images,
)

"""
By plotting the fit, we see that the Delaunay source does a good job at capturing the appearance of the source galaxy
using adaptive triangular pixels.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Model__

We now perform lens modeling using the Delaunay pixelization with the Overlay image-mesh.

The code below is a simple adaptive modeling example using the Delaunay mesh, which mirrors the
API used in other pixelization modeling examples.

The example `imaging/features/pixelization/adaptive.py` illustrates how to use adaptive features to
adapt the rectangular mesh and its regularization to the source's surface brightness. In particular, an image
of the lensed source is passed to the modeling via the `AdaptImages` object, in order to adapt
the mesh and regularization during the model-fit.

The same object is used to pass the `image_plane_mesh_grid` to the modeling. Above, this image-plane mesh grid
is an `Overlay` mesh and does not specifically adapt to the source's surface brightness, thus pairing it with
the source as done below seems redundant. However, in a moment we will switch to a `Hilbert` image-mesh, which
does adapt to the source's surface brightness, meaning this pairing is necessary.
"""
adapt_images = al.AdaptImages(
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

"""
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
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_1 = af.Nautilus(
    path_prefix=Path("features"),
    name="delaunay",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    preloads=preloads,
)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Adaptive Delaunay__

The example `imaging/features/pixelization/adaptive.py` illustrates how to use adaptive features to
adapt the rectangular mesh and its regularization to the source's surface brightness.

The image-mesh has a special adaptive variant called the `Hilbert` image-mesh, which adapts the distribution 
of source-pixels to the source's unlensed morphology. This means that the source's brightest regions are 
reconstructed using significantly more source pixels than seen for the `Overlay` image mesh. 
Conversely, the source's faintest regions are reconstructed using significantly fewer source pixels.

Unlike the adaptive rectangular mesh, the Hilbert image-plane mesh is computed before modeling, passed
to the `AdaptImages` object, and remains fixed during the model-fit.

It is recommend that the parameters governing these features are always fitted from using a fixed lens light and
mass model. This ensures the adaptation is performed quickly, and removes degeneracies in the lens model that
are difficult to sample. Given the Hilbert mesh is fixed, this modeling only fits for the regularization coefficients
of the adaptive regularization scheme.

For this reason, search 2 fixes the lens galaxy's light and mass model to the best-fit model of search 1. A third
search will then fit for the lens galaxy's light and mass model using these adaptive features.

The details of how the above features work is not provided here, but is given at the end of chapter 4 of the HowToLens
lecture series.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result_1)

image_mesh = al.image_mesh.Hilbert(pixels=1000)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

image_plane_mesh_grid_edge_pixels = 30

"""
__Image Plane Mesh Grid Edge Points__

What the code does:
When we developed the MGE lens light model, we learned that the edge pixels of the source reconstruction may reconstruct a faint amount of light that is from the edge of the lens galaxy emission rather than genuinely part of the source.
To stop this, we force all source pixels at the edge of the mask to be reconstructed with flux values of 0, which in turn means that the lens light model then correct goes to a slightly brighter solution which correctly fits its outskirts.
Pre-JAX, we could determine edge pixels inside the likelihood function and zero them, but JAX's static array shapes criteria means you have to assign these pixels before modeling. For the rectangular mesh, there is a line of code which passes Preloads the edge pixels of all rectangular pixels.
For image-mesh's and delaunay, what the code abovre does is add a circle of source-pixels to the image mesh which are the pixels to be zeroed. So, you code probably added image_plane_mesh_grid_edge_pixels=30 extra source pixels in the image plane for this purpose.
The bug is that it was adding them at an image-plane radius=3.0 + 0.05, instead of radius = mask_radius + 0.05.
10:31
This perfectly explains one of the very strange results we saw where there was caustics being formed at the outskirts of the source plane, basically the sources pixels a bit further "inside" the source plane were being zero'd, doing all sorts of horrible things to the results source reconstruction (but surprisingly not enough to be really-obviously-broken-by-eye
"""
image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=image_plane_mesh_grid_edge_pixels,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

total_linear_light_profiles = 0

mapper_indices = al.mapper_indices_from(
    total_linear_light_profiles=total_linear_light_profiles,
    total_mapper_pixels=total_mapper_pixels,
)

source_pixel_zeroed_indices = mapper_indices[-image_plane_mesh_grid_edge_pixels:]

preloads = al.Preloads(
    mapper_indices=mapper_indices,
    source_pixel_zeroed_indices=source_pixel_zeroed_indices,
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

"""
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
pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.AdaptiveBrightnessSplit,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=pixelization,
)

model_2 = af.Collection(
    galaxies=af.Collection(lens=result_1.instance.galaxies.lens, source=source)
)

"""
__Analysis (Search 2)__

We now create the analysis for the second search.
"""
analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
)

"""
__Search + Model-Fit (Search 2)__

We now create the non-linear search and perform the model-fit using this model.
"""
search_2 = af.Nautilus(
    path_prefix=Path("features"),
    name="delaunay_adapt",
    unique_tag=dataset_name,
    n_live=75,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
We could perform a third fit where we free all lens model parameters and fit them using the adaptive 
image mesh and regularization.

However, it is better to use all of these features with the Delaunay via the
SLaM pipelines, which we jump to immediately below.

__SLaM Pipelines__

The API above allows you to use adaptive features yourself, and you should go ahead an explore them on datasets you
are familiar with.

However, you may also wish to use the Source, Light and Mass (SLaM) pipelines, which are pipelines that
have been carefully crafted to automate lens modeling of large samples whilst ensuring models of the highest
complexity can be reliably fitted.

These pipelines are built around the use of adaptive features -- for example the Source pipeline comes first so that
these features are set up robustly before more complex lens light and mass models are fitted.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import slam_pipeline

dataset_name = "simple"
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

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam_delaunay",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0


"""
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(dataset=dataset)

# Lens Light

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

# Source Light

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

source_lp_result = slam_pipeline.source_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    lens_bulge=lens_bulge,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__JAX & Preloads__

Setup the Overlay image-mesh and preloads for the SOURCE PIX PIPELINE, following the same
code as earlier in this example.
"""
image_mesh = al.image_mesh.Overlay(shape=(26, 26))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

image_plane_mesh_grid_edge_pixels = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=image_plane_mesh_grid_edge_pixels,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

total_linear_light_profiles = 40

mapper_indices = al.mapper_indices_from(
    total_linear_light_profiles=total_linear_light_profiles,
    total_mapper_pixels=total_mapper_pixels,
)

# Extract the last `image_plane_mesh_grid_edge_pixels` indices, which correspond to the circle edge points we added

source_pixel_zeroed_indices = mapper_indices[-image_plane_mesh_grid_edge_pixels:]

preloads = al.Preloads(
    mapper_indices=mapper_indices,
    source_pixel_zeroed_indices=source_pixel_zeroed_indices,
)

"""
__SOURCE PIX PIPELINE__

The SOURCE PIX PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_lp_result
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    preloads=preloads,
)

source_pix_result_1 = slam_pipeline.source_pix.run_1(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    mesh_init=al.mesh.Delaunay(),
    regularization=al.reg.AdaptiveBrightnessSplit,
)

"""
__SOURCE PIX PIPELINE 2__

The SOURCE PIX PIPELINE 2 is identical to the `slam_start_here.ipynb` example.

This sets up the Hilbert image-mesh and preloads for the second source pixelization
using the same code as earlier in this example.
"""
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)

image_mesh = al.image_mesh.Hilbert(pixels=1000)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

image_plane_mesh_grid_edge_pixels = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=image_plane_mesh_grid_edge_pixels,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

total_linear_light_profiles = 40

mapper_indices = al.mapper_indices_from(
    total_linear_light_profiles=total_linear_light_profiles,
    total_mapper_pixels=total_mapper_pixels,
)

source_pixel_zeroed_indices = mapper_indices[-image_plane_mesh_grid_edge_pixels:]

preloads = al.Preloads(
    mapper_indices=mapper_indices,
    source_pixel_zeroed_indices=source_pixel_zeroed_indices,
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
)

source_pix_result_2 = slam_pipeline.source_pix.run_2(
    settings_search=settings_search,
    analysis=analysis,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.AdaptiveBrightnessSplit,
)

"""
__LIGHT LP PIPELINE__

The LIGHT LP PIPELINE is setup identically to the `slam_start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    preloads=preloads,
)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

light_result = slam_pipeline.light_lp.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    lens_bulge=lens_bulge,
    lens_disk=None,
)

"""
__MASS TOTAL PIPELINE__

The MASS TOTAL PIPELINE is identical to the `slam_start_here.ipynb` example.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[
        source_pix_result_2.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
    ],
    preloads=preloads,
)

mass_result = slam_pipeline.mass_total.run(
    settings_search=settings_search,
    analysis=analysis,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    mass=af.Model(al.mp.PowerLaw),
)

"""
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
    regularization=al.reg.ConstantSplit(coefficient=1.0),
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
    source_plane_data_grid_over_sampled=relocated_grid.over_sampled,
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

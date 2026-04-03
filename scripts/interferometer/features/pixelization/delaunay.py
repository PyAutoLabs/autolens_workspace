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

Generating a Delaunay mesh supports JAX and GPU acceleration, however certain operations (e.g. generating the Delaunay
triangulation itself) do not run on the GPU because they cannot be easily converted to JAX.

Instead, JAX sends them to a CPU, runs them there, and then sends the results back to the GPU. This process is
very efficient, because these operations run very fast on a CPU and the data being sent back and forth is small.
Current benchmarking suggests the Delaunay runs less than twice as long as the same fit using a rectangular mesh,
but scientfically offers better results in many cases.

If you do want to run only on CPU, you can use fast CPU method described in
example `imaging/features/pixelization/cpu_fast_modeling` with the Delaunay mesh.


__Source Science (Magnification, Flux and More)__

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or galaxies).

Using the reconstructed source model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

The example `autolens_workspace/*/guides/source_science` gives a complete overview of how to calculate these quantities,
including examples using a Delaunay source reconstruction. Once you have completed lens modeling using a Delaunay mesh,
you can jump to that example to study the source galaxy.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=True)

positions = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "positions.json"))
)

positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)


settings = al.Settings(use_positive_only_solver=False)

"""
__Image Mesh__

For a Delaunay mesh, the vertices of the triangles are defined by (y, x) coordinates in the image-plane. These
coordinates are then ray-traced into the source-plane for each mass model sampled during the non-linear search.
This `image_plane_mesh_grid` must be computed before lens modeling.

We compute this `image_plane_mesh_grid` using an `Overlay` image-mesh, which places a regular grid of
(y, x) points across the image-plane. This has a mild adaptive effect: regions of high lens magnification receive
more source pixels once they are ray-traced. Later in this example, we switch to a `Hilbert` image-mesh, which adapts
the pixel distribution more strongly to the source’s surface brightness.

The `Delaunay` mesh has an input number of `pixels`, which is the number of source pixels used to reconstruct the 
source. The number of `pixels` must be equal to the number of coordinates in the `image_plane_mesh_grid`. 

Like for the `mesh_shape` rectangular mesh, `pixels` must be fixed for lens modeling because JAX uses the 
number of `pixels` to determine static array shapes. 

To pass the `image_plane_mesh_grid` to the modeling, we use the `AdaptImages` object below, which pairs
the `image_plane_mesh_grid` to the source galaxy. For double source plane lenses, this means we can
attach an `image_plane_mesh_grid` to each source galaxy and use adaptive meshes for each source plane.
"""
image_mesh = al.image_mesh.Overlay(shape=(26, 26))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

"""
__Edge Zeroing__

By default, all pixels at the edge of the mesh in the source-plane are forced to solutions of zero brightness by 
the linear algebra solver. This prevents unphysical solutions where pixels at the edge of the mesh reconstruct 
bright surface brightnesses, often because they fit residuals from the lens light subtraction.

For a rectangular mesh, the source code computes edge pixels internally using the known pixels at the edge of the mesh,
requiring no input from the user. 

For the `Delaunay` mesh, we use the `append_with_circle_edge_points` function to manually setup the Delaunay image 
mesh to include a ring of edge pixels and then input the total number into the mesh to perform zeroing. 

These points are added to the edge of the image-plane mesh, ray-traced to the source-plane during lens modeling, 
included in the Delaunay triangulation but zeroed during the inversion.
"""
edge_pixels_total = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=real_space_mask.mask_centre,
    radius=mask_radius + real_space_mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

"""
__Fit__

In the example `interferometer/features/pixelization/fit.py`, we illustrate how to use a pixelized source
with a rectangular mesh.

Below, we use a Delaunay mesh to perform a fit using the Delaunay source reconstruction.

The API is nearly identical to the rectangular mesh example, noting that the inputs to the `Delaunay` 
mesh are different to the rectangular mesh and use image mesh quantities computed above.
"""
mesh = al.mesh.Delaunay(
    pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
)
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

fit = al.FitInterferometer(
    dataset=dataset,
    tracer=tracer,
    adapt_images=adapt_images,
)

"""
By plotting the fit, we see that the Delaunay source does a good job at capturing the appearance of the source galaxy
using adaptive triangular pixels.
"""
aplt.subplot_fit_interferometer(fit=fit)

"""
__Model__

We now perform lens modeling using the Delaunay pixelization with the Overlay image-mesh.

The code below is a simple adaptive modeling example using the Delaunay mesh, which mirrors the
API used in other pixelization modeling examples.

The example `interferometer/features/pixelization/adaptive.py` illustrates how to use adaptive features to
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
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
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

analysis_1 = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
)

"""
__VRAM__

The `pixelization/modeling` example explains how VRAM use is an important consideration for pixelization models
and how it depends on image resolution, number of source pixels and batch size.

This is true for the Delaunay mesh, therefore we print out the estimated VRAM required for this model-fit.
"""
analysis_1.print_vram_use(model=model_1, batch_size=search_1.batch_size)

"""
__Model-Fit (Search 1)__

Perform the model-fit using this model.
"""
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
galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=result_1, use_model_images=True
)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=result_1)

image_mesh = al.image_mesh.Hilbert(pixels=1000, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

# Repeat edge zeroing set up describe above.

edge_pixels_total = 30

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=real_space_mask.mask_centre,
    radius=mask_radius + real_space_mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
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

 - This pixelization is regularized using a `AdaptSplit` scheme [2 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=4.
"""
pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=al.reg.AdaptSplit,
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
analysis_2 = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
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
"""
__SOURCE LP PIPELINE__

Identical to `slam_start_here.py`, using an MGE for the lens and source light profiles.

Note that unlike the other interferometer SLaM scripts, this Delaunay script does include a source_lp pipeline.
Its result provides adapt images for source_pix_1, which are used to initialise the Delaunay image mesh.
"""
def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    redshift_lens: float,
    redshift_source: float,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisInterferometer(dataset=dataset)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=af.Model(al.mp.Isothermal),
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
            ),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to `slam_start_here.py`, except the source pixelization uses a Delaunay mesh.

The `source_pix_1` search uses an `Overlay` image-mesh to place the initial Delaunay mesh pixels, with
additional edge points added around the mask boundary to ensure full coverage.

Adapt images from the source LP result provide the initial image-plane mesh grid via `AdaptImages`, and
positions from the source LP result constrain the mass model.
"""
def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_lp_result: af.Result,
    settings,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result, use_model_images=True
    )

    image_mesh = al.image_mesh.Overlay(shape=(26, 26))

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
    )

    edge_pixels_total = 30

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=dataset.mask.mask_centre,
        radius=mask_radius + dataset.mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
        settings=settings,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=af.Model(al.mp.Isothermal),
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.Delaunay(
                        pixels=image_plane_mesh_grid.shape[0],
                        zeroed_pixels=edge_pixels_total,
                    ),
                    regularization=al.reg.ConstantSplit,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `slam_start_here.py`, except the source pixelization uses a Delaunay mesh.

The `source_pix_2` search uses a `Hilbert` image-mesh to place the final Delaunay mesh pixels, which adapts
the mesh to the source morphology using the high-quality adapt images from search 1.
"""
def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    settings,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    image_mesh = al.image_mesh.Hilbert(pixels=1000, weight_power=3.5, weight_floor=0.01)

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
    )

    edge_pixels_total = 30

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=dataset.mask.mask_centre,
        radius=mask_radius + dataset.mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        settings=settings,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.Delaunay(
                        pixels=image_plane_mesh_grid.shape[0],
                        zeroed_pixels=edge_pixels_total,
                    ),
                    regularization=al.reg.AdaptSplit,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`, except no lens light model is included as interferometer data does not
contain lens light emission.
"""
def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_pix_result_1: af.Result,
    source_pix_result_2: af.Result,
    settings,
    n_batch: int = 20,
) -> af.Result:
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1, use_model_images=True
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_1.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
        settings=settings,
    )

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_pix_result_1.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    source = al.util.chaining.source_from(result=source_pix_result_2)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_pix_result_1.instance.galaxies.lens.redshift,
                bulge=None,
                disk=None,
                mass=mass,
                shear=source_pix_result_1.model.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("interferometer") / "slam_delaunay",
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
__SLaM Pipeline__

The code below calls the full SLaM PIPELINE. See the documentation string above each Python function for
a description of each pipeline step.
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_lp_result=source_lp_result,
    settings=settings,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    settings=settings,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    settings=settings,
)

"""
__Likelihood Function: Pixelization__

This script provides a step-by-step guide of the **PyAutoLens** `log_likelihood_function` which is used to fit
`Interferometer` data with an inversion (specifically a `Delaunay` mesh and `Constant` regularization scheme`).

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoLens**, so that readers can understand the
 likelihood function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

 - To make inversions in **PyAutoLens** less of a "black-box" to users.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.

__Prerequisites__

The likelihood function of pixelizations is the most complicated likelihood function.

It is advised you read through the following two simpler likelihood functions first, which break down a number of the
concepts used in this script:

 - `interferometer/light_profile/log_likelihood_function.py` the likelihood function for a light profile.
 - `interferometer/linear_light_profile/log_likelihood_function.py` the likelihood function for a linear light profile, which
 introduces the linear algebra used for a pixelization but with a simpler use case.

This script repeats all text and code examples in the above likelihood function examples. It therefore can be used to
learn about the linear light profile likelihood function without reading other likelihood scripts.

__Likelihood Function__

The example `interferometer/pixelization/likelihood_function.py` provides a step-by-step description of how
a likelihood evaluation is performed for interferometer data using a pixelized source reconstruction with a rectangular
mesh.

We now give the same step-by-step description for a pixelized source reconstruction using a Delaunay mesh and
adaptive features.

We only describe code which is specific to Delaunay meshes and adaptive features -- for all other aspects of the likelihood
evaluation, refer to rectangular mesh example.

__Mask__
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=4.0
)

"""
__Dataset__
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

aplt.plot_grid(grid=dataset.grids.pixelization, title="")

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

lens_galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)

"""
__Source Galaxy Pixelization and Regularization__

We combine the pixelization into a single `Galaxy` object.

The galaxy includes the Delaunay mesh and constant regularization scheme, which will ultimately be used
to reconstruct its star forming clumps.

One of the biggest differences between a Delaunay mesh and rectangular mesh is how the centres of the mesh pixels
in the source-plane are computed. 

For the rectangular mesh, the pixel centres are computed by overlaying a uniform grid over the source-plane.

For a Delaunay mesh, the uniform grid is instead laid over the image-plane to create a course grid of (y,x) coordinates.
These are then ray-traced to the source-plane and are used as the vertexes of the Delaunay triangles.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=al.reg.ConstantSplit(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

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
image_mesh = al.image_mesh.Overlay(shape=(30, 30))  # Specific to Delaunay

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={source_galaxy: image_plane_mesh_grid},
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

"""
Plotting this grid shows a sparse grid of (y,x) coordinates within the mask, which will form our source pixel centres.
"""

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
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer, dataset=dataset, adapt_images=adapt_images
)

# A list of every grid (e.g. image-plane, source-plane) however we only need the source plane grid with index -1.
traced_grid_pixelization = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization
)[-1]

# This functions a bit weird - it returns a list of lists of ndarrays. Best not to worry about it for now!
traced_mesh_grid = tracer_to_inversion.traced_mesh_grid_pg_list[-1][-1]


aplt.plot_grid(grid=traced_grid_pixelization, title="")

aplt.plot_grid(grid=traced_mesh_grid, title="")

"""
__Border Relocation__

Coordinates that are ray-traced near the mass profile centres are heavily demagnified and may trace to far outskirts of
the source-plane. 

Border relocation is performed on both the traced image-pixel grid and traced mesh pixels, therefore ensuring that
the vertexes of the Delaunay triangles are not at the extreme outskirts of the source-plane.
"""
from autoarray.inversion.mesh.border_relocator import BorderRelocator

border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)

relocated_grid = border_relocator.relocated_grid_from(grid=traced_grid_pixelization)

relocated_mesh_grid = border_relocator.relocated_mesh_grid_from(
    grid=traced_grid_pixelization, mesh_grid=traced_mesh_grid
)


aplt.plot_grid(grid=relocated_grid, title="")

aplt.plot_grid(grid=relocated_mesh_grid, title="")

"""
__Delaunay Mesh__

The relocated mesh grid is used to create the `Pixelization`'s Delaunay mesh using the `scipy.spatial` library.
"""
interpolator = al.InterpolatorDelaunay(
    mesh=pixelization.mesh,
    mesh_grid=relocated_mesh_grid,
    data_grid=relocated_grid,
)

"""
Plotting the Delaunay mesh shows that the source-plane and been discretized into a grid of irregular Delaunay pixels.

(To plot the Delaunay mesh, we have to convert it to a `Mapper` object, which is described in the next likelihood step).

Below, we plot the Delaunay mesh without the traced image-grid pixels (for clarity) and with them as black dots in order
to show how each set of image-pixels fall within a Delaunay pixel.
"""
mapper = al.Mapper(
    interpolator=interpolator,
    image_plane_mesh_grid=image_plane_mesh_grid,
)

# mapper_plotter.figure_2d()
#
#     grid=mapper.source_plane_data_grid,
# )
# mapper_plotter.figure_2d()

"""
__Interpolation__
"""
pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index

print(pix_indexes_for_sub_slim_index[0:9])


aplt.plot_array(array=dataset.dirty_image, title="Image", positions=mapper.image_plane_data_grid)
aplt.plot_grid(grid=mapper.source_plane_mesh_grid, title="Source-Plane Mesh Grid")

pix_indexes = [[200]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)



aplt.plot_array(array=dataset.dirty_image, title="Image", positions=mapper.image_plane_data_grid)
aplt.plot_grid(grid=mapper.source_plane_mesh_grid, title="Source-Plane Mesh Grid")

mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for Delaunay
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for Delaunay
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=mapper.over_sampler.sub_fraction,
)

plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

indexes_source_pix_200 = np.nonzero(mapping_matrix[:, 200])

print(indexes_source_pix_200[0])

array_2d = al.Array2D(values=mapping_matrix[:, 200], mask=dataset.mask)

aplt.plot_array(array=array_2d, title="")

transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
    mapping_matrix=mapping_matrix
)

plt.imshow(
    transformed_mapping_matrix.real,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(
    transformed_mapping_matrix.imag,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

indexes_pix_200 = np.nonzero(transformed_mapping_matrix[:, 200])

print(indexes_pix_200[0])

visibilities = al.Visibilities(visibilities=transformed_mapping_matrix[:, 200])

aplt.plot_grid(grid=visibilities.in_grid, title="")

print(f"Mapping between visibility 0 and Delaunay pixel 2 = {mapping_matrix[0, 2]}")

data_vector = (
    al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=dataset.data,
        noise_map=dataset.noise_map,
    )
)

plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

print("Data Vector:")
print(data_vector)
print(data_vector.shape)

real_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.real,
    noise_map=dataset.noise_map.real,
)

imag_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.imag,
    noise_map=dataset.noise_map.imag,
)

curvature_matrix = np.add(real_curvature_matrix, imag_curvature_matrix)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

source_pixel_0 = 0
source_pixel_1 = 1

print(curvature_matrix[source_pixel_0, source_pixel_1])

visibilities = al.Visibilities(
    visibilities=transformed_mapping_matrix[:, source_pixel_0],
)

aplt.plot_grid(grid=visibilities.in_grid, title="")

visibilities = al.Visibilities(
    visibilities=transformed_mapping_matrix[:, source_pixel_1],
)

aplt.plot_grid(grid=visibilities.in_grid, title="")

regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=source_galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.neighbors,
    neighbors_sizes=mapper.neighbors.sizes,
)

plt.imshow(regularization_matrix)
plt.colorbar()
plt.show()
plt.close()

curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)


aplt.plot_grid(grid=mapper.source_plane_mesh_grid, title="Source-Plane Mesh Grid")

mapped_reconstructed_visibilities = (
    al.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        reconstruction=reconstruction,
    )
)

mapped_reconstructed_visibilities = al.Visibilities(
    visibilities=mapped_reconstructed_visibilities
)

aplt.plot_grid(grid=mapped_reconstructed_visibilities.in_grid, title="")


"""
__Likelihood Function__
"""
model_visibilities = mapped_reconstructed_visibilities

residual_map = dataset.data - model_visibilities


normalized_residual_map_real = (residual_map.real / dataset.noise_map.real).astype(
    "complex128"
)
normalized_residual_map_imag = (residual_map.imag / dataset.noise_map.imag).astype(
    "complex128"
)
normalized_residual_map = (
    normalized_residual_map_real + 1j * normalized_residual_map_imag
)


chi_squared_map_real = (residual_map.real / dataset.noise_map.real) ** 2
chi_squared_map_imag = (residual_map.imag / dataset.noise_map.imag) ** 2
chi_squared_map = chi_squared_map_real + 1j * chi_squared_map_imag


chi_squared_real = np.sum(chi_squared_map.real)
chi_squared_imag = np.sum(chi_squared_map.imag)
chi_squared = chi_squared_real + chi_squared_imag

print(chi_squared)

chi_squared_map = al.Visibilities(visibilities=chi_squared_map)

aplt.plot_grid(grid=chi_squared_map.in_grid, title="")

regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

noise_normalization_real = np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
noise_normalization_imag = np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
noise_normalization = noise_normalization_real + noise_normalization_imag

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

This process to perform a likelihood function evaluation performed via the `FitInterferometer` object.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(
    dataset=dataset,
    tracer=tracer,
    adapt_images=adapt_images,
    settings=al.Settings(use_border_relocator=True),
)
fit_log_evidence = fit.log_evidence
print(fit_log_evidence)

aplt.subplot_fit_interferometer(fit=fit)

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
multiple MCMC and optimization algorithms are supported.

__Log Likelihood Function: Source Code Speed Up__

The interferometer pixelization likelihood function described in this notebook performs certain calculations using
functions which are easier to understand, but are computationally slower than the actual source code implementation
(but the two produce identical results).

We end by pointing out some of these, but we do not provide an step-by-step description of how they work.
If you are interested, you will need to dive into the source code itself.

**Fast Chi Squared:**  The `chi_squared` above is computed using the `transformed_mapping_matrix`, which requires
many NUFFT's to compute and requires large memroy store. The source code uses a trick which computes the chi-squared
but bypasses the need to ever compute the `transformed_mapping_matrix`.

**Sparse Operator Curvature Matrix:** The `curvature_matrix` above is also computed using the `transformed_mapping_matrix`, 
which again means slow run times and large memory usage. The source code can instead use sparse operators to 
compute the curvature matrix in a way which again bypasses the need to compute the `transformed_mapping_matrix`.

The two tricks in combination lead to a significant speed up in the likelihood function evaluation and mean that
the large matrix of size [source pixels, visibilities] never needs to be stored in memory. This is at the heart
of why lens modeling interferometer data with pixelized source reconstructions is so fast!

__Wrap Up__

We have presented a visual step-by-step guide to the pixelization likelihood function.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in this package. In brief, these describe:

 - **Over Sampling**: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 paired fractionally with each Delaunay pixel.

 - **Source-plane Interpolation**: Using bilinear interpolation on the Delaunay pixelization to pair each 
 image (sub-)pixel to multiple Delaunay pixels with interpolation weights.

 - **Luminosity Weighted Regularization**: Using an adaptive regularization coefficient which adapts the level of 
 regularization applied to the source galaxy based on its luminosity.
"""

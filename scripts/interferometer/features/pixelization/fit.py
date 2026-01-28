"""
Features: Pixelization
======================

A pixelization reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a source galaxy model which uses a pixelization to reconstruct the source's light.

A rectangular mesh which adapts to the lens mass model magnification and constant regularization scheme are used, which
are the simplest forms of mesh and regularization with provide computationally fast and accurate solutions.

For simplicity, the lens galaxy's light is omitted from the model and is not present in the simulated data. It is
straightforward to include the lens galaxy's light in the model.

pixelizations are covered in detail in chapter 4 of the **HowToLens** lectures.

__JAX GPU Run Times__

Throughout the workspace, it has been emphasised that pixelized source reconstructions are computed using GPU or CPU
via JAX, where the linear algebra fully exploits sparsity in a way which minimizes VRAM use. This example uses
this functionality, and therefore is suitable for datasets with a low number of visibilities (e.g. < 10000) or
many visibilities (E.g. tens of millions).

This example fits the dataset with 273 visibilities used throughout the workspace, so the fit runs in seconds, but
provided the w tilde formalism is set up correctly, the same code can be used to fit datasets with millions of
visibilities.

If your dataset contains many visibilities (e.g. millions), setting up the matrices for pixelized source reconstruction
which speed up the linear algebra may take tens of minutes, or hours. Once you are comfortable with the API introduced
in this example, the `feature/pixelization/many_visibilities_preparation` explains how this initial setup can be
performed before lens modeling and saved to hard disk for fast loading before the model fit.

This script's default setup uses an adaptive 20 x 20 rectangular mesh (400 pixels), which is relatively low resolution
and may not provide the most accurate lens modeling results. The mesh resolution can be increased to improve
the fit, and the w-tilde formalism means this should still run fine on my laptop GPUs, requiring less than 4 GB VRAm.

CPU run times are also fast using the w-tilde formalism.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**Dataset & Mask:** Standard set up of interferometer dataset that is fitted.
**JAX & Preloads**: Preloading certain arrays for the pixelization's linear algebra, such that JAX knows their shapes in advance.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Fit:** Perform a fit to a dataset using a pixelization, and visualize its results.
**Interpolated Source:** Interpolate the source reconstruction from an irregular Voronoi mesh to a uniform square grid and output to a .fits file.
**Reconstruction CSV:** Output the source reconstruction to a .csv file, which can be used to perform calculations on the source reconstruction.
**Result (Advanced):** API for various pixelization outputs (magnifications, mappings) which requires some polishing.
**Simulate (Advanced):** Simulating a strong lens dataset with the inferred pixelized source.

__Advantages__

Many strongly lensed source galaxies exhibit complex, asymmetric, and irregular morphologies. Such structures
cannot be well approximated by analytic light profiles such as a Sérsic profile, or even combinations of multiple
Sérsic components. pixelizations are therefore required to accurately reconstruct this irregular source-plane light.

Even alternative basis-function approaches, such as shapelets or multi-Gaussian expansions, struggle to accurately
reconstruct sources with highly complex morphologies or multiple distinct source galaxies.

Pixelized source models are also essential for robustly constraining detailed components of the lens mass
distribution (e.g. the mass density slope or the presence of dark matter substructure). By fitting all of the lensed
source light, they reduce degeneracies between the source and lens mass model.

Finally, many science applications aim to study the highly magnified source galaxy itself, in order to learn about
distant and intrinsically faint galaxies. pixelizations reconstruct the unlensed source emission, enabling detailed
studies of the source-plane structure.

For CCD imaging, a disadvantage of pixelized source reconstructions is they are the most computationally expensive
modeling approach. However, for interferometer datasets, the way that JAX and GPUs can exploit the sparsity in the
linear algebra means pixelized source reconstructions are both significantly faster than other approaches (E.g.
light profiles) and can scale to millions of visibilities.

__Disadvantages__

Lens modeling with pixelizations is conceptually more complex. There are additional failure modes, such as
solutions where the source is reconstructed in a highly demagnified configuration due to an unphysical lens mass
model (e.g. too little or too much mass). These issues are discussed in detail later in the workspace.

As a result, learning to successfully fit lens models with pixelizations typically requires more time and experience
than the simpler modeling approaches introduced elsewhere in the workspace.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This could be problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelizaiton, this often produces negative source pixels which over-fit
the data, producing unphysical solutions.

For CCD imaging datsets pixelized source reconstructions use a positive-only solver, meaning that every source-pixel
is only allowed to reconstruct positive flux values. This ensures that the source reconstruction is physical and
that we don't reconstruct negative flux values that don't exist in the real source galaxy (a common systematic
solution in lens analysis).

However, for interferometer datasets this positive-only solver is often disabled, because negative pixel values
can be observed from the measurement process. All interferometer examples therefore disable the positive only solver,
but you may want to consider if using the positive-only solver is appropriate for your dataset.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)


"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files, which we will fit 
with the lens model.

This includes the method used to Fourier transform the real-space image of the strong lens to the uv-plane and compare 
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities.

If you want to use the high resolution ALMA dataset, uncomment the relevant lines of code below after downloading
the data from the repository described in the "High Resolution Dataset" section above.
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

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__W_Tilde__

Pixelized source modeling requires heavy linear algebra operations. These calculations are greatly accelerated
using an alternative mathematical approach called the **w_tilde formalism**.

You do not need to understand the full details of the method, but the key point is:

- `w_tilde` exploits the **sparsity** of the matrices used in pixelized source reconstruction.
- This leads to a **significant speed-up on GPU or CPU**, using JAX to perform the linear algebra calculations.

To enable this feature, we call `apply_w_tilde()` on the dataset. This computes and stores a `w_tilde_preload` matrix,
which reused in all subsequent pixelized source fits.

For datasets with over 100000 visibilities and many pixels in their real-space mask, this computation
can take 10 minutes or hours (for the small dataset loaded above its miliseconds). The `show_progress` input outputs 
a progress bar to the terminal so you can monitor the computation, which is useful when it is slow

When computing it is slow, it is recommend you compute it once, save it to hard-disk, and load it
before modeling. The example `pixelization/many_visibilities_preparation.py` illustrates how to do this.
"""
dataset = dataset.apply_w_tilde(use_jax=True, show_progress=True)

"""
__Settings__

As discussed above, disable the default position only linear algebra solver so the source
reconstruction can have negative pixel values.
"""
settings_inversion = al.SettingsInversion(use_positive_only_solver=False)

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling is used, 
which evaluates light profiles on a higher resolution grid than the image data to ensure the calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.

__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data, primarily because the lens light is omitted.

- `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
  are forced to values of zero, a technique tests have shown are required to give accruate lens models.
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
__Pixelization__

We create a `Pixelization` object to perform the pixelized source reconstruction, which is made up of three
components:

- `mesh:` Different types of mesh can be used to perform the source reconstruction, where the mesh changes the
details of how the source is reconstructed (e.g. interpolation weights). In this example, we use a rectangular mesh,
where the centres computed by overlayiong a rectangular mesh over the source plane.

- `regularization:` A pixelization uses many pixels to reconstructed the source, which will often lead to over fitting
of the noise in the data and an unrealistically complex and structured source. Regularization smooths the source
reconstruction solution by penalizing solutions where neighboring pixels have large flux differences.
"""
mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

"""
__Fit__

This is to illustrate the API for performing a fit via a pixelization using standard objects like 
the `Galaxy`, `Tracer` and `FitInterferometer` 

We simply create a `Pixelization` and pass it to the source galaxy, which then gets input into the tracer.
"""
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

fit = al.FitInterferometer(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
    settings_inversion=settings_inversion,
)

"""
By plotting the fit, we see that the pixelized source does a good job at capturing the appearance of the source galaxy
and fitting the data to roughly the noise level.
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

"""
Pixelizations have bespoke visualizations which show more details about the source-reconstruction, image-mesh
and other quantities.

These plots use an `InversionPlotter`, which gets its name from the internals of how pixelizations are performed in
the source code, where the linear algebra process which computes the source pixel fluxes is called an inversion.

The `subplot_mappings` overlays colored circles in the image and source planes that map to one another, thereby
allowing one to assess how the mass model ray-traces image-pixels and therefore to assess how the source reconstruction
maps to the image.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.subplot_of_mapper(mapper_index=0)
inversion_plotter.subplot_mappings(pixelization_index=0)

"""
The inversion can be extracted directly from the fit the perform these plots, which we also use below
for various calculations
"""
inversion = fit.inversion

inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)

"""
__Source Science (Magnification, Flux and More)__

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or galaxies).

Using the reconstructed source model, we can compute key quantities such as the magnification, total flux, and intrinsic 
size of the source.

The example `autolens_workspace/*/guides/source_science` gives a complete overview of how to calculate these quantities,
including examples using a pixelized source reconstruction. Now you know how to fit a pixelization, go check it out!

__Wrap Up__

Pixelizations are the most complex but also most powerful way to model a source galaxy.

Whether you need to use them or not depends on the science you are doing. If you are only interested in measuring a
simple quantity like the Einstein radius of a lens, you can get away with using light profiles like a Sersic, MGE or 
shapelets to model the source. Low resolution data also means that using a pixelization is not necessary, as the
complex structure of the source galaxy is not resolved anyway.

However, fitting complex mass models (e.g. a power-law, stellar / dark model or dark matter substructure) requires 
this level of complexity in the source model. Furthermore, if you are interested in studying the properties of the
source itself, you won't find a better way to do this than using a pixelization.

__Linear Objects__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `Voronoi` mesh).

In this example, the only linear object used to fit the data was a `Pixelization`, thus the `linear_obj_list`
contains just one entry corresponding to a `Mapper`:
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities will come in lists or require that we specific the linear object
we with to use. 

Thus, knowing what linear objects are contained in the `linear_obj_list` and what indexes they correspond to
is important.
"""
print(f"Mapper = {inversion.linear_obj_list[0]}")

"""
__Grids__

The role of a mapper is to map between the image-plane and source-plane. 

This includes mapping grids corresponding to the data grid (e.g. the centers of each image-pixel in the image and
source plane) and the pixelization grid (e.g. the centre of the Delaunay triangulation in the image-plane and 
source-plane).

All grids are available in a mapper via its `mapper_grids` property.
"""
mapper = inversion.linear_obj_list[0]

# Centre of each masked image pixel in the image-plane.
print(mapper.mapper_grids.image_plane_data_grid)

# Centre of each source pixel in the source-plane.
print(mapper.mapper_grids.source_plane_data_grid)

# Centre of each pixelization pixel in the image-plane (the `Overlay` image_mesh computes these in the image-plane
# and maps to the source-plane).
print(mapper.mapper_grids.image_plane_mesh_grid)

# Centre of each pixelization pixel in the source-plane.
print(mapper.mapper_grids.source_plane_mesh_grid)

"""
__Reconstruction__

The source reconstruction is also available as a 1D numpy array of values representative of the source pixelization
itself (in this example, the reconstructed source values at the vertexes of each Voronoi triangle).
"""
print(inversion.reconstruction)

"""
The (y,x) grid of coordinates associated with these values is given by the `Inversion`'s `Mapper` (which are 
described in chapter 4 of **HowToLens**.
"""
mapper = inversion.linear_obj_list[0]
print(mapper.source_plane_mesh_grid)

"""
The mapper also contains the (y,x) grid of coordinates that correspond to the ray-traced image sub-pixels.
"""
print(mapper.source_plane_data_grid)

"""
__Mapped Reconstructed Images__

The source reconstruction(s) are mapped to the image-plane in order to fit the lens model.

These mapped reconstructed images are also accessible via the `Inversion`. 

Note that any light profiles in the lens model (e.g. the `bulge` and `disk` of a lens galaxy) are not 
included in this image -- it only contains the source.
"""
print(inversion.mapped_reconstructed_image.native)

"""
__Linear Algebra Matrices (Advanced)__

To perform an `Inversion` a number of matrices are constructed which use linear algebra to perform the reconstruction.

These are accessible in the inversion object.
"""
print(inversion.curvature_matrix)
print(inversion.regularization_matrix)
print(inversion.curvature_reg_matrix)

"""
__Evidence Terms (Advanced)__

In **HowToLens** and the papers below, we cover how an `Inversion` uses a Bayesian evidence to quantify the goodness
of fit:

https://arxiv.org/abs/1708.07377
https://arxiv.org/abs/astro-ph/0601493

This evidence balances solutions which fit the data accurately, without using an overly complex regularization source.

The individual terms of the evidence and accessed via the following properties:
"""
print(inversion.regularization_term)
print(inversion.log_det_regularization_matrix_term)
print(inversion.log_det_curvature_reg_matrix_term)

"""
__Simulated Interferometer__

We load the source galaxy image from the pixelized inversion of a previous fit, which was performed on an irregular mesh.  

Since irregular meshes cannot be directly used to simulate lensed images, we interpolate the source onto a uniform 
grid with shape `interpolated_pixelized_shape`. This grid should have a high resolution (e.g., 1000 × 1000) to preserve 
all resolved structure from the original mesh.  
"""
from scipy.interpolate import griddata

interpolation_grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

reconstruction = inversion.reconstruction
source_plane_mesh_grid = mapper.mapper_grids.source_plane_mesh_grid

interpolated_reconstruction = griddata(
    points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid
)

# As a pure 2D numpy array in case its useful for calculations
interpolated_reconstruction_ndarray = interpolated_reconstruction.reshape(
    interpolation_grid.shape_native
)

interpolated_reconstruction = al.Array2D.no_mask(
    values=interpolated_reconstruction_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

"""
To create the lensed image, we ray-trace image pixels to the source plane and interpolate them onto the source 
galaxy image.  

This requires an image-plane grid of (y, x) coordinates. In this example, we use the real space mask grid.

To ensure accurate ray-tracing, we apply an 8×8 oversampling scheme. This means that for each pixel in the 
image-plane grid, an 8×8 sub-pixel grid is ray-traced. This approach fully resolves how light is distributed 
across each simulated image pixel, given the source pixelization.
"""
grid = al.Grid2D.uniform(
    shape_native=real_space_mask.shape_native,
    pixel_scales=real_space_mask.pixel_scales,
    over_sample_size=8,
)

"""
We create a tracer to generate the lensed grid onto which we overlay the interpolated source galaxy image, 
producing the lensed source galaxy image.  

The source-plane requires a source galaxy with a defined `redshift` for the tracer to function. Since the source’s 
emission is entirely determined by the source galaxy image, this galaxy has no light profiles.
"""
tracer = al.Tracer(
    galaxies=[
        lens,
        al.Galaxy(redshift=source.redshift),
    ]
)

"""
Using the tracer, we generate the lensed source galaxy image on the image-plane grid. This process incorporates 
the `interpolated_reconstruction`, preserving the irregular and asymmetric morphological features captured by the source reconstruction.  

Next, we configure the grid, PSF, and simulator settings to match the signal-to-noise ratio (S/N) and noise properties 
of the observed data used for sensitivity mapping.  

The `SimulatorInterferometer` takes the generated strong lens image and convolves it with the PSF before adding noise. To 
prevent edge effects, the image is padded before convolution and then trimmed to restore its original `shape_native`.
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=dataset.uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerDFT,
)

# dataset = simulator.via_interpolated_reconstruction_from(
#     tracer=tracer, grid=grid, interpolated_reconstruction=interpolated_reconstruction
# )
#
# plotter = aplt.InterferometerPlotter(dataset=dataset)
#
# output = aplt.Output(path=".", filename="interpolated_reconstruction", format="png")
#
# plotter = aplt.InterferometerPlotter(
#     dataset=dataset, mat_plot_2d=aplt.MatPlot2D(output=output)
# )

"""
__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More magnification calculations.
- Source gradient calculations.
- A calculation which shows differential lensing effects (e.g. magnification across the source plane).
"""

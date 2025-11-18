"""
Features: Pixelization
======================

A pixelization reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a source galaxy model which uses a pixelization to reconstruct the source's light.

A rectangular mesh which adapts to the lens mass model magnification and constant regularization scheme are used, which
are the simplest forms of mesh and regularization with provide computationally fast and accurate solutions in **PyAutoLens**.

For simplicity, the lens galaxy's light is omitted from the model and is not present in the simulated data. It is
straightforward to include the lens galaxy's light in the model.

Pixelizations are covered in detail in chapter 4 of the **HowToLens** lectures.

__JAX GPU Run Times__

Pixelizations run time depends on how modern GPU hardware is. GPU acceleration only provides fast run times on
modern GPUs with large amounts of VRAM, or when the number of pixels in the mesh are low (e.g. < 500 pixels).

This script's default setup uses an adaptive 20 x 20 rectangular mesh (400 pixels), which is relatively low resolution
and may not provide the most accurate lens modeling results. On most GPU hardware it will run in ~ 10 minutes,
however if your laptop has a large VRAM (GPU > 20 GB) or you can access a GPU cluster with better hardware you should use these
to perform modeling with increased mesh resolution.

__CPU Run Times__

JAX is not natively designed to provide significant CPU speed up, therefore users using CPUs to perform pixelization
analysis will not see fast run times using JAX (unlike GPUs).

The example `pixelization/cpu` shows how to set up a pixelization to use efficient CPU calculations via the library
`numba`.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**JAX & Preloads**: Preloading certain arrays for the pixelization's linear algebra, such that JAX knows their shapes in advance.
**Pixelization:** How to create a pixelization, including a description of its inputs.
**Fit:** Perform a fit to a dataset using a pixelization, and visualize its results.
**Interpolated Source:** Interpolate the source reconstruction from an irregular Voronoi mesh to a uniform square grid and output to a .fits file.
**Reconstruction CSV:** Output the source reconstruction to a .csv file, which can be used to perform calculations on the source reconstruction.
**Result (Advanced):** API for various pixelization outputs (magnifications, mappings) which requires some polishing.
**Simulate (Advanced):** Simulating a strong lens dataset with the inferred pixelized source.

__Advantages__

Many strongly lensed source galaxies are complex, and have asymmetric and irregular morphologies. These morphologies
cannot be well approximated by a light profiles like a Sersic, or many Sersics, and thus a pixelization
is required to reconstruct the source's irregular light.

Even basis functions like shapelets or a multi-Gaussian expansion cannot reconstruct a source-plane accurately
if there are multiple source galaxies, or if the source galaxy has a very complex morphology.

To infer detailed components of a lens mass model (e.g. its density slope, whether there's a dark matter subhalo, etc.)
then pixelized source models are required, to ensure the mass model is fitting all of the lensed source light.

There are also many science cases where one wants to study the highly magnified light of the source galaxy in detail,
to learnt about distant and faint galaxies. A pixelization reconstructs the source's unlensed emission and thus
enables this.

__Disadvantages__

Pixelizations are computationally slow and run times are typically longer than a parametric source model. It is not
uncommon for lens models using a pixelization to take hours or even days to fit high resolution imaging
data (e.g. Hubble Space Telescope imaging).

Lens modeling with pixelizations is also more complex than parametric source models, with there being more things
that can go wrong. For example, there are solutions where a demagnified version of the lensed source galaxy is
reconstructed, using a mass model which effectively has no mass or too much mass. These are described in detail below.

It will take you longer to learn how to successfully fit lens models with a pixelization than other methods illustrated
in the workspace!

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For a pixelizaiton, this often produces negative source pixels which over-fit
the data, producing unphysical solutions.

All pixelized source reconstructions use a positive-only solver, meaning that every source-pixel is only allowed
to reconstruct positive flux values. This ensures that the source reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real source galaxy (a common systematic solution in lens
analysis).

It may be surprising to hear that this is a feature worth pointing out, but it turns out setting up the linear algebra
to enforce positive reconstructions is difficult to make efficient. A lot of development time went into making this
possible, where a bespoke fast non-negative linear solver was developed to achieve this.

Other methods in the literature often do not use a positive only solver, and therefore suffer from these
unphysical solutions, which can degrade the results of lens model in general.
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
__Dataset__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

A pixelization uses a separate grid for ray tracing, with its own over sampling scheme, which below we set to a 
uniform grid of values of 2. 

The pixelization only reconstructs the source galaxy, therefore the adaptive over sampling used for the lens galaxy's 
light in other examples is not applied to the pixelization. 

This example does not model lens light, for examples which combine lens light and a pixelization both over sampling 
schemes should be used, with the lens light adaptive and the pixelization uniform.

Note that the over sampling is input into the `over_sample_size_pixelization` because we are using a `Pixelization`.
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=4,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
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

The `image_mesh` can be ignored, it is legacy API from previous versions which may or may not be reintegrated in future
versions.
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
__Pixelization__

We create a `Pixelization` object to perform the pixelized source reconstruction, which is made up of three
components:

- `image_mesh:`The coordinates of the mesh used for the pixelization need to be defined. The way this is performed
depends on pixelization used. In this example, we define the source pixel centers by overlaying a uniform regular grid
in the image-plane and ray-tracing these coordinates to the source-plane. Where they land then make up the coordinates
used by the mesh.

- `mesh:` Different types of mesh can be used to perform the source reconstruction, where the mesh changes the
details of how the source is reconstructed (e.g. interpolation weights). In this exmaple, we use a `Voronoi` mesh,
where the centres computed via the `image_mesh` are the vertexes of every `Voronoi` triangle.

- `regularization:` A pixelization uses many pixels to reconstructed the source, which will often lead to over fitting
of the noise in the data and an unrealistically complex and strucutred source. Regularization smooths the source
reconstruction solution by penalizing solutions where neighboring pixels (Voronoi triangles in this example) have
large flux differences.
"""
mesh = al.mesh.RectangularMagnification(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(
    image_mesh=image_mesh, mesh=mesh, regularization=regularization
)

"""
__Fit__

This is to illustrate the API for performing a fit via a pixelization using standard autolens objects like 
the `Galaxy`, `Tracer` and `FitImaging` 

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

fit = al.FitImaging(
    dataset=dataset,
    tracer=tracer,
    preloads=preloads,
)

"""
By plotting the fit, we see that the pixelized source does a good job at capturing the appearance of the source galaxy
and fitting the data to roughly the noise level.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

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
__Mask Extra Galaxies__

There may be extra galaxies nearby the lens and source galaxies, whose emission blends with the lens and source.

If their emission is significant, and close enough to the lens and source, we may simply remove it from the data
to ensure it does not impact the model-fit. A standard masking approach would be to remove the image pixels containing
the emission of these galaxies altogether. This is analogous to what the circular masks used throughout the examples
does.

For fits using a pixelization, masking regions of the image in a way that removes their image pixels entirely from
the fit. This can produce discontinuities in the pixelixation used to reconstruct the source and produce unexpected
systematics and unsatisfactory results. In this case, applying the mask in a way where the image pixels are not
removed from the fit, but their data and noise-map values are scaled such that they contribute negligibly to the fit,
is a better approach.

We illustrate the API for doing this below, using the `extra_galaxies` dataset which has extra galaxies whose emission
needs to be removed via scaling in this way. We apply the scaling and show the subplot imaging where the extra
galaxies mask has scaled the data values to zeros, increasing the noise-map values to large values and in turn made
the signal to noise of its pixels effectively zero.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=Path(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=0.1,
    invert=True,  # Note that we invert the mask here as `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=0.1, centre=(0.0, 0.0), radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
We do not explictly fit this data, for the sake of brevity, however if your data has these nearby galaxies you should
apply the mask as above before fitting the data.

__Pixelization / Mapper Calculations__

The pixelized source reconstruction output by an `Inversion` is often on an irregular grid (e.g. a
Voronoi triangulation or Voronoi mesh), making it difficult to manipulate and inspect after the lens modeling has
completed.

Internally, the inversion stores a `Mapper` object to perform these calculations, which effectively maps pixels
between the image-plane and source-plane.

After an inversion is complete, it has computed values which can be paired with the `Mapper` to perform calculations,
most notably the `reconstruction`, which is the reconstructed source pixel values.

By inputting the inversions's mapper and a set of values (e.g. the `reconstruction`) into a `MapperValued` object, we
are provided with all the functionality we need to perform calculations on the source reconstruction.

We set up the `MapperValued` object below, and illustrate how we can use it to interpolate the source reconstruction
to a uniform grid of values, perform magnification calculations and other tasks.
"""
mapper = inversion.cls_list_from(cls=al.AbstractMapper)[
    0
]  # Only one source-plane so only one mapper, would be a list if multiple source planes

mapper_valued = al.MapperValued(
    mapper=mapper, values=inversion.reconstruction_dict[mapper]
)

"""
__Interpolated Source__

A simple way to inspect the source reconstruction is to interpolate its values from the irregular
pixelization o a uniform 2D grid of pixels.

(if you do not know what the `slim` and `native` properties below refer too, it
is described in the `results/examples/data_structures.py` example.)

We interpolate the Voronoi triangulation this source is reconstructed on to a 2D grid of 401 x 401 square pixels.
"""
interpolated_reconstruction = mapper_valued.interpolated_array_from(
    shape_native=(401, 401)
)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(interpolated_reconstruction.slim)

plotter = aplt.Array2DPlotter(
    array=interpolated_reconstruction,
)
plotter.figure_2d()

"""
By inputting the arc-second `extent` of the source reconstruction, the interpolated array will zoom in on only these
regions of the source-plane. The extent is input via the notation (xmin, xmax, ymin, ymax), therefore  unlike the standard
API it does not follow the (y,x) convention.

Note that the output interpolated array will likely therefore be rectangular, with rectangular pixels, unless
symmetric y and x arc-second extents are input.
"""
interpolated_reconstruction = mapper_valued.interpolated_array_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_reconstruction.slim)

"""
The interpolated errors on the source reconstruction can also be computed, in case you are planning to perform
model-fitting of the source reconstruction.
"""
mapper_valued_errors = al.MapperValued(
    mapper=mapper, values=inversion.reconstruction_noise_map_dict[mapper]
)

interpolated_errors = mapper_valued_errors.interpolated_array_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_errors.slim)

"""
__Magnification__

The magnification of the lens model and source reconstruction can also be computed via the `MapperValued` object,
provided we pass it the reconstruction as the `values`.

This magnification is the ratio of the surface brightness of image in the image-plane over the surface brightness
of the source in the source-plane.

In the image-plane, this is computed by mapping the reconstruction to the image, summing all reconstructed
values and multiplying by the area of each image pixel. This image-plane image is not convolved with the
PSF, as the source plane reconstruction is a non-convolved image.

In the source-plane, this is computed by interpolating the reconstruction to a regular grid of pixels, for
example a 2D grid of 401 x 401 pixels, and summing the reconstruction values multiplied by the area of each
pixel. This calculation uses interpolation to compute the source-plane image.

The calculation is relatively stable, but depends on subtle details like the resolution of the source-plane
pixelization and how exactly the interpolation is performed.
"""
mapper_valued = al.MapperValued(
    mapper=mapper,
    values=inversion.reconstruction_dict[mapper],
)

print("Magnification via Interpolation:")
print(mapper_valued.magnification_via_interpolation_from(shape_native=(401, 401)))

"""
The magnification calculated above used an interpolation of the source-plane reconstruction to a 2D grid of 401 x 401
pixels.

For a `RectangularMagnification` or `Voronoi` pixelization, the magnification can also be computed using the source-plane mesh
directly, where the areas of the mesh pixels themselves are used to compute the magnification. In certain situations
this is more accurate than interpolation, especially when the source-plane pixelization is irregular. However,
it does not currently work for the `Delanuay` pixelization and is commented out below.
"""
# print("Magnification via Mesh:")
# print(mapper_valued.magnification_via_mesh_from())

"""
The magnification value computed can be impacted by faint source pixels at the edge of the source reconstruction.

The input `mesh_pixel_mask` can be used to remove these pixels from the calculation, such that the magnification
is based only on the brightest regions of the source reconstruction.

We create a source-plane signal-to-noise map and use this to create a mask that removes all pixels with
a signal-to-noise < 5.0.
"""
reconstruction = inversion.reconstruction_dict[mapper]
errors = inversion.reconstruction_noise_map_dict[mapper]

signal_to_noise_map = reconstruction / errors

mesh_pixel_mask = signal_to_noise_map < 5.0

mapper_valued = al.MapperValued(
    mapper=mapper,
    values=inversion.reconstruction_dict[mapper],
    mesh_pixel_mask=mesh_pixel_mask,
)

print("Magnification via Interpolation:")
print(mapper_valued.magnification_via_interpolation_from(shape_native=(401, 401)))

"""
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
print(f"Voronoi Mapper = {inversion.linear_obj_list[0]}")

"""
__Grids__

The role of a mapper is to map between the image-plane and source-plane. 

This includes mapping grids corresponding to the data grid (e.g. the centers of each image-pixel in the image and
source plane) and the pixelization grid (e.g. the centre of the Voronoi triangulation in the image-plane and 
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
__Mapped To Source__

Mapping can also go in the opposite direction, whereby we input an image-plane masked 2D array and we use 
the `Inversion` to map these values to the source-plane.

This creates an array which is analogous to the `reconstruction` in that the values are on the source-plane 
pixelization grid, however it bypass the linear algebra and inversion altogether and simply computes the sum of values 
mapped to each source pixel.

[CURRENTLY DOES NOT WORK, BECAUSE THE MAPPING FUNCTION NEEDS TO INCORPORATE THE VARYING VORONOI PIXEL AREA].
"""
mapper_list = inversion.cls_list_from(cls=al.AbstractMapper)

image_to_source = mapper_list[0].mapped_to_source_from(array=dataset.data)

mapper_plotter = aplt.MapperPlotter(mapper=mapper_list[0])
mapper_plotter.plot_source_from(pixel_values=image_to_source)

"""
We can interpolate these arrays to output them to fits.

Although the model-fit used a Voronoi mesh, there is no reason we need to use this pixelization to map the image-plane
data onto a source-plane array.

We can instead map the image-data onto a rectangular pixelization, which has the nice property of giving us a
regular 2D array of data which could be output to .fits format.

[NOT CLEAR IF THIS WORKS YET, IT IS UNTESTED!].
"""
mesh = al.mesh.RectangularMagnification(shape=(50, 50))

source_plane_grid = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization)[1]

mapper_grids = mesh.mapper_grids_from(
    mask=mask, source_plane_data_grid=source_plane_grid
)
mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=al.reg.Constant(coefficient=1.0),
)

image_to_source = mapper.mapped_to_source_from(array=dataset.data)

mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.plot_source_from(pixel_values=image_to_source)

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
__Simulated Imaging__

We load the source galaxy image from the pixelized inversion of a previous fit, which was performed on an irregular 
RectangularMagnification or Voronoi mesh.  

Since irregular meshes cannot be directly used to simulate lensed images, we interpolate the source onto a uniform 
grid with shape `interpolated_pixelized_shape`. This grid should have a high resolution (e.g., 1000 × 1000) to preserve 
all resolved structure from the original RectangularMagnification or Voronoi mesh.  
"""
mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]

mapper_valued = al.MapperValued(
    mapper=mapper,
    values=inversion.reconstruction_dict[mapper],
)

source_image = mapper_valued.interpolated_array_from(
    shape_native=(1000, 1000),
)

"""
To create the lensed image, we ray-trace image pixels to the source plane and interpolate them onto the source 
galaxy image.  

This requires an image-plane grid of (y, x) coordinates. In this example, we use a grid with the same 
resolution as the `Imaging` dataset, but without applying a mask.  

To ensure accurate ray-tracing, we apply an 8×8 oversampling scheme. This means that for each pixel in the 
image-plane grid, an 8×8 sub-pixel grid is ray-traced. This approach fully resolves how light is distributed 
across each simulated image pixel, given the source pixelization.
"""
grid = al.Grid2D.uniform(
    shape_native=mask.shape_native,
    pixel_scales=mask.pixel_scales,
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
the `source_image`, preserving the irregular and asymmetric morphological features captured by the source reconstruction.  

Next, we configure the grid, PSF, and simulator settings to match the signal-to-noise ratio (S/N) and noise properties 
of the observed data used for sensitivity mapping.  

The `SimulatorImaging` takes the generated strong lens image and convolves it with the PSF before adding noise. To 
prevent edge effects, the image is padded before convolution and then trimmed to restore its original `shape_native`.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=dataset.psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=False,
    noise_seed=1,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

dataset = simulator.via_source_image_from(
    tracer=tracer, grid=grid, source_image=source_image
)

plotter = aplt.ImagingPlotter(dataset=dataset)
plotter.subplot_dataset()

output = aplt.Output(path=".", filename="source_image", format="png")

plotter = aplt.ImagingPlotter(
    dataset=dataset, mat_plot_2d=aplt.MatPlot2D(output=output)
)
plotter.subplot_dataset()

"""
__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More magnification calculations.
- Source gradient calculations.
- A calculation which shows differential lensing effects (e.g. magnification across the source plane).
"""

"""
Results: Pixelization
=====================

This tutorial illustrates how to analyse the results of lens modeling where the source is modeled using an
`Inversion` and therefore has a pixelized source reconstruction we may be interested in inspecting.

This includes examples of how to output the source reconstruction to .fits files, so that a more detailed analysis
can be performed enabling source science studies.

This tutorial focuses on explaining how to use the inferred inversion to compute results as numpy arrays and only
briefly discusses visualization.

__Plot Module__

This example uses the **PyAutoLens** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
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
__Model Fit__

The code below (which we have omitted comments from for brevity) performs a lens model-fit using Nautilus. You should
be familiar enough with lens modeling to understand this, if not you should go over the beginner model-fit script again!
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

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

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="mass[sie]_source[pix]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=al.PositionsLHPenalty(positions=positions, threshold=0.5),
)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Inversion__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit`, which contains the
`Inversion` object we need.
"""
inversion = result.max_log_likelihood_fit.inversion

inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
__Linear Objects__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `Delaunay` mesh).

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
print(f"Delaunay Mapper = {inversion.linear_obj_list[0]}")

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
__Interpolated Source__

The pixelized source reconstruction used by an `Inversion` is often on an irregular grid (e.g. a Delaunay triangulation
or Voronoi mesh), making it difficult to manipulate and inspect after the lens modeling has completed (although we show 
how to do this below).

A simpler way to inspect the source reconstruction is to interpolate the reconstruction values from the irregular
pixelization (e.g. a Delaunay triangulation or Voronoi mesh) to a uniform 2D grid of pixels.

(if you do not know what the `slim` and `native` properties below refer too, check back to tutorial 2 of the results
for a description).

Inversions can have multiple source reconstructions (e.g. double Einstein ring strong lenses) thus the majority of
quantities are returned as a list. It is likely you are only using one `Inversion` to reconstruction one source galaxy,
so these lists will likely contain only one entry

We interpolate the Delaunay triangulation this source is reconstructed on to a 2D grid of 401 x 401 square pixels. 
"""
interpolated_reconstruction_list = inversion.interpolated_reconstruction_list_from(
    shape_native=(401, 401)
)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(interpolated_reconstruction_list[0].slim)

"""
We can alternatively input the arc-second `extent` of the source reconstruction we want, which will not use square 
pixels unless symmetric y and x arc-second extents are input.

The extent is input via the notation (xmin, xmax, ymin, ymax), therefore unlike most of the **PyAutoLens** API it
does not follow the (y,x) convention. This will be updated in a future version.
"""
interpolated_reconstruction_list = inversion.interpolated_reconstruction_list_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_reconstruction_list[0].slim)

"""
The interpolated errors on the source reconstruction can also be computed, in case you are planning to perform 
model-fitting of the source reconstruction.
"""
interpolated_errors_list = inversion.interpolated_errors_list_from(
    shape_native=(401, 401), extent=(-1.0, 1.0, -1.0, 1.0)
)

print(interpolated_errors_list[0].slim)

"""
__Reconstruction__

The source reconstruction is also available as a 1D numpy array of values representative of the source pixelization
itself (in this example, the reconstructed source values at the vertexes of each Delaunay triangle).
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

Note that any parametric light profiles in the lens model (e.g. the `bulge` and `disk` of a lens galaxy) are not 
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
We can create a source-plane magnification map by passed the image-plane magnification map computed via the
tracer.

[CURRENTLY DOES NOT WORK, BECAUSE THE MAPPING FUNCTION NEEDS TO INCORPORATE THE VARYING VORONOI PIXEL AREA].
"""
tracer = result.max_log_likelihood_tracer

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=dataset.grid_pixelization)
tracer_plotter.figures_2d(magnification=True)

magnification_2d = tracer.magnification_2d_from(grid=dataset.grid_pixelization)

magnification_to_source = mapper_list[0].mapped_to_source_from(array=magnification_2d)

mapper_plotter = aplt.MapperPlotter(mapper=mapper_list[0])
mapper_plotter.plot_source_from(pixel_values=magnification_2d)

"""
We can interpolate these arrays to output them to fits.
"""

"""
Although the model-fit used a Voronoi mesh, there is no reason we need to use this pixelization to map the image-plane
data onto a source-plane array.

We can instead map the image-data onto a rectangular pixelization, which has the nice property of giving us a
regular 2D array of data which could be output to .fits format.

[NOT CLEAR IF THIS WORKS YET, IT IS UNTESTED!].
"""
mesh = al.mesh.Rectangular(shape=(50, 50))

mapper_grids = mesh.mapper_grids_from(source_plane_data_grid=dataset.grid)

mapper = al.Mapper(mapper_grids=mapper_grids, regularization=None)

image_to_source = mapper.mapped_to_source_from(array=dataset.data)

mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.plot_source_from(pixel_values=image_to_source)


"""
__Magnification__

The inversion includes the magnification of the lens model, which is computed as the sum of flux
in every image-plane image pixel divided by the sum of flux values in every source-plane source pixel.

[INSERT CODE HERE]

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
__Future Ideas / Contributions__

Here are a list of things I would like to add to this tutorial but haven't found the time. If you are interested
in having a go at adding them contact me on SLACK! :)

- More 
- Source gradient calculations.
- A calculation which shows differential lensing effects (e.g. magnification across the source plane).
"""

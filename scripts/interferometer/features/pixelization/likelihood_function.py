"""
__Log Likelihood Function: Pixelization__

This script provides a step-by-step guide of the **PyAutoLens** `log_likelihood_function` which is used to fit
`Interferometer` data with an inversion (specifically a `RectangularMagnification` mesh and `Constant` regularization scheme`).

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
 - `imaging/linear_light_profile/log_likelihood_function.py` the likelihood function for a linear light profile, which
 introduces the linear algebra used for a pixelization but with a simpler use case.

This script repeats all text and code examples in the above likelihood function examples. It therefore can be used to
learn about the linear light profile likelihood function without reading other likelihood scripts.
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

import autolens as al
import autolens.plot as aplt

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=4.0
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `simple` from .fits files, which we will fit 
with the model.

This includes the method used to Fourier transform the real-space image of the galaxy to the uv-plane and compare 
directly to the visibilities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities. We will discuss how the calculation of the likelihood
function changes for different methods of Fourier transforming in this guide.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
This guide uses in-built visualization tools for plotting. 

For example, using the `InterferometerPlotter` the dataset we perform a likelihood evaluation on is plotted.

The `subplot_dataset` displays the visibilities in the uv-plane, which are the raw data of the interferometer
dataset. These are what will ultimately be directly fitted in the Fourier space.

The `subplot_dirty_images` displays the dirty images of the dataset, which are the reconstructed images of visibilities
using an inverse Fourier transform to convert these to real-space. These dirty images are not the images we fit, but
visualization of the dirty images are often used in radio interferometry to show the data in a way that is more
interpretable to the human eye.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, in previous likelihood function examples we disabled over sampling by setting `sub_size=1`. 

A full description of over sampling and how to use it is given in `autogalaxy_workspace/*/guides/over_sampling.py`.

Over sampling is used for the same purpose in a pixelization, whereby it uses multiple samples of a pixel to
perform the reconstruction via the pixelization. It uses an independent over sampling factor to the light profile
over sampling factor, called `over_sample_size_pixelization`.

However, for interferometer datasets, over sampling is not used in the pixelization (or for light profiles)
therefore it is implicitly set to 1 and can be ignored hereafter.

__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

For light profiles these are given by `dataset.lp`, which is a uniform grid of (y,x) Cartesian coordinates
which have had the 3.0" circular mask applied.

A pixelization uses a separate grid of (y,x) coordinates, called `dataset.grids.pixelization`, which is
identical to the light profile grid but may of had a different over-sampling scale applied (but in this example
does not).

Each (y,x) coordinate coordinates to the centre of each image-pixel in the dataset, meaning that when this grid is
used to construct a pixelization there is a straight forward mapping between the image data and pixelization pixels.
"""
grid_plotter = aplt.Grid2DPlotter(grid=dataset.grids.pixelization)
grid_plotter.figure_2d()

"""
__Lens Galaxy__

We set up a lens galaxy with the lens light and mass, which we will use to demonstrate a pixelized source
reconstruction.
"""
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

The galaxy includes the RectangularMagnification mesh and constant regularization scheme, which will ultimately be used
to reconstruct its star forming clumps.
"""
pixelization = al.Pixelization(
    image_mesh=None,
    mesh=al.mesh.RectangularMagnification(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Source Pixel Centre Calculation__

In order to reconstruct the source galaxy using a RectangularMagnification mesh, we need to determine the centres of the RectangularMagnification 
source pixels.

The image-mesh `Overlay` object computes the source-pixel centres in the image-plane (which are ray-traced to the 
source-plane below). The source pixelization therefore adapts to the lens model magnification, because more
source pixels will congregate in higher magnification regions.

This calculation is performed by overlaying a uniform regular grid with an `pixelization_shape_2d` over the image
mask and retaining all pixels that fall within the mask. This uses a `Grid2DSparse` object.

"""
image_plane_mesh_grid = pixelization.image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

"""
Plotting this grid shows a sparse grid of (y,x) coordinates within the mask, which will form our source pixel centres.
"""
visuals = aplt.Visuals2D(grid=image_plane_mesh_grid)
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(dirty_image=True)

"""
__Ray Tracing__

To perform lensing calculations we ray-trace every 2d (y,x) coordinate $\theta$ from the image-plane to its (y,x) 
source-plane coordinate $\beta$ using the summed deflection angles $\alpha$ of the mass profiles:

 $\beta = \theta - \alpha(\theta)$

The likelihood function of a pixelized source reconstruction ray-traces two grids from the image-plane to the source-plane:

 1) A 2D grid of (y,x) coordinates aligned with the imaging data's image-pixels.

 2) The sparse 2D grid of (y,x) coordinates above which form the centres of the RectangularMagnification pixels.

The function below computes the 2D deflection angles of the tracer's lens galaxies and subtracts them from the 
image-plane 2D (y,x) coordinates $\theta$ of each grid, thus ray-tracing their coordinates to the source plane to 
compute their $\beta$ values.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
The source code gets quite complex when handling grids for a pixelization, but it is all handled in
the `TracerToInversion` objects.

The plots at the bottom of this cell show the traced grids used by the source pixelization, showing
how the RectangularMagnification mesh and traced image pixels are constructed.
"""
tracer_to_inversion = al.TracerToInversion(tracer=tracer, dataset=dataset)

# A list of every grid (e.g. image-plane, source-plane) however we only need the source plane grid with index -1.
traced_grid_pixelization = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization
)[-1]

# This functions a bit weird - it returns a list of lists of ndarrays. Best not to worry about it for now!
traced_mesh_grid = tracer_to_inversion.traced_mesh_grid_pg_list[-1][-1]

mat_plot = aplt.MatPlot2D(axis=aplt.Axis(extent=[-1.5, 1.5, -1.5, 1.5]))

grid_plotter = aplt.Grid2DPlotter(grid=traced_grid_pixelization, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

grid_plotter = aplt.Grid2DPlotter(grid=traced_mesh_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Border Relocation__

Coordinates that are ray-traced near the mass profile centres are heavily demagnified and may trace to far outskirts of
the source-plane. 

We relocate these pixels (for both grids above) to the edge of the source-plane border (defined via the border of the 
image-plane mask). This is detailed in **HowToLens chapter 4 tutorial 5** and figure 2 of https://arxiv.org/abs/1708.07377.
"""
from autoarray.inversion.pixelization.border_relocator import BorderRelocator

border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)

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
__Rectangular Mesh__

The relocated pixelization grid is used to create the `Pixelization`'s RectangularMagnification mesh using the `scipy.spatial` library.
"""
grid_Rectangular = al.Mesh2DRectangular(
    values=relocated_mesh_grid,
)

"""
Plotting the RectangularMagnification mesh shows that the source-plane and been discretized into a grid of irregular RectangularMagnification pixels.

(To plot the RectangularMagnification mesh, we have to convert it to a `Mapper` object, which is described in the next likelihood step).

Below, we plot the RectangularMagnification mesh without the traced image-grid pixels (for clarity) and with them as black dots in order
to show how each set of image-pixels fall within a RectangularMagnification pixel.
"""
mapper_grids = al.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=relocated_grid,
    source_plane_mesh_grid=grid_Rectangular,
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

We now combine grids computed above to create a `Mapper`, which describes how every image-plane pixel maps to
every source-plane RectangularMagnification pixel. 

There are two steps in this calculation, which we show individually below.
"""
mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)


"""
__Image-Source Mapping__

We now combine grids computed above to create a `Mapper`, which describes how every image-plane pixel maps to
every source-plane RectangularMagnification pixel. 

There are two steps in this calculation, which we show individually below.
"""
mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

"""
The `Mapper` contains:

 1) `source_plane_data_grid`: the traced grid of (y,x) image-pixel coordinate centres (`relocated_grid`).
 2) `source_plane_mesh_grid`: The RectangularMagnification mesh of traced (y,x) source-pixel coordinates (`grid_Rectangular`).

We have therefore discretized the source-plane into a RectangularMagnification mesh, and can pair every traced image-pixel coordinate
with the corresponding RectangularMagnification source pixel it lands in.

This pairing is contained in the ndarray `pix_indexes_for_sub_slim_index` which maps every image-pixel index to 
every source-pixel index.

In the API, the `pix_indexes` refers to the source pixel indexes (e.g. source pixel 0, 1, 2 etc.) and `sub_slim_index` 
refers to the index of an image pixel (e.g. image-pixel 0, 1, 2 etc.). 

For example, printing the first ten entries of `pix_indexes_for_sub_slim_index` shows the first ten source-pixel
indexes these image sub-pixels map too.
"""
pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index

print(pix_indexes_for_sub_slim_index[0:9])

"""
This array can be used to visualize how an input list of image-pixel indexes map to the source-plane.

It also shows that image-pixel indexing begins from the top-left and goes rightwards and downwards, accounting for 
all image-pixels which are not masked.
"""
visuals = aplt.Visuals2D(indexes=[list(range(2050, 2090))])

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)
mapper_plotter.subplot_image_and_mapper(
    image=dataset.dirty_image, interpolate_to_uniform=False
)

"""
The reverse mappings of source-pixels to image-pixels can also be used.

If we choose the right source-pixel index, we can see that multiple imaging occur whereby image-pixels in different
regions of the image-plane are grouped into the same source-pixel.
"""
pix_indexes = [[200]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(indexes=indexes)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(
    image=dataset.dirty_image, interpolate_to_uniform=False
)

"""
__Mapping Matrix__

The `mapping_matrix` represents the image-pixel to source-pixel mappings above in a 2D matrix. 

It has dimensions `(total_image_pixels, total_source_pixels)`.

(A number of inputs are not used for the `RectangularMagnification` pixelization and are expanded upon in the `features.ipynb`
log likelihood guide notebook).
"""

mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for RectangularMagnification
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for RectangularMagnification
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

"""
A 2D plot of the `mapping_matrix` shows of all image-source pixel mappings.

No row of pixels has more than one non-zero entry. It is not possible for two image pixels to map to the same source 
pixel (meaning that there are no correlated pixels in the mapping matrix).
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
Each column of the `mapping_matrix` can therefore be used to show all image-pixels it maps too. 

For example, above, we plotted all image-pixels of source-pixel 200 (as well as 202 and 204). We can extract all
image-pixel indexes of source pixels 200 using the `mapping_matrix` and use them to plot the image of this
source-pixel (which corresponds to only values of zeros or ones).
"""
indexes_source_pix_200 = np.nonzero(mapping_matrix[:, 200])

print(indexes_source_pix_200[0])

array_2d = al.Array2D(values=mapping_matrix[:, 200], mask=dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

"""
__Transformed Mapping Matrix ($f$)__

Each pixelization pixel can therefore be thought of as an image (where all entries of this image are zeros and ones). 

However, for interferometer datasets we want to fit the visibilities in the uv-plane, not the image-plane. Therefore,
each image in the `mapping_matrix` must be transformed to the uv-plane via a Fourier transform, such that each
column in the `transformed_mapping_matrix` represents the visibilities in the uv-plane of each pixelization pixel.

This operation changes the dimensions of the mapping matrix, meaning the `transformed_mapping_matrix` has
dimensions `(total_image_pixels, total_visibilities)`. 

If the number of visibilities is large (e.g. 10^6) this matrix becomes extremely large and computationally expensive to 
store memory, meaning the `w_tilde` likelihood function, described in 
the `/log_likelihood_function/interferometer/`w_tilde.ipynb` notebook must be used instead.

The `transformed_mapping_matrix` is also complex, storing all entries of the visibilities after the NUFFT as real
and complex values.
"""
transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
    mapping_matrix=mapping_matrix
)

"""
A 2D plot of the `transformed_mapping_matrix` shows all visibility-source pixel mappings.

Note how, unlike for the `mapping_matrix`, every row of image-pixels fully consists of non-zero entries. This
means the matrix is fully dense, making it even more difficult to store in memory for large datasets.

Below, we plot the real and imaginary components of the `transformed_mapping_matrix` separately.
"""
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

"""
Each column of the `transformed_mapping_matrix` shows all visibilities it maps to after the NUFFT.
"""
indexes_pix_200 = np.nonzero(transformed_mapping_matrix[:, 200])

print(indexes_pix_200[0])

visibilities = al.Visibilities(visibilities=transformed_mapping_matrix[:, 200])

grid_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
grid_plotter.figure_2d()

"""
In Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587) the `transformed_mapping_matrix` is denoted $f_{ij}$
where $i$ maps over all $I$ source pixels and $j$ maps over all $J$ visibilities. 

For example: 

 - $f_{0, 2} = 0.3$ indicates that visibility number $2$ maps to pixelization pixel $0$ with a weight of $0.3$ after the NUFFT.

The indexing of the `mapping_matrix` is reversed compared to the notation of WD03 (e.g. visibilities
are the first entry of `mapping_matrix` whereas for $f$ they are the second index).
"""
print(
    f"Mapping between visibility 0 and RectangularMagnification pixel 2 = {mapping_matrix[0, 2]}"
)

"""
__Data Vector (D)__

To solve for the RectangularMagnification pixel fluxes we now pose the problem as a linear inversion.

This requires us to convert the `transformed_mapping_matrix` and our `data` and `noise map` into matrices of certain dimensions. 

The `data_vector`, $D$, is the first matrix and it has dimensions `(total_Rectangular_pixels,)`.

In WD03 (https://arxiv.org/abs/astro-ph/0302587) and N15 (https://arxiv.org/abs/1412.7436) the data vector 
is give by: 

 $\vec{D}_{i} = \sum_{\rm  j=1}^{J}f_{ij}(d_{j})/\sigma_{j}^2 \, \, .$

Where:

 - $d_{\rm j}$ are the image-pixel data flux values.
 - $\sigma{\rm _j}^2$ are the statistical uncertainties of each image-pixel value.

$i$ maps over all $I$ source pixels and $j$ maps over all $J$ image pixels. 
"""
data_vector = (
    al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=np.array(dataset.data),
        noise_map=np.array(dataset.noise_map),
    )
)

"""
$D$ describes which RectangularMagnification pixels trace to which visibilities, with associated weights, after the NUFFT. This 
ensures the reconstruction fully accounts for the NUFFT when fitting the data.

We can plot $D$ as a column vector:
"""
plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

"""
The dimensions of $D$ are the number of source pixels.
"""
print("Data Vector:")
print(data_vector)
print(data_vector.shape)

"""
__Curvature Matrix (F)__

The `curvature_matrix` $F$ is the second matrix and it has 
dimensions `(total_Rectangular_pixels, total_Rectangular_pixels)`.

In WD03 / N15 (https://arxiv.org/abs/astro-ph/0302587) the curvature matrix is a 2D matrix given by:

 ${F}_{ik} = \sum_{\rm  j=1}^{J}f_{ij}f_{kj}/\sigma_{j}^2 \, \, .$

NOTE: this notation implicitly assumes a summation over $K$, where $k$ runs over all pixelization pixel indexes $K$.

Note how summation over $J$ runs over $f$ twice, such that every entry of $F$ is the sum of the multiplication
between all values in every two columns of $f$.

For example, $F_{0,1}$ is the sum of all visibility values in $f$ of source pixel 0 multiplied by
all visibility values of source pixel 1.

Visibilities are both real and complex values, and the `curvature_matrix` is computed separately for the real and
imaginary components of the visibilities and then summed together.
"""
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

"""
For $F_{ik}$ to be non-zero, this requires that the images of RectangularMagnification pixels $i$ and $k$ share at least one
image-pixel, which for visibilities after the NUFFT is always true for all $i$ and $k$.

For example, we can see a non-zero entry for $F_{100,101}$ and plotting their images
show overlap.
"""
source_pixel_0 = 0
source_pixel_1 = 1

print(curvature_matrix[source_pixel_0, source_pixel_1])

visibilities = al.Visibilities(
    visibilities=transformed_mapping_matrix[:, source_pixel_0],
)

grid_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
grid_plotter.figure_2d()

visibilities = al.Visibilities(
    visibilities=transformed_mapping_matrix[:, source_pixel_1],
)

grid_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
grid_plotter.figure_2d()

"""
The following chi-squared is minimized when we perform the inversion and reconstruct the source_galaxy:

$\chi^2 = \sum_{\rm  j=1}^{J} \bigg[ \frac{(\sum_{\rm  i=1}^{I} s_{i} f_{ij}) - d_{j}}{\sigma_{j}} \bigg]$

Where $s$ is the reconstructed pixel fluxes in all $I$ RectangularMagnification pixels.

The solution for $s$ is therefore given by (equation 5 WD03):

 $s = F^{-1} D$

We can compute this using NumPy linear algebra:
"""

# Because we are no using regularizartion (see below) it is common for the curvature matrix to be singular and lead
# to a LinAlgException. The loop below mitigates this -- you can ignore it as it is not important for understanding
# the PyAutoLens likelihood function.

for i in range(curvature_matrix.shape[0]):
    curvature_matrix[i, i] += 1e-8

reconstruction = np.linalg.solve(curvature_matrix, data_vector)

"""
We can plot this reconstruction -- it looks like a mess.

The pixelization pixels have noisy and unsmooth values, and it is hard to make out if a source galaxy is even being 
reconstructed. 

In fact, the linear inversion is (over-)fitting noise in the image data, meaning this system of equations is 
ill-posed. We need to apply some form of smoothing on the reconstruction to avoid over fitting noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction, interpolate_to_uniform=False)

"""
__Regularization Matrix (H)__

Regularization adds a linear regularization term $G_{\rm L}$ to the $\chi^2$ we solve for giving us a new merit 
function $G$ (equation 11 WD03):

 $G = \chi^2 + \lambda \, G_{\rm L}$

where $\lambda$ is the `regularization_coefficient` which describes the magnitude of smoothness that is applied. A 
higher $\lambda$ will regularize the source more, leading to a smoother source galaxy reconstruction.

Different forms for $G_{\rm L}$ can be defined which regularize the reconstruction in different ways. The 
`Constant` regularization scheme used in this example applies gradient regularization (equation 14 WD03):

 $G_{\rm L} = \sum_{\rm  i}^{I} \sum_{\rm  n=1}^{N}  [s_{i} - s_{i, v}]$

This regularization scheme is easier to express in words -- the summation goes to each RectangularMagnification pixelization pixel,
determines all RectangularMagnification pixels with which it shares a direct vertex (e.g. its neighbors) and penalizes solutions 
where the difference in reconstructed flux of these two neighboring pixels is large.

The summation does this for all RectangularMagnification pixels, thus it favours solutions where neighboring RectangularMagnification 
pixels reconstruct similar values to one another (e.g. it favours a smooth source galaxy reconstruction).

We now define the `regularization matrix`, $H$, which allows us to include this smoothing when we solve for $s$. $H$
has dimensions `(total_Rectangular_pixels, total_Rectangular_pixels)`.

This relates to $G_{\rm L}$ as (equation 13 WD03):

 $H_{ik} = \frac{1}{2} \frac{\partial G_{\rm L}}{\partial s_{i} \partial s_{k}}$

$H$ has the `regularization_coefficient` $\lambda$ folded into it such $\lambda$'s control on the degree of smoothing
is accounted for.
"""
regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=source_galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

"""
We can plot the regularization matrix and note that:

 - non-zero entries indicate that two RectangularMagnification pixelization pixels are neighbors and therefore are regularized 
 with one another.

 - Zeros indicate the two RectangularMagnification pixels do not neighbor one another.

The majority of entries are zero, because the majority of RectangularMagnification pixels are not neighbors with one another.
"""
plt.imshow(regularization_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
__F + Lamdba H__

$H$ enters the linear algebra system we solve for as follows (WD03 equation (12)):

 $s = [F + H]^{-1} D$
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

"""
__Galaxy Reconstruction (s)__

We can now solve the linear system above using NumPy linear algebra. 

Note that the for loop used above to prevent a LinAlgException is no longer required.
"""
reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
By plotting this source galaxy reconstruction we can see that regularization has lead us to reconstruct a smoother 
source galaxy, which actually looks like the star forming clumps in the imaging data! 

This also implies we are not over-fitting the noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction, interpolate_to_uniform=False)

"""
__Visibilities Reconstruction__

Using the reconstructed pixel fluxes we can map the reconstruction back to the image plane (via 
the `blurred mapping_matrix`) and produce a reconstruction of the image data.
"""
mapped_reconstructed_visibilities = (
    al.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        reconstruction=reconstruction,
    )
)

mapped_reconstructed_visibilities = al.Visibilities(
    visibilities=mapped_reconstructed_visibilities
)

grid_plotter = aplt.Grid2DPlotter(grid=mapped_reconstructed_visibilities.in_grid)
grid_plotter.figure_2d()


"""
__Likelihood Function__

We now quantify the goodness-of-fit of our pixelization source galaxy reconstruction. 

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for source galaxy modeling consists of five terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + s^{T} H s + \mathrm{ln} \, \left[ \mathrm{det} (F + H) \right] - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right] + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

This expression was first derived by Suyu 2006 (https://arxiv.org/abs/astro-ph/0601493) and is given by equation (19).
It was derived into **PyAutoLens** notation in Dye 2008 (https://arxiv.org/abs/0804.4002) equation (5).

We now explain what each of these terms mean.

__Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `mapped_reconstructed_visibilities`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not. 

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image 
for, corresponding to a fit with a lower likelihood.
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

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = al.Visibilities(visibilities=chi_squared_map)

grid_plotter = aplt.Grid2DPlotter(grid=chi_squared_map.in_grid)
grid_plotter.figure_2d()

"""
__Regularization Term__

The second term, $s^{T} H s$, corresponds to the $\lambda $G_{\rm L}$ regularization term we added to our merit 
function above.

This is the term which sums up the difference in flux of all reconstructed RectangularMagnification pixels, and reduces the 
likelihood of solutions where there are large differences in flux (e.g. the source galaxy is less smooth and more 
likely to be overfitting noise).

We compute it below via matrix multiplication, noting that the `regularization_coefficient`, $\lambda$, is built into 
the `regularization_matrix` already.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

"""
__Complexity Terms__

Up to this point, it is unclear why we chose a value of `regularization_coefficient=1.0`. 

We cannot rely on the `chi_squared` and `regularization_term` above to optimally choose its value, because increasing 
the `regularization_coefficient` smooths the solution more and therefore:

 - Decreases `chi_squared` by fitting the data worse, producing a lower `log_likelihood`.

 - Increases the `regularization_term` by penalizing the differences between source pixel fluxes more, again reducing
 the inferred `log_likelihood`.

If we set the regularization coefficient based purely on these two terms, we would set a value of 0.0 and be back where
we started over-fitting noise!

The terms $\left[ \mathrm{det} (F + H) \right]$ and $ - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right]$ address 
this problem. 

They quantify how complex the reconstruction is, and penalize solutions where *it is more complex*. Reducing 
the `regularization_coefficient` makes the source galaxy reconstruction more complex (because a galaxy that is 
smoothed less uses more flexibility to fit the data better).

These two terms therefore counteract the `chi_squared` and `regularization_term`, so as to attribute a higher
`log_likelihood` to solutions which fit the data with a more smoothed and less complex source (e.g. one with a higher 
`regularization_coefficient`).

In **HowToGalaxy** -> `chapter 4` -> `tutorial_4_bayesian_regularization` we expand on this further and give a more
detailed description of how these different terms impact the `log_likelihood_function`. 
"""
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

"""
__Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term ins the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the lens modeling process and has no impact on the 
model we infer.
"""
noise_normalization_real = np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
noise_normalization_imag = np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
noise_normalization = noise_normalization_real + noise_normalization_imag

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the model, by combining the five terms computed above using
the likelihood function defined above.
"""
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
    settings_inversion=al.SettingsInversion(
        use_w_tilde=False, use_border_relocator=True
    ),
)
fit_log_evidence = fit.log_evidence
print(fit_log_evidence)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus` (https://github.com/joshspeagle/nautilus)
but **PyAutoLens** supports multiple MCMC and optimization algorithms. 

__Log Likelihood Function: Fast Chi Squared__

This script describes how the `chi_squared` of an interferometer pixelization can be computed without using a
`transformed_mapping_matrix` or an NUFFT algorithm at all.

This means the likelihood function can be computed without ever performing an NUFFT, which for datasets of 10^6
visibilities or more can be extremely computationally expensive.

This can make the likelihood function significantly faster, for example with speed ups of hundreds of times or more
for tens or millions of visibilities. In fact, the run time does not scale with the number of visibilities at all,
meaning datasets of any size can be fitted in seconds.

It directly follows on from the `pixelization/log_likelihood_function.py` and ``pixelization/w_tilde.py` notebooks and
you should read through those examples before reading this script.

__Prerequisites__

You must read through the following likelihood functions first:

 - `pixelization/log_likelihood_function.py` the likelihood function for a pixelization.
 - `pixelization/w_tilde.py` the w-tilde formalism used to compute the likelihood function without an NUFFT.
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

import autolens as al
import autolens.plot as aplt


"""
__Dataset__

Following the `pixelization/log_likelihood_function.py` script, we load and mask an `Imaging` dataset and
set oversampling to 1.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=4.0
)

dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)


"""
__W Tilde__

The fast chi-squared method uses the w-tilde matrix, which we compute now.
"""
from autoarray import numba_util


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


w_tilde = w_tilde_curvature_interferometer_from(
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    grid_radians_slim=np.array(dataset.grid.in_radians),
)


"""
__Mapping Matrix__

It also uses the `mapping_matrix` which we compute now.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

galaxy = al.Galaxy(redshift=0.5, pixelization=pixelization)

grid_rectangular = al.Mesh2DRectangular.overlay_grid(
    shape_native=galaxy.pixelization.mesh.shape, grid=dataset.grids.pixelization
)

mapper_grids = al.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=dataset.grids.pixelization,
    source_plane_mesh_grid=grid_rectangular,
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for rectangular
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for rectangular
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

"""
__Log Likelihood Function: W Tilde__

This script describes how a pixelization can be computed using a different linear algebra calculation, but
one which produces an identical likelihood at the end.

This is called the `w_tilde` formalism, and for interferometer datasets it avoids storing the `operated_mapping_matrix`
in memory, meaning that in the regime of 1e6 or more visibilities this extremely large matrix does not need to be
stored in memory.

This can make the likelihood function significantly faster, for example with speed ups of hundreds of times or more
for tens or millions of visibilities. In fact, the run time does not scale with the number of visibilities at all,
meaning datasets of any size can be fitted in seconds.

It directly follows on from the `pixelization/log_likelihood_function.py` notebook and you should read through that
script before reading this script.

__Prerequisites__

You must read through the following likelihood functions first:

 - `pixelization/log_likelihood_function.py` the likelihood function for a pixelization.
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

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Following the `pixelization/log_likelihood_function.py` script, we load and mask an `Imaging` dataset and
set oversampling to 1.
"""
real_space_mask = al.Mask2D.circular(shape_native=(8, 8), pixel_scales=0.05, radius=4.0)

dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
__W Tilde__

We now compute the `w_tilde` matrix.

The `w_tilde` matrix is applied to the `curvature_matrix`, and allows us to efficiently compute the curvature matrix
without computing the `transformed_mapping_matrix` matrix. 

The functions used to do this has been copy and pasted from the `inversion` module of PyAutoArray source code below,
so you can see the calculation in full detail.

REMINDER: for the `real_space_mask` above with shape (800, 800) the `w_tilde` matrix will TAKE A LONG
TIME TO COMPUTE.
"""
from autoarray import numba_util


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


"""
We now compute the `w_tilde` matrices.
"""
w_tilde = w_tilde_curvature_interferometer_from(
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    grid_radians_slim=np.array(dataset.grid.in_radians),
)

"""
__Mapping Matrix__

The `w_tilde` matrix is applied directly to the `mapping_matrix` to compute the `curvature_matrix`.

Below, we perform the likelihood function steps described in the `pixelization/log_likelihood_function.py` script,
to create the `mapping_matrix` we will apply the `w_tilde` matrix to.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

galaxy = al.Galaxy(redshift=0.5, pixelization=pixelization)

grid_rectangular = al.Mesh2DRectangular.overlay_grid(
    shape_native=galaxy.pixelization.mesh.shape, grid=dataset.grids.pixelization
)

mapper_grids = al.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=dataset.grids.pixelization,
    source_plane_mesh_grid=grid_rectangular,
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for rectangular
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for rectangular
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

"""
__Curvature Matrix__

We can now compute the `curvature_matrix` using the `w_tilde` matrix and `mapping_matrix`, which amazingly uses
simple matrix multiplication.
"""


def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray, mapping_matrix: np.ndarray
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) from `w_tilde`.

    The dimensions of `w_tilde` are [image_pixels, image_pixels], meaning that for datasets with many image pixels
    this matrix can take up 10's of GB of memory. The calculation of the `curvature_matrix` via this function will
    therefore be very slow, and the method `curvature_matrix_via_w_tilde_curvature_preload_imaging_from` should be used
    instead.

    Parameters
    ----------
    w_tilde
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    return np.dot(mapping_matrix.T, np.dot(w_tilde, mapping_matrix))


curvature_matrix = curvature_matrix_via_w_tilde_from(
    w_tilde=w_tilde, mapping_matrix=mapping_matrix
)

"""
If you compare the `curvature_matrix` computed using the `w_tilde` matrix to the `curvature_matrix` computed using the
`operated_mapping_matrix` matrix in the other example scripts, you'll see they are identical.

__Data Vector__

The `data_vector` was computed in the `pixelization/log_likelihood_function.py` script using 
the `transformed_mapping_matrix`.

Fortunately, there is also an easy way to compute the `data_vector` which bypasses the need to compute the
`transformed_mapping_matrix` matrix, again using simple matrix multiplication.
"""
data_vector = np.dot(mapping_matrix.T, dataset.w_tilde.dirty_image)

"""
__Reconstruction__

The `reconstruction` is computed using the `curvature_matrix` and `data_vector` as per usual.
"""
regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
__Fast Chi Squared__

In the `pixelization/log_likelihood_function.py` example the mapped reconstructed visibilities were another quantity 
computed which used the `transformed_mapping_matrix` matrix, which is another step that must skip computing this matrix.

The w-tilde matrix again provides a trick which skips the need to compute the `transformed_mapping_matrix` matrix,
with the code for this shown below.
"""
print(mapping_matrix.shape)
print(w_tilde.shape)

chi_squared_term_1 = np.linalg.multi_dot(
    [
        reconstruction.T,  # NOTE: shape = (M, )
        curvature_matrix,  # NOTE: shape = (M, M)
        reconstruction,  # NOTE: shape = (M, )
    ]
)
chi_squared_term_2 = -2.0 * np.linalg.multi_dot(
    [reconstruction.T, data_vector]  # NOTE: shape = (M, )  # NOTE: i.e. dirty_image
)
chi_squared_term_3 = np.add(  # NOTE: i.e. noise_normalization
    np.sum(dataset.data.real**2.0 / dataset.noise_map.real**2.0),
    np.sum(dataset.data.imag**2.0 / dataset.noise_map.imag**2.0),
)

chi_squared = chi_squared_term_1 + chi_squared_term_2 + chi_squared_term_3

print(chi_squared)

"""
__Log Likelihood__

Finally, we verify that the log likelihood computed using the `curvature_matrix` and `data_vector` computed using the
`w_tilde` matrix is identical to the log likelihood computed using the `operated_mapping_matrix` matrix in the
other example scripts.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]


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
__Repeated Pattern in W_Tilde__

The `w_tilde` matrix has a repeated pattern, which can be used to perform the above calculations using far less
memory, at the expense of code complexity. 

First, let us consider the pattern of the `w_tilde` matrix, which is seen in the following 7 values: 
"""
print(w_tilde[0, 1])
print(w_tilde[1, 2])
print(w_tilde[2, 3])
print(w_tilde[3, 4])
print(w_tilde[4, 5])
print(w_tilde[5, 6])
print(w_tilde[6, 7])

"""
However, the pattern breaks for the next value, which is:
"""
print(w_tilde[7, 8])

"""
What do the first 7 values have in common?

Let us think about the `real_space_mask` of the interferometer dataset, which I have made a really basic cartoon of
below:

![w_tilde](https://github.com/Jammy2211/autogalaxy_workspace/blob/main/scripts/advanced/log_likelihood_function/interferometer/pixelization/w_tilde_cartoon.png?raw=true)

What elements 0 -> 6 of the `w_tilde` matrix have in common is that they are next to one another in the real-space,
to the right, in the mask.

The element 6 -> 7 breaks this pattern, as it is at the end of the mask and there is no pixel to the right of it,
so it "jumps" to the next row.

We can now reinspect how the `w_tilde` matrix is computed, and see that the pattern of the `w_tilde` matrix is
determined by the real-space mask:
"""


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            """

            !!!LOOK HERE!!!!

            """

            y_offset = (
                grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            )  # The y-offset is 0 for pixels 0 -> 6, but becomes non-zero for 6 -> 7
            x_offset = (
                grid_radians_slim[i, 0] - grid_radians_slim[j, 0]
            )  # The x-offset is the same for pixels 0 -> 6 and 6 -> 7

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


"""
The `y_offset` and `x_offset` values are what determine the repeated pattern of the `w_tilde` matrix, and therefore
mean it has far fewer unique values than the number of pixels in the real-space mask.

This could make our calculation way more efficient: as if we can exploit it we would not store [image_pixels, image_pixels]
values (where image_pixels is the number of pixels in the real-space mask and can easily reach 100,000, or 100GB+ memory),
but instead far fewer values.

This could also, maybe, speed up the matrix multiplication calculation, as we would be performing far fewer operations.

__W Tilde 1D__

The function below shows how we compute `w_tilde_curvature_preload`, which is a 2D array of dimensions
[2*shape_masked_pixels_y, 2*shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x) size corresponding to the
extent of unmasked pixels that go vertically and horizontally across the mask.

print(real_space_mask.shape_native_masked_pixels)

The idea behind this is we don't need to store all [image_pixels, image_pixels] values of the `w_tilde` matrix, but
instead only the unique values of the `w_tilde` matrix that are computed for each unique (y,x) offset between pairs of
pixels in the real-space mask.

Another complication is that the `y_offset` and `x_offset` values can be negative, for example if we pair a pixel
to its neighbor to the left.

That is why it has shape [2*shape_masked_pixels_y, 2*shape_masked_pixels_x, 2], with a factor of 2* in front of the
shape of the real-space mask. This is so that negative offsets can be stored in the negative half of the 2D array.

The function also has four inner four loops, which store the values of the `w_tilde` matrix for each unique (y,x) offset
between pairs of pixels in the real-space mask.
"""
from typing import Tuple


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [unmasked_image_pixels, unmasked_image_pixels] that encodes the
    NUFFT of every pair of image pixels given the noise map. This can be used to efficiently compute the curvature
    matrix via the mapping matrix, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.
    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. This methods creates
    a preload matrix that can compute the matrix w_tilde via an efficient preloading scheme which exploits the
    symmetries in the NUFFT.
    To compute w_tilde, one first defines a real space mask where every False entry is an unmasked pixel which is
    used in the calculation, for example:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an imaging.Mask2D, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     x = `True` (Pixel is masked and excluded from lens)
        IxIxIxIoIoIoIxIxIxIxI     o = `False` (Pixel is not masked and included in lens)
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and
    downwards, therefore:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxI0I1I2IxIxIxIxI
        IxIxIxI3I4I5IxIxIxIxI
        IxIxIxI6I7I8IxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
    In the standard calculation of `w_tilde` it is a matrix of
    dimensions [unmasked_image_pixels, unmasked_pixel_images], therefore for the example mask above it would be
    dimensions [9, 9]. One performs a double for loop over `unmasked_image_pixels`, using the (y,x) spatial offset
    between every possible pair of unmasked image pixels to precompute values that depend on the properties of the NUFFT.
    This calculation has a lot of redundancy, because it uses the (y,x) *spatial offset* between the image pixels. For
    example, if two image pixel are next to one another by the same spacing the same value will be computed via the
    NUFFT. For the example mask above:

    - The value precomputed for pixel pair [0,1] is the same as pixel pairs [1,2], [3,4], [4,5], [6,7] and [7,9].

    - The value precomputed for pixel pair [0,3] is the same as pixel pairs [1,4], [2,5], [3,6], [4,7] and [5,8].

    - The values of pixels paired with themselves are also computed repeatedly for the standard calculation (e.g. 9
      times using the mask above).

    The `w_tilde_preload` method instead only computes each value once. To do this, it stores the preload values in a
    matrix of dimensions [shape_masked_pixels_y, shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x)
    size of the vertical and horizontal extent of unmasked pixels, e.g. the spatial extent over which the real space
    grid extends.
    Each entry in the matrix `w_tilde_preload[:,:,0]` provides the precomputed NUFFT value mapping an image pixel
    to a pixel offset by that much in the y and x directions, for example:

    - w_tilde_preload[0,0,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
      in the x direction by 0 - the values of pixels paired with themselves.

    - w_tilde_preload[1,0,0] gives the precomputed values of image pixels that are offset in the y direction by 1 and
      in the x direction by 0 - the values of pixel pairs [0,3], [1,4], [2,5], [3,6], [4,7] and [5,8]

    - w_tilde_preload[0,1,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
      in the x direction by 1 - the values of pixel pairs [0,1], [1,2], [3,4], [4,5], [6,7] and [7,9].

    Flipped pairs:

    The above preloaded values pair all image pixel NUFFT values when a pixel is to the right and / or down of the
    first image pixel. However, one must also precompute pairs where the paired pixel is to the left of the host
    pixels. These pairings are stored in `w_tilde_preload[:,:,1]`, and the ordering of these pairings is flipped in the
    x direction to make it straight forward to use this matrix when computing w_tilde.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    shape_masked_pixels_2d
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    Returns
    -------
    ndarray
        A matrix that precomputes the values for fast computation of w_tilde.
    """

    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
            x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[i, j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(y_shape):
        for j in range(x_shape):
            if j > 0:
                y_offset = (
                    grid_radians_2d[0, -1, 0]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[0, -1, 1]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0:
                y_offset = (
                    grid_radians_2d[-1, 0, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, 0, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0 and j > 0:
                y_offset = (
                    grid_radians_2d[-1, -1, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, -1, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload


curvature_preload = w_tilde_curvature_preload_interferometer_from(
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    shape_masked_pixels_2d=np.array(
        dataset.transformer.grid.mask.shape_native_masked_pixels
    ),
    grid_radians_2d=np.array(
        dataset.transformer.grid.mask.derive_grid.all_false.in_radians.native
    ),
)

"""
We can now use the `curvature_preload` matrix to compute the `w_tilde` matrix with its original dimensions
of [image_pixels, image_pixels] using the function below.

This is a lot faster than the original calculation, as we are only storing the unique values of the `w_tilde` matrix
and avoid repeating the same calculation for every pair of pixels in the real-space mask.
"""


@numba_util.jit()
def w_tilde_via_preload_from(w_tilde_preload, native_index_for_slim_index):
    """
    Use the preloaded w_tilde matrix (see `w_tilde_preload_interferometer_from`) to compute
    w_tilde (see `w_tilde_interferometer_from`) efficiently.

    Parameters
    ----------
    w_tilde_preload
        The preloaded values of the NUFFT that enable efficient computation of w_tilde.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    slim_size = len(native_index_for_slim_index)

    w_tilde_via_preload = np.zeros((slim_size, slim_size))

    for i in range(slim_size):
        i_y, i_x = native_index_for_slim_index[i]

        for j in range(i, slim_size):
            j_y, j_x = native_index_for_slim_index[j]

            y_diff = j_y - i_y
            x_diff = j_x - i_x

            w_tilde_via_preload[i, j] = w_tilde_preload[y_diff, x_diff]

    for i in range(slim_size):
        for j in range(i, slim_size):
            w_tilde_via_preload[j, i] = w_tilde_via_preload[i, j]

    return w_tilde_via_preload


w_matrix = w_tilde_via_preload_from(
    w_tilde_preload=curvature_preload,
    native_index_for_slim_index=real_space_mask.derive_indexes.native_for_slim,
)

"""
The following function is how we compute `curvature_matrix` using the `w_tilde` matrix computed using the preload
method.
"""


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
    [image_pixels, 2]). The massive reduction in the size of this matrix in memory allows for much fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
    pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
    compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    image_pixels = len(native_index_for_slim_index)

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]

            ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]

            for ip1 in range(image_pixels):
                ip1_y, ip1_x = native_index_for_slim_index[ip1]

                for ip1_pix in range(pix_size_for_sub_slim_index[ip1]):
                    sp1 = pix_indexes_for_sub_slim_index[ip1, ip1_pix]

                    # This is where the magic happens.

                    # Basically, `curvature_preload` stores the unique values of the w_tilde matrix in a structure
                    # where each combination of index differences are the dimensions of the arrays.

                    # So, if y_diff=0 and x_diff=1, it goes to the 0,1 index of the `curvature_preload` array,
                    # which by definition is the unique value of the w_tilde matrix for pixels that are offset by
                    # 0 in the y direction and 1 in the x direction in pixel units.

                    ip1_weight = pix_weights_for_sub_slim_index[ip1, ip1_pix]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    curvature_matrix[sp0, sp1] += (
                        curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight
                    )

    return curvature_matrix


curvature_matrix_fast = curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload=dataset.w_tilde.curvature_preload,
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
    native_index_for_slim_index=dataset.transformer.real_space_mask.derive_indexes.native_for_slim,
    pix_pixels=mapper.pixels,
)

print(curvature_matrix_fast - curvature_matrix)

"""
__Wrap Up__

We have presented a visual step-by-step guide to the pixelization likelihood function.

There are a number of other inputs features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in this package. In brief, these describe:

 - **Over Sampling**: Oversampling the image grid into a finer grid of sub-pixels, which are all individually 
 paired fractionally with each RectangularMagnification pixel.

 - **Source-plane Interpolation**: Using bilinear interpolation on the RectangularMagnification pixelization to pair each 
 image (sub-)pixel to multiple RectangularMagnification pixels with interpolation weights.

 - **Luminosity Weighted Regularization**: Using an adaptive regularization coefficient which adapts the level of 
 regularization applied to the source galaxy based on its luminosity.
"""

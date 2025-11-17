"""
__Log Likelihood Function: Pixelization__

This script provides a step-by-step guide of the **PyAutoLens** `log_likelihood_function` which is used to fit
`Imaging` data with a pixelization (`RectangularMagnification` mesh and `Constant` regularization scheme`).

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoLens**, so that readers can understand the
 likelihood function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

 - To make inversions in **PyAutoLens** less of a "black-box" to users.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions, modules and
packages are called when the likelihood is evaluated.

__Prerequisites__

The likelihood function of a pixelization builds on that used for standard parametric light profiles and
linear light profiles, therefore you must read the following notebooks before this script:

- `imaging/likelihood_function.ipynb`.
- `imaging/linear_light_profile/likelihood_function.ipynb`.
"""

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

In order to perform a likelihood evaluation, we first load a dataset.

This example fits a simulated strong lens which is simulated using a 0.1 arcsecond-per-pixel resolution (this is lower
resolution than the best quality Hubble Space Telescope imaging and close to that of the Euclid space satellite).
"""
dataset_path = Path("dataset", "imaging", "simple")

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
Throughout this guide, I will use **PyAutoLens**'s in-built visualization tools for plotting. 

For example, using the `ImagingPlotter` I can plot the imaging dataset we performed a likelihood evaluation on.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

The likelihood is only evaluated using image pixels contained within a 2D mask, which we choose before performing
lens modeling.

Below, we define a 2D circular mask with a 3.0" radius.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
When we plot the masked imaging, only the circular masked region is shown.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=masked_dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, we disable over sampling in this guide by setting `sub_size=1`. 

a full description of over sampling and how to use it is given in `autolens_workspace/*/guides/over_sampling.py`.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1,
    over_sample_size_pixelization=1,
)

"""
__Masked Image Grid__

To perform lensing calculations we first must define the 2D image-plane (y,x) coordinates used in the calculation.

These are given by `masked_dataset.grid`, which we can plot and see is a uniform grid of (y,x) Cartesian coordinates
which have had the 3.0" circular mask applied.
"""
grid_plotter = aplt.Grid2DPlotter(grid=masked_dataset.grids.pixelization)
grid_plotter.figure_2d()

"""
__Lens Galaxy__

We set up a lens galaxy with the lens light and mass, which we will use to demonstrate a pixelized source
reconstruction.
"""
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

The source galaxy is reconstructed using a pixel-grid, in this example a RectangularMagnification mesh, which accounts for 
irregularities and asymmetries in the source's surface brightness. 

A constant regularization scheme is applied which applies a smoothness prior on the reconstruction. 
"""
pixelization = al.Pixelization(
    image_mesh=None,
    mesh=al.mesh.RectangularMagnification(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Lens Light__

Compute a 2D image of the lens galaxy's light as the sum of its individual light profiles (the `Sersic` 
bulge). 

This computes the `image` of each `LightProfile` and adds them together. 
"""
image = lens_galaxy.image_2d_from(grid=masked_dataset.grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=masked_dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
To convolve the lens's 2D image with the imaging data's PSF, we need its `blurring_image`. This represents all flux 
values not within the mask, which are close enough to it that their flux blurs into the mask after PSF convolution.

To compute this, a `blurring_mask` and `blurring_grid` are used, corresponding to these pixels near the edge of the 
actual mask whose light blurs into the image:
"""
blurring_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grids.blurring)

galaxy_plotter = aplt.GalaxyPlotter(
    galaxy=lens_galaxy, grid=masked_dataset.grids.blurring
)
galaxy_plotter.figures_2d(image=True)

"""
__Lens Light Convolution + Subtraction__

Convolve the 2D lens light images above with the PSF in real-space (as opposed to via an FFT) using a `Kernel2D`.
"""
convolved_image_2d = masked_dataset.psf.convolved_image_from(
    image=image, blurring_image=blurring_image_2d
)

array_2d_plotter = aplt.Array2DPlotter(array=convolved_image_2d)
array_2d_plotter.figure_2d()

"""
We can now subtract this image from the observed image to produce a `lens_subtracted_image_2d`:
"""
lens_subtracted_image_2d = masked_dataset.data - convolved_image_2d

array_2d_plotter = aplt.Array2DPlotter(array=lens_subtracted_image_2d)
array_2d_plotter.figure_2d()

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
    mask=mask,
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
    image=lens_subtracted_image_2d, interpolate_to_uniform=False
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
    image=lens_subtracted_image_2d, interpolate_to_uniform=False
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

array_2d = al.Array2D(values=mapping_matrix[:, 200], mask=masked_dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

"""
__Blurred Mapping Matrix ($f$)__

Each source-pixel can therefore be thought of as an image (where all entries of this image are zeros and ones). 

To incorporate the imaging data's PSF, we simply blur each one of these source-pixel images with the imaging data's 
Point Spread Function (PSF) via 2D convolution.

This operation does not change the dimensions of the mapping matrix, meaning the `blurred_mapping_matrix` also has
dimensions `(total_image_pixels, total_source_pixels)`. It turns the values of zeros and ones into 
non-integer values which have been blurred by the PSF.
"""
blurred_mapping_matrix = masked_dataset.psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix
)

"""
A 2D plot of the `blurred_mapping_matrix` shows all image-source pixel mappings including PSF blurring.

Note how, unlike for the `mapping_matrix`, every row of image-pixels now has multiple non-zero entries. It is now 
possible for two image pixels to map to the same source pixel, because they become correlated by PSF convolution.
"""
plt.imshow(
    blurred_mapping_matrix,
    aspect=(blurred_mapping_matrix.shape[1] / blurred_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

"""
Each column of the `blurred_mapping_matrix` shows all image-pixels it maps to after PSF blurring. 
"""
indexes_source_pix_200 = np.nonzero(blurred_mapping_matrix[:, 200])

print(indexes_source_pix_200[0])

array_2d = al.Array2D(values=blurred_mapping_matrix[:, 200], mask=masked_dataset.mask)

array_2d_plotter = aplt.Array2DPlotter(array=array_2d)
array_2d_plotter.figure_2d()

"""
In Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587) the `blurred_mapping_matrix` is denoted $f_{ij}$
where $i$ maps over all $I$ source pixels and $j$ maps over all $J$ image pixels. 

For example: 

 - $f_{0, 2} = 0.3$ indicates that image-pixel $2$ maps to source-pixel $0$ with a weight of $0.3$ after PSF convolution.
 - $f_{4, 8} = 0$ indicates that image-pixel $8$ does not map to source-pixel $4$, even after PSF convolution.

The indexing of the `mapping_matrix` is reversed compared to the notation of WD03 (e.g. image pixels
are the first entry of `mapping_matrix` whereas for $f$ they are the second index).
"""
print(f"Mapping between image pixel 0 and source pixel 2 = {mapping_matrix[0, 2]}")

"""
__Data Vector (D)__

To solve for the source pixel fluxes we now pose the problem as a linear inversion.

This requires us to convert the `blurred_mapping_matrix` and our `data` and `noise map` into matrices of certain dimensions. 

The `data_vector`, $D$, is the first matrix and it has dimensions `(total_source_pixels,)`.

In WD03 (https://arxiv.org/abs/astro-ph/0302587) and N15 (https://arxiv.org/abs/1412.7436) the data vector 
is give by: 

 $\vec{D}_{i} = \sum_{\rm  j=1}^{J}f_{ij}(d_{j} - b_{j})/\sigma_{j}^2 \, \, .$

Where:

 - $d_{\rm j}$ are the image-pixel data flux values.
 - $b_{\rm j}$ are the brightness values of the lens light model (therefore $d_{\rm  j} - b_{\rm j}$ is the lens light
 subtracted image).
 - $\sigma{\rm _j}^2$ are the statistical uncertainties of each image-pixel value.

$i$ maps over all $I$ source pixels and $j$ maps over all $J$ image pixels. 

NOTE: WD03 assume the data is already lens subtracted thus $b_{j}$ is omitted (e.g. all values are zero).
"""
data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(lens_subtracted_image_2d),
    noise_map=np.array(masked_dataset.noise_map),
)

"""
$D$ describes which deconvolved source pixels trace to which image-plane pixels. This ensures the source reconstruction
fully accounts for the PSF when fitting the data.

We can plot $D$ as a column vector:
"""
plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

"""
__Curvature Matrix (F)__

The `curvature_matrix` $F$ is the second matrix and it has dimensions `(total_source_pixels, total_source_pixels)`.

In WD03 / N15 (https://arxiv.org/abs/astro-ph/0302587) the curvature matrix is a 2D matrix given by:

 ${F}_{ik} = \sum_{\rm  j=1}^{J}f_{ij}f_{kj}/\sigma_{j}^2 \, \, .$

NOTE: this notation implicitly assumes a summation over $K$, where $k$ runs over all source-pixel indexes $K$.

Note how summation over $J$ runs over $f$ twice, such that every entry of $F$ is the sum of the multiplication
between all values in every two columns of $f$.

For example, $F_{0,1}$ is the sum of every blurred image pixels values in $f$ of source pixel 0 multiplied by
every blurred image pixel value of source pixel 1.
"""
curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
For $F_{ik}$ to be non-zero, this requires that the images of source pixels $i$ and $k$ share at least one
image-pixel, which we saw above is only possible due to PSF blurring.

For example, we can see a non-zero entry for $F_{100,101}$ and plotting their images
show overlap.
"""
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

"""
The following chi-squared is minimized when we perform the inversion and reconstruct the source:

$\chi^2 = \sum_{\rm  j=1}^{J} \bigg[ \frac{(\sum_{\rm  i=1}^{I} s_{i} f_{ij}) + b_{j} - d_{j}}{\sigma_{j}} \bigg]$

Where $s$ is the reconstructed source pixel fluxes in all $I$ source pixels.

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
We can plot this source reconstruction -- it looks like a mess.

The source pixels have noisy and unsmooth values, and it is hard to make out if a source is even being reconstructed. 

In fact, the linear inversion is (over-)fitting noise in the image data, meaning this system of equations is 
ill-posed. We need to apply some form of smoothing on the source reconstruction to avoid over fitting noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction, interpolate_to_uniform=False)

"""
__Regularization Matrix (H)__

Regularization adds a linear regularization term $G_{\rm L}$ to the $\chi^2$ we solve for giving us a new merit 
function $G$ (equation 11 WD03):

 $G = \chi^2 + \lambda \, G_{\rm L}$
 
where $\lambda$ is the `regularization_coefficient` which describes the magnitude of smoothness that is applied. A 
higher $\lambda$ will regularize the source more, leading to a smoother source reconstruction.
 
Different forms for $G_{\rm L}$ can be defined which regularize the source reconstruction in different ways. The 
`Constant` regularization scheme used in this example applies gradient regularization (equation 14 WD03):

 $G_{\rm L} = \sum_{\rm  i}^{I} \sum_{\rm  n=1}^{N}  [s_{i} - s_{i, v}]$

This regularization scheme is easier to express in words -- the summation goes to each RectangularMagnification source pixel,
determines all RectangularMagnification source pixels with which it shares a direct vertex (e.g. its neighbors) and penalizes solutions 
where the difference in reconstructed flux of these two neighboring source pixels is large.

The summation does this for all RectangularMagnification pixels, thus it favours solutions where neighboring RectangularMagnification source
pixels reconstruct similar values to one another (e.g. it favours a smooth source reconstruction).

We now define the `regularization matrix`, $H$, which allows us to include this smoothing when we solve for $s$. $H$
has dimensions `(total_source_pixels, total_source_pixels)`.

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

 - non-zero entries indicate that two RectangularMagnification source-pixels are neighbors and therefore are regularized with one 
 another.
 
 - Zeros indicate the two RectangularMagnification source pixels do not neighbor one another.
 
The majority of entries are zero, because the majority of RectangularMagnification source pixels are not neighbors with one another.
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
__Source Reconstruction (s)__

We can now solve the linear system above using NumPy linear algebra. 

Note that the for loop used above to prevent a LinAlgException is no longer required.
"""
reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
By plotting this source reconstruction we can see that regularization has lead us to reconstruct a smoother source,
which actually looks like a galaxy! This also implies we are not over-fitting the noise.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)

mapper_plotter.figure_2d(solution_vector=reconstruction, interpolate_to_uniform=False)

"""
__Image Reconstruction__

Using the reconstructed source pixel fluxes we can map the source reconstruction back to the image plane (via
the `blurred mapping_matrix`) and produce a reconstruction of the image data.
"""
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

We now quantify the goodness-of-fit of our lens model and source reconstruction. 

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for lens modeling consists of five terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + s^{T} H s + \mathrm{ln} \, \left[ \mathrm{det} (F + H) \right] - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right] + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

This expression was first derived by Suyu 2006 (https://arxiv.org/abs/astro-ph/0601493) and is given by equation (19).
It was derived into **PyAutoLens** notation in Dye 2008 (https://arxiv.org/abs/0804.4002) equation (5).

We now explain what each of these terms mean.

__Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `mapped_reconstructed_image_2d` + `lens_light_convolved_image`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not. 

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image 
for, corresponding to a fit with a lower likelihood.
"""
model_image = convolved_image_2d + mapped_reconstructed_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = al.Array2D(values=chi_squared_map, mask=mask)

array_2d_plotter = aplt.Array2DPlotter(array=chi_squared_map)
array_2d_plotter.figure_2d()


"""
__Regularization Term__

The second term, $s^{T} H s$, corresponds to the $\lambda $G_{\rm L}$ regularization term we added to our merit 
function above.

This is the term which sums up the difference in flux of all reconstructed source pixels, and reduces the likelihood of 
solutions where there are large differences in flux (e.g. the source is less smooth and more likely to be 
overfitting noise).

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

They quantify how complex the source reconstruction is, and penalize solutions where *it is more complex*. Reducing 
the `regularization_coefficient` makes the source reconstruction more complex (because a source that is 
smoothed less uses more flexibility to fit the data better).

These two terms therefore counteract the `chi_squared` and `regularization_term`, so as to attribute a higher
`log_likelihood` to solutions which fit the data with a more smoothed and less complex source (e.g. one with a higher 
`regularization_coefficient`).

In **HowToLens** -> `chapter 4` -> `tutorial_4_bayesian_regularization` we expand on this further and give a more
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
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the lens model, by combining the five terms computed above using
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

This process to perform a likelihood function evaluation is what is performed in the `FitImaging` object.
"""
fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(
        use_w_tilde=False, use_border_relocator=True
    ),
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

For the `VoronoiNoInterp` pixelization used in this example, every image-sub pixel maps to a single source Voronoi
pixel. Therefore, the plural use of `pix_indexes` is not required. However, for other pixelizations each sub-pixel
can map to multiple source pixels with an interpolation weight (e.g. `RectangularMagnification` triangulation or a `Voronoi` mesh
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
example, for a `RectangularMagnification` triangulation, every sub-pixel maps to 3 RectangularMagnification triangles based on which triangle
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
 
 - **Source-plane Interpolation**: Using a RectangularMagnification triangulation or RectangularMagnification mesh with natural neighbor interpolation
 to pair each image (sub-)pixel to multiple source-plane pixels with interpolation weights.
 
 - **Source Morphology Pixelization Adaption**: Adapting the pixelization such that is congregates source pixels around
 the source's brightest regions, as opposed to the magnification-based pixelization used here.
 
 - **Luminosity Weighted Regularization**: Using an adaptive regularization coefficient which adapts the level of 
 regularization applied to the source based on its luminosity.
"""

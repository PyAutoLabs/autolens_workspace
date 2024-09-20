"""
Over Sampling
=============

Throughout the workspace, we have created 2D grids of (y,x) coordinates and input them into light profiles to
compute their image.

This calculates how much of the light profile's emission is observed with every 2D pixel defined on the grid.

However, there is a problem. If we only input the (y,x) coordinates at the centre of every pixel, we are not
evaluating how the entire light profile is observed within that pixel. If the light profile has a very steep gradient
in intensity from one edge of the pixel to the other, only evaluating the intensity at the centre of the pixel will
not give an accurate estimate of the total amount of light that falls within that pixel.

Over-sampling addresses this problem. Instead of evaluating the light profile at the centre of every pixel, we
evaluate it using a sub-grid of coordinates within every pixel and take the average of the intensity values.
Provided the sub-grid is high enough resolution that it "over-samples" the light profile within the pixel enough, this
will give an accurate estimate of the total intensity within the pixel.

__Default Over-Sampling__

Examples throughout the workspace use a default over-sampling set up that should ensure accurate results for any
analysis you have done. This default over-sampling is as follows:

- When evaluating the image of the foreground lens galaxy, an adaptive over sampling grid is used which uses sub
grids of size 32 x 32 in the central regions of the image, 4x4 further out and 2x2 beyond that.

- When evaluating the image of the source galaxy, no over-sampling (e.g. a 1 x 1 subgrid) is performed but instead
cored light profiles for the source are used which can be evaluated accurate without over-sampling.

This guide will explain why these choices were made for the default over-sampling behaviour.

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Illustration__

To illustrate over sampling, lets first create a uniform grid which does not over-sample the pixels, using 
the `over_sampling` input.

We input an `OverSamplingUniform` object, which for every pixel on the grid over-samples it using a uniform sub-grid 
with dimensions specified by the input `sub_size`. 

For example, the input below uses `sub_size=1`, therefore each pixel is split into a sub-grid of size 
`sub_size x sub_size` = `1 x 1`. This is equivalent to not over-sampling the grid at all.  
"""
grid_sub_1 = al.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingUniform(sub_size=1),
)

"""
We now plot the grid, over laying a uniform grid of pixels to illustrate the area of each pixel within which we
want light profile intensities to be computed.
"""
mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Grid Without Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid_sub_1, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True)

"""
We now create and plot a uniform grid which does over-sample the pixels, using a `sub_size` of 2.

The image shows that each pixel is now split into a 2x2 sub-grid of coordinates, which will be used to compute the
intensity of the light profile and therefore more accurately estimate the total intensity within each pixel if 
there is a significant gradient in intensity within the pixel.
"""
grid_sub_2 = al.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingUniform(sub_size=2),
)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Grid With 2x2 Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid_sub_2, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True, plot_over_sampled_grid=True)

print(grid_sub_1)
print(grid_sub_2.over_sampler.over_sampled_grid)

"""
__Numerics__

Lets quickly check how the sub-grid is defined and stored numerically.

The first four pixels of this sub-grid correspond to the first four sub-pixels in the first pixel of the grid. 

The top-left pixel image above shows how the sub-pixels are spaced within the pixel. 

The `grid_sub_2` object has the same shape as the `grid_sub_1` object, and its coordinates are identical. The 
`grid_sub_2` object therefore does not naturally store the sub-pixel coordinates:
"""
print("(y,x) pixel 0 of grid_sub_1:")
print(grid_sub_1[0])
print("(y,x) pixel 0 of grid_sub_2:")
print(grid_sub_2[0])

"""
To numerically access the sub-pixel coordinates, we use the `over_sampler.over_sampled_grid` property of the grid,
which uses the input `sub_size` to create a grid with the sub-pixel coordinates.

We use this below and show that the grid created, has a shape of (400, 2), where the 400 corresponds to the 20x20
sub-pixels of the original 10x10 grid.

Notably, the grid is not stored in its native shape of (20, 20, 2) but instead as a 1D array of shape (400, 2).
Below, we will explain why this is the case.
"""
over_sampled_grid = grid_sub_2.over_sampler.over_sampled_grid

print("Over-sampled grid shape:")
print(over_sampled_grid.shape)

"""
We now confirm that the first four sub-pixels of the over-sampled grid correspond are contained within the 
first pixel of the grid.
"""
print("(y,x) pixel 0 (of original grid):")
print(grid_sub_2[0])
print("(y,x) sub-pixel 0 (of pixel 0):")
print(over_sampled_grid[0])
print("(y,x) sub-pixel 1 (of pixel 0):")
print(over_sampled_grid[1])
print("(y,x) sub-pixel 2 (of pixel 0):")
print(over_sampled_grid[2])
print("(y,x) sub-pixel 3 (of pixel 0):")
print(over_sampled_grid[3])

"""
Numerically, the over-sampled grid contains the sub-pixel coordinates of every pixel in the grid, going from the 
first top-left pixel right and downwards to the bottom-right pixel. 

So the pixel to the right of the first pixel is the next 4 sub-pixels in the over-sampled grid, and so on.

__Images__

We now use over-sampling to compute the image of a Sersic light profile, which has a steep intensity gradient
at its centre which a lack of over-sampling does not accurately capture.

We create the light profile, input the two grids (with `sub_size=1` and `sub_size=2`) and compute the image of the
light profile using each grid. We then plot the residuals between the two images in order to show the difference
between the two images and thus why over-sampling is important.
"""
light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=1.0,
    effective_radius=0.2,
    sersic_index=3.0,
)

image_sub_1 = light.image_2d_from(grid=grid_sub_1)
image_sub_2 = light.image_2d_from(grid=grid_sub_2)

plotter = aplt.Array2DPlotter(
    array=image_sub_1,
)
plotter.set_title("Image of Serisc Profile")
plotter.figure_2d()

residual_map = image_sub_2 - image_sub_1

plotter = aplt.Array2DPlotter(
    array=residual_map,
)
plotter.set_title("Residuals Due to Lack of Over-Sampling")
plotter.figure_2d()


"""
In the central 4 pixels of the image, the residuals are large due to the steep intensity gradient of the Sersic
profile at its centre. 

The gradient in these pixels is so steep that evaluating the intensity at the centre of the pixel, without over 
sampling, does not accurately capture the total intensity within the pixel.

At the edges of the image, the residuals are very small, as the intensity gradient of the Sersic profile is very 
shallow and it is an accurate approximation to evaluate the intensity at the centre of the pixel.

The absolute value of the central residuals are 0.74, however it is difficult to assess whether this is a large or
small value. We can quantify this by dividing by the evaluated value of the Sersic image in each pixel in order
to compute the fractional residuals.
"""
fractional_residual_map = residual_map / image_sub_2

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)
plotter.set_title("Fractional Residuals Due to Lack of Over-Sampling")

plotter.figure_2d()

"""
The fractional residuals in the centre exceed 0.1, or 10%, which is a significant error in the image and
demonstrates why over-sampling is important.

Lets confirm sub-griding can converge to central residuals that are very small.

The fractional residuals with high levels of over-sampling are below 0.01, or 1%, which is sufficiently accurate
for most scientific purposes (albeit you should think carefully about the level of over-sampling you need for
your specific science case).
"""
grid_sub_16 = al.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingUniform(sub_size=16),
)
grid_sub_32 = al.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingUniform(sub_size=32),
)

image_sub_16 = light.image_2d_from(grid=grid_sub_16)
image_sub_32 = light.image_2d_from(grid=grid_sub_32)

residual_map = image_sub_32 - image_sub_16

plotter = aplt.Array2DPlotter(
    array=residual_map,
)
plotter.set_title("Over-Sampling Reduces Residuals")
plotter.figure_2d()

fractional_residual_map = residual_map / image_sub_32

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)
plotter.set_title("Fractional Residuals With Over-Sampling")
plotter.figure_2d()

"""
__Iterative Over-Sampling__

We have shown that over-sampling is important for accurate image evaluation. However, there is a major drawback to
over-sampling, which is that it is computationally expensive. 

For example, for the 32x32 over-sampled grid above, 1024 sub-pixels are used in every pixel, which must all be 
evaluated using the Sersic light profile. The calculation of the image is therefore at least 1000 times slower than if
we had not used over-sampling.

Speeding up the calculation is crucial for model-fitting where the image is evaluated many times to fit the
model to the data.

Fortunately, there is an obvious solution to this problem. We saw above that the residuals rapidly decrease away
from the centre of the light profile. Therefore, we only need to over-sample the central regions of the image,
where the intensity gradient is steep, and can use much lower levels of over-sampling away from the centre.

The `OverSamplingIterate` object performs this iterative over-sampling by performing the following steps:

 1) It computes the image using a low level of over-sampling (e.g. `sub_size=1`).
 2) It computes another image using a user input higher level of over-sampling (e.g. `sub_size=2`).
 3) It computes the fractional residuals between the two images.
 4) If the fractional residuals are above a threshold input by the user, it increases the level of over-sampling
    in only those pixels where the residuals are above the threshold.
 5) It then repeats this process using higher and higher levels of over-sampling in pixels which have not met the
    accuracy threshold, until all pixels do or the user-defined maximum level of over-sampling is reached.

This object is used throughout the workspace to simulate images of galaxies in the `simulators` package

We now use this object and confirm that it can compute the image of the Sersic profile accurately by comparing
to the image computed using a 32x32 degree of over-sampling.

The object has the following inputs:

 - `fractional_accuracy`: The fractional accuracy threshold the iterative over-sampling aims to meet. The value of
    0.9999 means that the fractional residuals in every pixel must be below 0.0001, or 0.01%.
 
  - `sub_steps`: The sub-size values that are iteratively increased which control the level of over-sampling used to
    compute the image.
"""
grid_iterate = al.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingIterate(
        fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    ),
)

image_iterate = light.image_2d_from(grid=grid_iterate)

residual_map = image_sub_32 - image_iterate

fractional_residual_map = residual_map / image_sub_32

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)
plotter.set_title("Fractional Residuals Using Iterative Over-Sampling")
plotter.figure_2d()

"""
The fractional residuals are below 0.01% in every pixel, showing that the `OverSamplingIterate` object has
accurately computed the image of the Sersic profile.

__Manual Adaptive Grid__

The iterative over-sampling is a powerful tool, but it is still computationally expensive. This is because it
has to evaluate the light profile image many times at increasing levels of over-sampling until the fractional
residuals are below the threshold.

For modeling, where the image is evaluated many times to fit the model to the data, this is not ideal. A faster
approach which reap the benefits of over-sampling is to manually define a grid which over-samples the regions of
the image where the intensity gradient is expected steep, and uses low levels of over-sampling elsewhere.

For an unlensed galaxy (e.g. the foreground lens galaxy), this is simple. The intensity gradient is known to be steep 
at its centre, therefore we just require a high level of over-sampling at its centre.

Below, we define a grid which uses a 24 x 24 sub-grid within the central 0.3" of pixels, uses a 8 x 8 grid between
0.3" and 0.6" and a 2 x 2 grid beyond that. By comparing this manual adaptive grid to the iterative over-sampling
grid, we can confirm that the adaptive grid provides a good balance between accuracy and computational efficiency.

Modeling uses masked grids, therefore the grid we use below is computed via a circular mask.

Throughout the modeling examples in the workspace, we use this adaptive grid to ensure that the image of the lens
galaxy is evaluated accurately and efficiently.
"""
mask = al.Mask2D.circular(shape_native=(40, 40), pixel_scales=0.1, radius=5.0)

grid = al.Grid2D.from_mask(mask=mask)

grid_adaptive = al.Grid2D(
    values=grid,
    mask=mask,
    over_sampling=al.OverSamplingUniform.from_radial_bins(
        grid=grid, sub_size_list=[32, 8, 2], radial_list=[0.3, 0.6]
    ),
)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Grid With Adaptive Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid_adaptive, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True, plot_over_sampled_grid=True)

image_adaptive = light.image_2d_from(grid=grid_adaptive)
image_sub_32 = light.image_2d_from(grid=grid_sub_32)

residual_map = image_adaptive - image_sub_32

fractional_residual_map = residual_map / image_sub_32

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)
plotter.set_title("Adaptive Over-Sampling Residuals")
plotter.figure_2d()

"""
__Default Over-Sampling__

The iterative over-sampling used above is accurate, but it is computationally expensive and not ideal
for tasks like lens modeling which require the image to be evaluated many times.

The default over-sampling (e.g. if you do not manually input an over-sampling object) is created as follows:

 1) Extract the centre of the light or mass profiles being evaluated (a value of (0.5", 0.5") is used below).
 2) Use the name of the light or mass profile to load pre defined over sampling values from `config/grid.yaml`.
 3) Use these values to set up an adaptive over-sampling grid around the profile centre, which by default contains 3 
    sub-size levels, a 32 x 32 sub-grid in the central regions, a 4 x 4 sub-grid further out and a 2 x 2 sub-grid 
    beyond that.

This default behaviour occurs whenever a light or mass profile is evaluated using a grid, and therefore you can be 
confident that all calculations you have performed are over-sampled accurately and efficiently.

We illustrate and plot this default adaptive over sampling grid below.
"""
grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1, over_sampling=None)

over_sampling = al.OverSamplingUniform.from_adaptive_scheme(
    grid=grid, name="Sersic", centre=(0.5, 0.5)
)

grid = al.Grid2D.uniform(
    shape_native=(40, 40), pixel_scales=0.1, over_sampling=over_sampling
)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Grid With Default Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True, plot_over_sampled_grid=True)

"""
This default over sampling scheme only works because the input grid is uniform, because this means the centre of the
light or mass profile is where in the grid high levels of over sampling are required. This assumption is not true of a 
grid which has been deflected via a mass model. 

This adaptive over-sampling scheme therefore cannot be used for evaluating lensed sources and is only used to 
evaluate the image of the foreground lens galaxy.

__Multiple Lens Galaxies__

The analysis may contain multiple lens galaxies, each of which must be over-sampled accurately. 

The default over-sampling can already handle this, as it uses the centre of each galaxy to set up the adaptive
over-sampling grid. It does this for every light profile of every galaxy in the analysis, thus different adaptive
grids will be used if the galaxies are at different centres.

We therefore recommend you always use the default over-sampling for multi-galaxy modeling.

__Ray Tracing__

So far, we have evaluated the image of a light profile using over-sampling on an unlensed uniform grid. 

For lensing calculations, the grid is ray-traced via a mass model to an irregular grid in the source plane, which
as discussed a moment ago means that the adaptive over-sampling scheme cannot be used.

We now illustrate using over-sampling with a mass profile, noting that for lensing:

 1) The fraction residuals due to differing over-sampling levels now occur in the lensed source's brightest multiply 
 imaged pixels. 
 
 2) It is the combination of a rapidly changing source light profile and the magnification pattern of the mass model
    which requires over sampling. The mass model focuses many image-pixels to the source's brightest regions.
"""
mass = al.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.1, 0.0), einstein_radius=1.0)

light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=1.0,
    effective_radius=0.2,
    sersic_index=3.0,
)

lens = al.Galaxy(redshift=0.5, mass=mass)

source = al.Galaxy(redshift=1.0, bulge=light)

tracer = al.Tracer(galaxies=[lens, source])

image_sub_1 = tracer.image_2d_from(grid=grid_sub_1)
image_sub_2 = tracer.image_2d_from(grid=grid_sub_2)

plotter = aplt.Array2DPlotter(
    array=image_sub_1,
)
plotter.set_title("Image of Lensed Source")
plotter.figure_2d()

residual_map = image_sub_2 - image_sub_1

fractional_residual_map = residual_map / image_sub_2

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)
plotter.set_title("Lensing Low Over-Sampling Residuals")
plotter.figure_2d()

"""
The iterative over sampling grid can be used to perform lensing calculations accurately and efficiently.

This reason this grid works, but the default adaptive over-sampling grid does not, is because the iterative grid
recomputes the deflection angles and performs th ray-tracing at every level of over-sampling. This default grid
cannot be defined in a way that does this, as one does not know where the brightest regions of the source will be
before the ray-tracing is performed.
"""
image_sub_32 = tracer.image_2d_from(grid=grid_sub_32)
image_iterate = tracer.image_2d_from(grid=grid_iterate)

residual_map = image_sub_32 - image_iterate

fractional_residual_map = residual_map / image_sub_32

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)
plotter.set_title("Lensing High Over-Sampling Residuals")
plotter.figure_2d()

"""
__Default Ray Tracing__

There is not default adaptive over sampling scheme that can work for ray tracing in an efficient and accurate manner.

This is why the majority of workspace examples use cored light profiles for the source galaxy. A cored light profile
does not rapidly change in its central regions, and therefore can be evaluated accurately without over-sampling.

__Dataset & Modeling__

Throughout this guide, grid objects have been used to compute the image of light and mass profiles and illustrate
over sampling.

If you are performing calculations with imaging data or want to fit a lens model to the data with a specific
over-sampling level, the following API is used:
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

# This can be any of the over-sampling objects we have used above.

dataset = dataset.apply_over_sampling(
    over_sampling=al.OverSamplingDataset(uniform=al.OverSamplingUniform(sub_size=4))
)

"""
__Search Chaining__

To efficiently perform lens modeling with appropriate over-sampling for the lensed source, we use the advanced
lens modeling feature search chaining.

An example of how to use search chaining to over sample sources efficient is provided in 
the `autolens_workspace/*/imaging/advanced/chaining/over_sampling.ipynb` example.

__Pixelization__

Source galaxies can be reconstructed using pixelizations, which discretize the source's light onto a mesh,
for example a Voronoi mesh.

Over sampling is used by pixelizations in an analogous way to light profiles. By default, a 4 x 4 sub-grid is used,
whereby every image pixel is ray-traced on its 4 x 4 sub grid to the source mesh and fractional mappings are computed.

This is explained in more detail in the pixelization examples.

Here is an example of how to change the over sampling applied to a pixelization for a lens model fit:
"""
dataset = dataset.apply_over_sampling(
    over_sampling=al.OverSamplingDataset(
        pixelization=al.OverSamplingUniform(sub_size=4)
    )
)

"""
Finish.
"""

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

- When evaluating the image of a galaxy, an adaptive over sampling grid is used which uses sub grids of size 8 x 8 
in the central regions of the image, 4x4 further out and 1x1 beyond that.

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

To illustrate over sampling, lets first create a uniform grid which does not over sample the pixels, using 
the `over_sample_size` input.

The input below uses `over_sample_size=1`, therefore each pixel is split into a sub-grid of 
size  `over_sample_size x over_sample_size` = `1 x 1`. This means the light profile is evaluated once at the centre of each pixel, 
which is equivalent to not over-sampling the grid at all.  
"""
grid_sub_1 = al.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sample_size=1,
)

"""
We now plot the grid, over laying a uniform grid of pixels to illustrate the area of each pixel within which we
want light profile intensities to be computed.
"""
mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Grid Without Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid_sub_1, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True)

"""
We now create and plot a uniform grid which does over-sample the pixels, by inputting `over_sample_size=2`.
"""
grid_sub_2 = al.Grid2D.uniform(
    shape_native=(40, 40),
    pixel_scales=0.1,
    over_sample_size=2,
)

"""
If we print `grid_sub_2` and its shape, we will find it is actually identical to `grid_sub_1`, despite the change
in `over_sample_size`:
"""
print(grid_sub_1)
print(grid_sub_2)
print(grid_sub_1.shape)
print(grid_sub_2.shape)

"""
This is because the over sampled version of the grid is stored in a separate attribute, called `over_sampled`,
which we print below.

We see that for `grid_sub_1` and `grid_sub_2` the `over_sampled` grids are different, with the over sampled grid for
`grid_sub_2` containing four times as many entries corresponding to each pixel being sub-gridded in a 2 x 2 shape.
"""
print(grid_sub_1.over_sampled)
print(grid_sub_2.over_sampled)
print(grid_sub_1.over_sampled.shape)
print(grid_sub_2.over_sampled.shape)

"""
We now plot the over sampled grid over the image, showing that each pixel is now split into a 2x2 sub-grid of 
coordinates. 

These are used to compute the intensity of the light profile and therefore more accurately estimate the total 
intensity within each pixel if there is a significant gradient in intensity within the pixel.

In the code below, it is the input `plot_over_sampled_grid=True` which ensures we plot the over sampled grid.
"""
mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Grid With 2x2 Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid_sub_2, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True, plot_over_sampled_grid=True)

"""
__Numerics__

Lets quickly check how the sub-grid is defined and stored numerically.

The first four pixels of this sub-grid correspond to the first four sub-pixels in the first pixel of the grid. 

The top-left pixel image above shows how the sub-pixels are spaced within the pixel. 
"""
print("(y,x) pixel 0 of grid_sub_1:")
print(grid_sub_1.over_sampled[0])
print("(y,x) pixel 0 of grid_sub_2:")
print(grid_sub_2.over_sampled[0])

"""
We now confirm that the first four sub-pixels of the over-sampled grid correspond are contained within the 
first pixel of the grid.
"""
print("(y,x) pixel 0 (of original grid):")
print(grid_sub_2[0])
print("(y,x) sub-pixel 0 (of pixel 0):")
print(grid_sub_2.over_sampled[0])
print("(y,x) sub-pixel 1 (of pixel 0):")
print(grid_sub_2.over_sampled[1])
print("(y,x) sub-pixel 2 (of pixel 0):")
print(grid_sub_2.over_sampled[2])
print("(y,x) sub-pixel 3 (of pixel 0):")
print(grid_sub_2.over_sampled[3])

"""
Numerically, the over-sampled grid contains the sub-pixel coordinates of every pixel in the grid, going from the 
first top-left pixel right and downwards to the bottom-right pixel. 

So the pixel to the right of the first pixel is the next 4 sub-pixels in the over-sampled grid, and so on.

__Images__

We now use over-sampling to compute the image of a Sersic light profile, which has a steep intensity gradient
at its centre which a lack of over-sampling does not accurately capture.

We create the light profile, input the two grids (with `over_sample_size=1` and `over_sample_size=2`) and compute the image of the
light profile using each grid. We then plot the residuals between the two images in order to show the difference
between the two images and thus why over-sampling is important.

Over sampling occurs automatically when a grid is input into a function like `image_2d_from`, therefore internally 
the line of code, `image_sub_2 = light.image_2d_from(grid=grid_sub_2)`, is evaluating the light profile using the
2 x 2 oversampled grid and internally binning it up in to fully perform over sampling.
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
plotter.set_title("Image of Sersic Profile")
plotter.figure_2d()

residual_map = image_sub_2 - image_sub_1

plotter = aplt.Array2DPlotter(
    array=residual_map,
)
plotter.set_title("Over-Sampling Residuals")
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
plotter.set_title("Fractional Over-Sampling Residuals")

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
    shape_native=(40, 40), pixel_scales=0.1, over_sample_size=16
)
grid_sub_32 = al.Grid2D.uniform(
    shape_native=(40, 40), pixel_scales=0.1, over_sample_size=32
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
__Adaptive Over Sampling__

We have shown that over-sampling is important for accurate image evaluation. However, there is a major drawback to
over-sampling, which is that it is computationally expensive. 

For example, for the 32x32 over-sampled grid above, 1024 sub-pixels are used in every pixel, which must all be 
evaluated using the Sersic light profile. The calculation of the image is therefore at least 1000 times slower than if
we had not used over-sampling.

Speeding up the calculation is crucial for model-fitting where the image is evaluated many times to fit the
model to the data.

Fortunately, there is a solution to this problem. We saw above that the residuals rapidly decrease away
from the centre of the light profile. Therefore, we only need to over-sample the central regions of the image,
where the intensity gradient is steep. We can use lower levels of over-sampling away from the centre, which
will be fast to evaluate.

Up to now, the `over_sample_size` input has been an integer, however it can also be an `ndarray` of values corresponding
to each pixel. Below, we create an `ndarray` of values which are high in the centre, but reduce to 2 at the outskirts,
therefore providing high levels of over sampling where we need it whilst using lower values which are computationally
fast to evaluate at the outskirts.

Specifically, we define a 24 x 24 sub-grid within the central 0.3" of pixels, uses a 8 x 8 grid between
0.3" and 0.6" and a 2 x 2 grid beyond that. 
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid_sub_1, sub_size_list=[24, 8, 2], radial_list=[0.3, 0.6]
)

grid_adaptive = al.Grid2D.no_mask(
    values=grid_sub_1.native,
    pixel_scales=grid_sub_1.pixel_scales,
    over_sample_size=over_sample_size,
)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Adaptive Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid_adaptive, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True, plot_over_sampled_grid=True)

print(over_sample_size)


"""
Modeling uses masked grids, therefore the code below shows how we would create this adaptive over sample grid via 
a circular mask, which can be used for modeling.

Throughout the modeling examples in the workspace, we use this adaptive grid to ensure that the image of the
galaxy is evaluated accurately and efficiently.
"""
mask = al.Mask2D.circular(shape_native=(40, 40), pixel_scales=0.1, radius=5.0)

grid = al.Grid2D.from_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid, sub_size_list=[24, 8, 2], radial_list=[0.3, 0.6]
)

grid_adaptive = al.Grid2D(values=grid, mask=mask, over_sample_size=over_sample_size)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Adaptive Over-Sampling"))

grid_plotter = aplt.Grid2DPlotter(grid=grid_adaptive, mat_plot_2d=mat_plot)
grid_plotter.figure_2d(plot_grid_lines=True, plot_over_sampled_grid=True)

"""
We can compare this adaptive grid to the grid with over sampling of 32 x 32 to confine it produces low amounts
of residuals.
"""
image_adaptive = light.image_2d_from(grid=grid_adaptive)
image_sub_32 = light.image_2d_from(grid=grid_sub_32)

residual_map = image_adaptive - image_sub_32

fractional_residual_map = residual_map / image_sub_32

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)

plotter.set_title("Adaptive Fractional Residuals")
plotter.figure_2d()

"""
__Default Over-Sampling__

The default over-sampling scheme used by the source code is 4 x 4 uniform over sampling over the whole image. 

A uniform scheme is used, instead of the adaptive scheme above, because the adaptive scheme requires input knowledge of 
where the centre of the galaxy is (e.g. above the centre is at (0.0", 0.0").

Uniform over sampling is precise enough for many calculations, especially when you are simply performing quick 
calculations to investigate a problem. However, for detailed calculations you must ensure that high enough
levels of over sampling are used.

For modeling, all example scripts begin by switching to an adaptive over sampling scheme, as modeling assumes
the centre of the lens galaxy is at (0.0", 0.0").

__Multiple Lens Galaxies__

The analysis may contain multiple lens galaxies, each of which must be over-sampled accurately. 


There are two approaches you can take to over sampling multi-galaxy systems:

1) Use a high level of uniform over sampling over the full image.

2) Use an adaptive over sampling scheme with multiple centres of high over sampling levels, with the API shown below
  for two galaxies with centres (1.0, 0.0) and (-1.0, 0.0).
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid_sub_1,
    sub_size_list=[24, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(1.0, 0.0), (-1.0, 0.0)],
)

"""
__Ray Tracing__

So far, we have evaluated the image of a light profile using over-sampling on an unlensed uniform grid. 

For lensing calculations, the grid is ray-traced via a mass model to an irregular grid in the source plane.

The over sampling of lensed images is therefore describe as follows: 

1) Splits each image-pixel into a sub-grid of pixels in the image-plane.
2) Ray trace this sub-grid of pixels using the mass model.
3) Evaluate the source light of each sub-pixel in the source-plane.
4) Bin up the values to evaluate the over sampled values.

For lensing calculations, over sampling therefore requires us to ray-trace (and therefore compute the deflecitons angles
of) many more (y,x) coordinates!

We now illustrate using over-sampling with a mass profile, noting that for lensing:

1) The fractional residuals due to differing over-sampling levels now occur in the lensed source's brightest multiply 
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
plotter.set_title("Source Image 1x1")
plotter.figure_2d()

residual_map = image_sub_2 - image_sub_1

fractional_residual_map = residual_map / image_sub_2

plotter = aplt.Array2DPlotter(
    array=fractional_residual_map,
)
plotter.set_title("Fractional Residuals")
plotter.figure_2d()

"""
__Default Ray Tracing__

By assuming the lens galaxy is near (0.0", 0.0"), it was simple to set up an adaptive over sampling grid which is
applicable to all strong lens dataset.

An adaptive oversampling grid cannot be defined for the lensed source because its light appears in different regions of 
the image plane for each dataset. For this reason, most workspace examples utilize cored light profiles for the 
source galaxy. Cored light profiles change gradually in their central regions, allowing accurate evaluation without 
requiring oversampling.

__Adaptive Over Sampling__

There is a way to set up an adaptive over sampling grid for a lensed source, however it requries one to use and
understanding the advanced lens modeling feature search chaining.

An example of how to use search chaining to over sample sources efficient is provided in 
the `autolens_workspace/*/imaging/advanced/chaining/over_sampling.ipynb` example.

__Dataset & Modeling__

Throughout this guide, grid objects have been used to compute the image of light and mass profiles and illustrate
over sampling.

If you are performing calculations with imaging data or want to fit a model to the data with a specific
over-sampling level, the `apply_over_sampling` method is used to update the over sampling scheme of the dataset.

The grid this is applied to is called `lp`, to indicate that it is the grid used to evaluate the emission of light
profiles for which this over sampling scheme is applied.
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

dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

"""
__Pixelization__

Source galaxies can be reconstructed using pixelizations, which discretize the source's light onto a mesh,
for example a Voronoi mesh.

Over sampling is used by pixelizations in an analogous way to light profiles. By default, a 4 x 4 sub-grid is used,
whereby every image pixel is ray-traced on its 4 x 4 sub grid to the source mesh and fractional mappings are computed.

A different grid and over sampling scheme is applied to light profiles and pixelizations, which is why
there are separate inputs called `lp` and `pix`.

This is explained in more detail in the pixelization examples.

Here is an example of how to change the over sampling applied to a pixelization for a lens model fit:
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=4
)

"""
Finish.
"""

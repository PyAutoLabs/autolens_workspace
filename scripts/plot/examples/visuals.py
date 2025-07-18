"""
Plots: Visuals
==============

This example illustrates the API for adding visuals to plots and customizing their appearance.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how visuals work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, tracer, data) used to illustrate plotting.
**

__Setup__

To illustrate plotting, we require standard objects like a grid, tracer and dataset.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path

import autolens as al
import autolens.plot as aplt

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.2, 0.2)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-1.0, 0.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-1.0, 0.0), einstein_radius=0.8, ell_comps=(0.2, 0.2)
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.2, 0.2), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)


tracer_x2 = al.Tracer(
    galaxies=[lens_galaxy, lens_galaxy_1, source_galaxy, source_galaxy_1]
)

dataset_path = Path("dataset") / "slacs" / "slacs1430+4105"
data_path = dataset_path / "data.fits"
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.03)

"""
__Critical Curves__

The tangential and radial critical curves are plotted over data as lines.
"""
tangential_critical_curve_list = tracer.tangential_critical_curve_list_from(grid=grid)
radial_critical_curves_list = tracer.radial_critical_curve_list_from(grid=grid)

visuals = aplt.Visuals2D(
    tangential_critical_curves=tangential_critical_curve_list,
    radial_critical_curves=radial_critical_curves_list,
)

image = tracer.image_2d_from(grid=grid)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the tangential and radial critical curves are customized using 
`TangentialCriticalCurvesPlot`  and `RadialCriticalCurvesPlot` objects.

To plot the critical curves this object wraps the following matplotlib method:

 https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
"""
tangential_critical_curves_plot = aplt.TangentialCriticalCurvesPlot(
    linestyle="--", linewidth=10, c="k"
)
radial_critical_curves_plot = aplt.RadialCriticalCurvesPlot(
    linestyle="--", linewidth=10, c="w"
)

mat_plot = aplt.MatPlot2D(
    tangential_critical_curves_plot=tangential_critical_curves_plot,
    radial_critical_curves_plot=radial_critical_curves_plot,
)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
If a `Tracer` has multiple mass profiles, it may also have multiple tangential and radial critical curves, which
are plotted in different colors by default.

This is the case for `tracer_x2`, which we use below.

By specifying two colors to the `TangentialCriticalCurvesPlot` and `RadialCriticalCurvesPlot` objects each tangential 
and critical_curve will be plotted in different colors.

By inputting the same alternating colors for the critical curves and caustics each pair will appear the same color 
on image-plane and source-plane figures.
"""
radial_critical_curves_plot = aplt.RadialCriticalCurvesPlot(c=["w", "b"])

mat_plot = aplt.MatPlot2D(
    tangential_critical_curves_plot=tangential_critical_curves_plot,
    radial_critical_curves_plot=radial_critical_curves_plot,
)

tangential_critical_curve_list = tracer_x2.tangential_critical_curve_list_from(
    grid=grid
)
radial_critical_curves_list = tracer_x2.radial_critical_curve_list_from(grid=grid)

visuals = aplt.Visuals2D(
    tangential_critical_curves=tangential_critical_curve_list,
    radial_critical_curves=radial_critical_curves_list,
)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_x2, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
__Caustics__

The tangential and radial caustics are plotted over data as lines.
"""
tangential_caustic_list = tracer.tangential_caustic_list_from(grid=grid)
radial_caustics_list = tracer.radial_caustic_list_from(grid=grid)

visuals = aplt.Visuals2D(
    tangential_caustics=tangential_caustic_list,
    radial_caustics=radial_caustics_list,
)
image = tracer.image_2d_from(grid=grid)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the tangential and radial critical curves are customized using 
`TangentialCausticsPlot`  and `RadialCausticsPlot` objects.

To plot the critical curves this object wraps the following matplotlib method:

 https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
"""
tangential_caustics_plot = aplt.TangentialCausticsPlot(
    linestyle="--", linewidth=10, c="k"
)
radial_caustics_plot = aplt.RadialCausticsPlot(linestyle="--", linewidth=10, c="w")

mat_plot = aplt.MatPlot2D(
    tangential_caustics_plot=tangential_caustics_plot,
    radial_caustics_plot=radial_caustics_plot,
)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(source_plane=True)

"""
If a `Tracer` has multiple mass profiles, it may also have multiple tangential and radial critical curves, which
are plotted in different colors by default. 

This is the case for `tracer_x2`, which we use below.

By specifying two colors to the `TangentialCausticsPlot` and `RadialCausticsPlot` objects each tangential 
and caustic will be plotted in different colors.

By inputting the same alternating colors for the critical curves and caustics each pair will appear the same color 
on image-plane and source-plane figures.
"""
tangential_caustics_plot = aplt.TangentialCausticsPlot(c=["k", "r"])
radial_caustics_plot = aplt.RadialCausticsPlot(c=["w", "b"])

mat_plot = aplt.MatPlot2D(
    tangential_caustics_plot=tangential_caustics_plot,
    radial_caustics_plot=radial_caustics_plot,
)

tangential_caustic_list = tracer.tangential_caustic_list_from(grid=grid)
radial_caustics_list = tracer.radial_caustic_list_from(grid=grid)

visuals = aplt.Visuals2D(
    tangential_caustics=tangential_caustic_list,
    radial_caustics=radial_caustics_list,
)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_x2, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(source_plane=True)

"""
__Multiple Images__

The multiple images of the source in the image-plane and source-plane can be plotted.

The multiple images are computed using the `PointSolver` object.
"""
solver = al.PointSolver(grid=grid)
multiple_images = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

visuals = aplt.Visuals2D(multiple_images=multiple_images)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, visuals_2d=visuals)
tracer_plotter.figures_2d(image=True)

"""
The appearance of the multiple images are customized using a `MultipleImagesScatter` object.

To plot the multiple images this object wraps the following matplotlib method:

https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
multiple_images_scatter = aplt.MultipleImagesScatter(marker="o", c="r", s=150)

mat_plot = aplt.MatPlot2D(multiple_images_scatter=multiple_images_scatter)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
By specifying two colors to the `MultipleImagesScatter` object the multiple images of different source galaxies
are plotted in different colors.

Below, we compute the multiple images for two different centres, as if there are two sources in the source-plane,
and plot them separately.
"""
multiple_images_0 = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)
multiple_images_1 = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy_1.bulge.centre
)

multiple_images = [multiple_images_0, multiple_images_1]

visuals = aplt.Visuals2D(multiple_images=multiple_images)

multiple_images_scatter = aplt.MultipleImagesScatter(c=["r", "w"], s=150)

mat_plot = aplt.MatPlot2D(multiple_images_scatter=multiple_images_scatter)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)


"""
__Positions__

Positions can be plotted over an image, which have an API closely mirroring that of multiple images above.

The most common use case for positions is plotting the (y,x) coordinates of the lensed source emission which are
used to apply a penalty to the likelihood function.

However, the code below can be used to mark any interesting points over an image.
"""
positions = al.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

visuals = aplt.Visuals2D(positions=positions)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the positions is customized using a `Scatter` object.

To plot the positions this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
positions_scatter = aplt.PositionsScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(positions_scatter=positions_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Light Profile Centres__

The centres of all light profiles in a tracer (or other object, like a galaxy) can be extracted and plotted.

We are producing an image-plane plot of the light profile centres, therefore we extract the centres via
`tracer.galxies[0]`.
"""
light_profile_centres = tracer.galaxies[0].extract_attribute(
    cls=al.LightProfile, attr_name="centre"
)

visuals = aplt.Visuals2D(light_profile_centres=light_profile_centres)
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
The source-plane centres can also be extracted and plotted, via `tracer.galaxies[-1]`
"""
light_profile_centres_source_plane = tracer.galaxies[-1].extract_attribute(
    cls=al.LightProfile, attr_name="centre"
)

visuals = aplt.Visuals2D(light_profile_centres=light_profile_centres_source_plane)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(source_plane=True)

"""
The appearance of the light profile centres are customized using a `LightProfileCentresScatter` object.

To plot the light profile centres this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
light_profile_centres = tracer_x2.galaxies[0].extract_attribute(
    cls=al.LightProfile, attr_name="centre"
)

light_profile_centres_scatter = aplt.LightProfileCentresScatter(
    marker="o", c="r", s=150
)
mat_plot = aplt.MatPlot2D(light_profile_centres_scatter=light_profile_centres_scatter)
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_x2, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
By specifying two colors to the `LightProfileCentresScatter` object the light profile centres of each plane
are plotted in different colors.

The plot below uses the `tracer_x2` object which consists of multiple galaxies with multiple light profiles.
"""
light_profile_centres_scatter = aplt.LightProfileCentresScatter(c=["r", "w"], s=150)

mat_plot = aplt.MatPlot2D(light_profile_centres_scatter=light_profile_centres_scatter)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_x2, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)


"""
__Mass Profile Centre__

The centres of all mass profiles in a tracer (or other object, like a galaxy) can be extracted and plotted.

We are producing an image-plane plot of the mass profile centres, therefore we extract the centres via
`tracer.galaxies[0]`.
"""
mass_profile_centres = tracer.extract_attribute(
    cls=al.mp.MassProfile, attr_name="centre"
)
visuals = aplt.Visuals2D(mass_profile_centres=mass_profile_centres)
image = tracer.image_2d_from(grid=grid)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the mass profile centres are customized using a `MassProfileCentresScatter` object.

To plot the mass profile centres this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
mass_profile_centres_scatter = aplt.MassProfileCentresScatter(marker="o", c="r", s=150)
mat_plot = aplt.MatPlot2D(mass_profile_centres_scatter=mass_profile_centres_scatter)
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
By specifying two colors to the `MassProfileCentresScatter` object the mass profile centres of each plane
are plotted in different colors.

The plot below uses the `tracer_x2` object which consists of multiple galaxies with multiple mass profiles.
"""
mass_profile_centres = tracer_x2.extract_attribute(
    cls=al.mp.MassProfile, attr_name="centre"
)

mass_profile_centres_scatter = aplt.MassProfileCentresScatter(c=["r", "w"], s=150)

mat_plot = aplt.MatPlot2D(mass_profile_centres_scatter=mass_profile_centres_scatter)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer_x2, grid=grid, mat_plot_2d=mat_plot, visuals_2d=visuals
)
tracer_plotter.figures_2d(image=True)

"""
__Mask__

The mask is plotted over all images by default as black points.

We now show how to manually pass in a mask to plot and customize its appearance.
"""
mask = al.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image_2d = al.Array2D(values=data.native, mask=mask)

visuals = aplt.Visuals2D(mask=mask)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the mask is customized using a `Scatter` object.

To plot the mask this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
mask_scatter = aplt.MaskScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(mask_scatter=mask_scatter)

array_plotter = aplt.Array2DPlotter(
    array=masked_image_2d, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Origin__

We can plot the (y,x) origin on the data to show where the grid is defined from.

By default the origin of (0.0", 0.0") is at the centre of the image.
"""
visuals = aplt.Visuals2D(origin=al.Grid2DIrregular(values=[(1.0, 1.0)]))

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the (y,x) origin coordinates is customized using a `Scatter` object.

To plot these (y,x) grids of coordinates these objects wrap the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html

The example script `plot/mat_wrap/Scatter.py` gives a more detailed description on how to customize its appearance.
"""
origin_scatter = aplt.OriginScatter(marker="o", s=50)

mat_plot = aplt.MatPlot2D(origin_scatter=origin_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Grid__

We can plot a grid of (y,x) coordinates over an image.

We'll use a uniform grid at a coarser resolution than our dataset.
"""
grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.1)

visuals = aplt.Visuals2D(grid=grid)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We customize the grid's appearance using the `GridScatter` `matplotlib wrapper object which wraps the following method(s): 

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
grid_scatter = aplt.GridScatter(c="r", marker=".", s=1)

mat_plot = aplt.MatPlot2D(grid_scatter=grid_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Border__

A border is the `Grid2D` of (y,x) coordinates at the centre of every pixel at the border of a mask. 

A border is defined as a pixel that is on an exterior edge of a mask (e.g. it does not include the inner pixels of 
an annular mask).

Borders are rarely plotted, but are important when it comes to defining the edge of the source-plane for pixelized
source reconstructions, with examples on this topic sometimes plotting the border.
"""
mask = al.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image_2d = al.Array2D(values=data.native, mask=mask)

visuals = aplt.Visuals2D(border=mask.derive_grid.border)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the border is customized using a `BorderScatter` object.

To plot the border this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
border_scatter = aplt.BorderScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(border_scatter=border_scatter)

array_plotter = aplt.Array2DPlotter(
    array=masked_image_2d, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Array Overly__

We can overlay a 2D array over an image.

This is mostly used for dark matter subhalo analysis, where an array of log evidences increases is overlaid images
to show whether a dark matter substructure is detected.
"""
arr = al.Array2D.no_mask(
    values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=0.5
)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We customize the overlaid array using the `ArrayOverlay` matplotlib wrapper object which wraps the following method(s):

To overlay the array this objects wrap the following matplotlib method:

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
"""
array_overlay = aplt.ArrayOverlay(alpha=0.5)

mat_plot = aplt.MatPlot2D(array_overlay=array_overlay)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Patch Overlay__

The matplotlib patch API can be used to plot shapes over an image.

This is used in certain weak lensing plots to visualize galaxy shapes and ellipticities.

To plot a patch on an image, we use the `matplotlib.patches` module. 

In this example, we will use the `Ellipse` patch.
"""
from matplotlib.patches import Ellipse

patch_0 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)
patch_1 = Ellipse(xy=(-2.0, -3.0), height=1.0, width=2.0, angle=1.0)

visuals = aplt.Visuals2D(patches=[patch_0, patch_1])

array_plotter = aplt.Array2DPlotter(array=data)  # , visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can customize the patches using the `Patcher` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/collections_api.html
"""
patch_overlay = aplt.PatchOverlay(
    facecolor=["r", "g"], edgecolor="none", linewidth=10, offsets=3.0
)

mat_plot = aplt.MatPlot2D(patch_overlay=patch_overlay)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Vector Field__

A quiver plot showing vectors (e.g. 2D (y,x) directions at (y,x) coordinates) can be plotted using the `matplotlib`
`quiver` function.

This is often used for weak lensing plots, showing the direction and magnitude of weak lensing infrred at each
galaxy's location.
"""
vectors = al.VectorYX2DIrregular(
    values=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
)
visuals = aplt.Visuals2D(vectors=vectors)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can customize the appearance of the vectors using the `VectorYXQuiver` matplotlib wrapper object which wraps 
the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
"""
quiver = aplt.VectorYXQuiver(
    headlength=1,
    pivot="tail",
    color="w",
    linewidth=10,
    units="width",
    angles="uv",
    scale=None,
    width=0.5,
    headwidth=3,
    alpha=0.5,
)

mat_plot = aplt.MatPlot2D(vector_yx_quiver=quiver)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Half Light Radius__

For 1D plots of a light profile (e.g. radius vs intensity) a 1D line of its half light radius can be plotted.
"""
visuals = aplt.Visuals1D(half_light_radius=tracer.galaxies[0].bulge.half_light_radius)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=tracer.galaxies[0].bulge, grid=grid, visuals_1d=visuals
)
light_profile_plotter.figures_1d(image=True)

"""
The appearance of the half-light radius is customized using a `HalfLightRadiusAXVLine` object.

To plot the half-light radius as a vertical line this wraps the following matplotlib method:

 plt.axvline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
"""
half_light_radius_axvline = aplt.HalfLightRadiusAXVLine(
    linestyle="-.", c="r", linewidth=20
)

mat_plot = aplt.MatPlot1D(half_light_radius_axvline=half_light_radius_axvline)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=tracer.galaxies[0].bulge,
    grid=grid,
    mat_plot_1d=mat_plot,
    visuals_1d=visuals,
)
light_profile_plotter.figures_1d(image=True)

"""
__Einstein Radius__

For 1D plots of a mass profile (e.g. radius vs convergence) a 1D line of its Einstein radius can be plotted.
"""
visuals = aplt.Visuals1D(
    einstein_radius=tracer.galaxies[0].mass.einstein_radius_from(grid=grid)
)

mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=tracer.galaxies[0].mass, grid=grid, visuals_1d=visuals
)
mass_profile_plotter.figures_1d(convergence=True)

"""
The appearance of the einstein radius is customized using a `EinsteinRadiusAXVLine` object.

To plot the einstein radius as a vertical line this wraps the following matplotlib method:

 plt.axvline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
"""
einstein_radius_axvline = aplt.EinsteinRadiusAXVLine(
    linestyle="-.", c="r", linewidth=20
)

mat_plot = aplt.MatPlot1D(einstein_radius_axvline=einstein_radius_axvline)

mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=tracer.galaxies[0].mass,
    grid=grid,
    mat_plot_1d=mat_plot,
    visuals_1d=visuals,
)
mass_profile_plotter.figures_1d(convergence=True)


"""
Finish.
"""

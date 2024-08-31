"""
Plots: CausticsLine
=========================

This example illustrates how to customize the tangential and radial critical curves plotted over data.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
__Caustics x1__

To plot a critical curve, we use a `Tracer` object which performs the strong lensing calculation to
produce a critical curve. 

By default, caustics are only plotted on source-plane images.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.2, 0.2)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
We also need the `Grid2D` that we can use to make plots of the `Tracer`'s quantities.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
The `Tracer` includes a method to compute its tangential and radial critical curves, meaning we can plot 
them via an `Include2D` object.
"""
include = aplt.Include2D(tangential_caustics=True, radial_caustics=False)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, include_2d=include)
tracer_plotter.figures_2d(source_plane=True)


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
    tracer=tracer, grid=grid, include_2d=include, mat_plot_2d=mat_plot
)
tracer_plotter.figures_2d(source_plane=True)

"""
__Visuals__

To plot caustics manually, we can pass them into a` Visuals2D` object. 

This is useful for plotting caustics on figures where they are not an internal property, like an `Array2D` of an 
image-plane image.
"""
tangential_caustic_list = tracer.tangential_caustic_list_from(grid=grid)
radial_caustics_list = tracer.radial_caustic_list_from(grid=grid)

visuals = aplt.Visuals2D(
    tangential_caustics=tangential_caustic_list,
    radial_caustics=radial_caustics_list,
)
image = tracer.image_2d_from(grid=grid)

array_plotter = aplt.Array2DPlotter(
    array=image, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
__Caustics x2__

If a `Tracer` has multiple mass profiles, it may also have multiple tangential and radial critical curves, which
are plotted in different colors by default.

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

lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(1.0, 0.0), einstein_radius=0.8, ell_comps=(0.2, 0.2)),
)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(-1.0, 0.0), einstein_radius=0.8, ell_comps=(0.2, 0.2)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCoreSph(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, include_2d=include, mat_plot_2d=mat_plot
)
tracer_plotter.figures_2d(source_plane=True)

"""
Finish.
"""

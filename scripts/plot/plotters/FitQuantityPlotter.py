"""
Plots: FitQuantityPlotter
========================

This example illustrates how to plot a `FitQuantity` object using a `FitQuantityPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
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
__Grid__

Define the 2D grid the quantity (in this example, the convergence) is evaluated using.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

"""
__Tracer__

Create a tracer which we will use to create our `DatasetQuantity`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy])

"""
__Dataset__

Use this `Tracer`'s 2D convergence to create the `DatasetQuantity`.

We assume a noise-map where all values are arbritrarily 0.01.
"""
convergence = tracer.convergence_2d_from(grid=grid)

dataset = al.DatasetQuantity(
    data=convergence,
    noise_map=al.Array2D.full(
        fill_value=0.01,
        shape_native=convergence.shape_native,
        pixel_scales=convergence.pixel_scales,
    ),
)

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the convergence we fit, which we define and apply to the 
`DatasetQuantity` object.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Fit__

We now fit the `DatasetQuantity` with a `Tracer`'s to create a `FitQuantity` object.

This `Tracer` has a slightly different lens galaxy and therefore convergence map, creating residuals in the plot.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

tracer_fit = al.Tracer(galaxies=[lens_galaxy])

fit = al.FitQuantity(dataset=dataset, tracer=tracer_fit, func_str="convergence_2d_from")

"""
__Figures__

We now pass the FitQuantity to an `FitQuantityPlotter` and call various `figure_*` methods to plot different attributes.
"""
fit_dataset_plotter = aplt.FitQuantityPlotter(fit=fit)
fit_dataset_plotter.figures_2d(
    image=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
__Subplots__

The `FitQuantityPlotter` may also plot a subplot of these attributes.
"""
fit_dataset_plotter.subplot_fit()

"""`
__Include__

`FitQuantity` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
    tangential_caustics=True,
    radial_caustics=True,
)

fit_plotter = aplt.FitQuantityPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()


"""
__Dataset Deflection Angles__

Use now repeat plots using a `Tracer`'s 2D deflection angles to create the `DatasetQuantity`.

Deflection angles are a `VectorYX2D` data structure, changing the behaviour of visualization.
"""
deflections_yx_2d = tracer.deflections_yx_2d_from(grid=grid)

dataset = al.DatasetQuantity(
    data=deflections_yx_2d,
    noise_map=al.VectorYX2D.full(
        fill_value=0.01,
        shape_native=deflections_yx_2d.shape_native,
        pixel_scales=deflections_yx_2d.pixel_scales,
    ),
)

"""
__Mask__

The again apply  `Mask2D` defining the regions of the deflections we fit, which despite the change in structure of the
data to vectors uses the same API.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Fit__

We now fit the `DatasetQuantity` with a `Tracer`'s to create a `FitQuantity` object.

This `Tracer` has a slightly different lens galaxy and therefore deflections_yx map, creating residuals in the plot.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

tracer_fit = al.Tracer(galaxies=[lens_galaxy])

fit = al.FitQuantity(
    dataset=dataset, tracer=tracer_fit, func_str="deflections_yx_2d_from"
)

"""
__Figures__

We now pass the FitQuantity to an `FitQuantityPlotter` and call various `figure_*` methods to plot different attributes.

Separate plots for the y and x components of the deflection angles are plotted.
"""
fit_dataset_plotter = aplt.FitQuantityPlotter(fit=fit)
fit_dataset_plotter.figures_2d(
    image=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
__Subplots__

The `FitQuantityPlotter` may also plot separate subplots for the y and x components of the deflection angles.
"""
fit_dataset_plotter.subplot_fit()

"""`
__Include__

The `Include2D` object can again be used to customize these plots.
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
    tangential_caustics=True,
    radial_caustics=True,
)

fit_plotter = aplt.FitQuantityPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()

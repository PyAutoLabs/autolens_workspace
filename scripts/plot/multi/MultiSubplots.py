"""
Plots: MultiSubPlots
====================

This example illustrates how to plot different figures from different plotters on the same subplot, using the example
of combining an `ImagingPlotter` and `FitImagingPlotter`.

An example of when to use this plotter would be when a range of different figures are plotted on the same subplot,
for example an image and its signal-to-nosie map, and the normalized residual-map of a fit to the image. This combined
an `ImagingPLotter` and `FitImagingPlotter` and is the example used in this example script.

The script `MultiFigurePlotter.py` illustrates a similar example, but plots the same figures from a single `Plotter`
object on the same subplot. This script offers a more concise way of plotting the same figures on the same subplot, but
does not have the flexibility of plotting different figures from different `Plotter` objects shown here.

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
__Dataset__

Load and plot the `lens_sersic` dataset, which we visualize in this example script.
"""
dataset_name = "lens_sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask + Fit__

We now mask the data and fit it with a `Tracer` to create a `FitImaging` object.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
We now pass the imaging to an `ImagingPlotter` and the fit to an `FitImagingPlotter`.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
fit_plotter = aplt.FitImagingPlotter(fit=fit)

"""
We next pair the `MatPlot2D` objects of the two plotters, which ensures the figures plot on the same subplot.
"""
dataset_plotter.mat_plot_2d = fit_plotter.mat_plot_2d

"""
We next open the subplot figure, specifying: 

 - How many subplot figures will be on our image.
 - The shape of the subplot.
 - The figure size of the subplot. 
"""
dataset_plotter.open_subplot_figure(
    number_subplots=5, subplot_shape=(1, 5), subplot_figsize=(18, 3)
)

"""
We now call the `figures_2d` method of all the plots we want to be included on our subplot. These figures will appear
sequencially in the subplot in the order we call them.
"""
dataset_plotter.figures_2d(data=True, signal_to_noise_map=True)
fit_plotter.figures_2d(
    model_image=True, normalized_residual_map=True, chi_squared_map=True
)

"""
This outputs the figure, which in this example goes to your display as we did not specify a file format.
"""
dataset_plotter.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot")

"""
Close the subplot figure, in case we were to make another subplot.
"""
dataset_plotter.close_subplot_figure()

"""
Start Here: Imaging
===================

Strong gravitational lenses are often observed with CCD imaging, for example using HST, JWST,
or ground-based telescopes.

This script shows you how to model such a lens system using **PyAutoLens** with as little setup
as possible. In about 15 minutes you’ll be able to point the code at your own FITS files and
fit your first lens.

We focus on a *galaxy-scale* lens (a single lens galaxy). If you have multiple lens galaxies,
see the `start_here_group.ipynb` and `start_here_cluster.ipynb` examples.

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (a few minutes instead of an hour). If you don’t have
a GPU locally, consider Google Colab to get started quickly.

Finally, we also show how to simulate strong lens imaging. This is useful for practice, for
building training datasets, or for investigating lensing effects in a controlled way.
"""
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

We begin by loading the dataset. Three ingredients are needed for lens modeling:

1. The image itself (CCD counts).
2. A noise-map (per-pixel RMS noise).
3. The PSF (Point Spread Function).

Here we use HST imaging of a SLACS strong lens. Replace these FITS paths with your own to
immediately try modeling your data.

The `pixel_scales` value converts pixel units into arcseconds. It is critical you set this
correctly for your data.
"""
dataset_name = "slacs1430+4105"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Masking__

Lens modeling does not need to fit the entire image, only the region containing lens and
source light. We therefore define a circular mask around the lens.

- Make sure the mask fully encloses the lensed arcs and the lens galaxy.
- Avoid masking too much empty sky, as this slows fitting without adding information.

We’ll also oversample the central pixels, which improves modeling accuracy without adding
unnecessary cost far from the lens.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

# Over sampling is important for accurate lens modeling, but details are omitted
# for simplicity here, so don't worry about what this code is doing yet!

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

To perform lens modeling we must define a lens model, describing the light profiles of the lens and source galaxies,
and the mass profile of the lens galaxy.

A brilliant lens model to start with is one which uses a Multi Gaussian Expansion (MGE) to model the lens and source
light, and a Singular Isothermal Ellipsoid (SIE) plus shear to model the lens mass. 

Full details of which this models is so good are provided in the main workspace docs, but in a nutshell it 
provides a brilliant balance of being fast to fit, flexible enough to capture complex galaxy morphologies and 
providing accurate fits to the vast majority of strong lenses.

The MGE model composiiton API is quite long and technical, so we simply load the model below via a utility function
and print `model.info` to show the parameters that the model is composed of.
"""
model = al.model_util.mge_start_here_lens_model_from(mask_radius=mask_radius)

print(model.info)

"""
__Model Fit__

We now fit the data with the lens model using the non-linear fitting method and nested sampling algorithm Nautilus.

This requires an `AnalysisImaging` object, which defines the `log_likelihood_function` used by Nautilus to fit
the model to the imaigng data.
"""
search = af.Nautilus(
    path_prefix=Path("imaging"), # The path where results and output
    name="start_here", # The name of the fit and folder results are output to.
    unique_tag=dataset_name, # A unique tag which also defines the folder.
    n_live=100, # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50, # For fast GPU fitting lens model fits are batched and run simultaneously.
    iterations_per_update=20000, # Every N iterations the results are written to hard-disk for inspection.
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Result_

Now this is running you should checkout the `autolens_workspace/output` folder, where many results of the fit
are written in a human readable format (e.g. .json files) and .fits and .png images of the fit are stored.

When the fit is complex, we can print the results by printing `result.info`.
"""
print(result.info)

"""
The result also contains the maximum likelihood lens model which can be used to plot the best-fit lensing information
and fit to the data.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
The result object contains pretty much everything you need to do science with your own strong lens, but details
of all the information it contains are beyond the scope of this introductory script. The `guides` and `result` 
packages of the workspace contains all the information you need to analyze your results yourself.

__Extra Galaxy Removal__

__Model Your Own Lens__

If you have your own strong lens imaging data, you are now ready to model it yourself by adapting the code above
and simply inputting the path to your own .fits files into the `Imaging.from_fits()` function.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- Supply your own CCD image, PSF, and RMS noise-map.
- Ensure the lens galaxy is roughly centered in the image.
- Double-check `pixel_scales` for your telescope/detector.
- Adjust the mask radius to include all relevant light.
- Start with the default model — it works very well for pretty much all galaxy scale lenses!

__Simulator__

Let’s now switch gears and simulate our own strong lens imaging. This is a great way to:

- Practice lens modeling before using real data.
- Build large training sets (e.g. for machine learning).
- Test lensing theory in a controlled environment.

We’ll start by simulating “perfect” images without telescope effects. This means no blurring
from a PSF and no noise — just the raw light from galaxies and deflections from gravity.

To do this we need to define a 2D grid of (y,x) coordinates in the image-plane. This grid is
where we’ll evaluate the light from the lens and source galaxies.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
We now define a `Tracer` — this is the key object that combines all galaxies in the system
and computes how light rays are deflected.

- The lens galaxy has both light (a Sersic bulge) and mass (an isothermal profile + shear).
- The source galaxy has its own light (a SersicCore profile).

Together they define a strong lens system. The tracer will “ray-trace” our grid through
this mass distribution and generate a lensed image.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
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
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Plotting the tracer’s image gives us a “perfect” view of the strong lens system, before
adding telescope effects.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
The tracer’s outputs can also be saved as arrays — for example:

- The image itself.
- The convergence (κ).
- The potential (ψ).

This allows you to build datasets for training or for direct comparison with models.
"""
image = tracer.image_2d_from(grid=grid)

al.output_to_fits(
    values=image.native,
    file_path=Path("image.fits"),
    overwrite=True,
)

"""
__Simulator__

The images above do not represent real CCD imaging data, as they do not include the blurring due to the telescope 
optics or sources of noise.

The `SimulatorImaging` class simulates these two key properties of real imaging data, which we use below to create
realistic imaging of the strong lens system.

The units of the image are arbitrary, with the workspace providing guides on how to convert to physical units for lens
simulations.

The code below performs the simulation, plots the simulated imaging data and outputs it to .fits files with .png
files included for easy visualization.
"""
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), # The 2D shape of the PSF array.
    sigma=0.1, # The size of the Gaussian PSF, where FWHM = 2.35 * sigma.
    pixel_scales=grid.pixel_scales # The pixel scale of the PSF, matches the image's pixel scale.
)

simulator = al.SimulatorImaging(
    exposure_time=300.0, # The exposure time of the observation, increases the S/N of the image.
    psf=psf, # The PSF which blurs the image.
    background_sky_level=0.1, # The background sky level of the image, increases the noise.
    add_poisson_noise_to_data=True, # Whether Poisson noise is added to the image or not.
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

dataset.output_to_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

"""
We can now inspect the simulated dataset: image, noise-map, and PSF. These can also be
written to FITS files and visualized as PNGs. This is exactly the same format as real data,
so you can immediately try fitting the simulated dataset with the modeling workflow above.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

"""
__Sample__

Often we want to simulate *many* strong lenses — for example, to train a neural network
or to explore population-level statistics.

To do this we use the model API to draw galaxies randomly from realistic distributions.
Below we generate just 3 lenses for speed, but you can easily scale this to hundreds or
thousands.
"""
lens_model, source_model = al.model_util.simulator_start_here_model_from()

print(lens_model.info)
print(source_model.info)

"""
We now simulate a sample of strong lens, we just do 3 for efficiency here but you can increase this to any number.
"""
total_datasets = 3

for sample_index in range(total_datasets):

    dataset_sample_path = Path(dataset_path, f"dataset_{sample_index}")

    lens_galaxy = lens_model.random_instance()
    source_galaxy = source_model.random_instance()

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    dataset.output_to_fits(
        data_path=Path(dataset_sample_path, "data.fits"),
        psf_path=Path(dataset_sample_path, "psf.fits"),
        noise_map_path=Path(dataset_sample_path, "noise_map.fits"),
        overwrite=True,
    )


    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_sample_path, format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
    tracer_plotter.subplot_tracer()
    tracer_plotter.subplot_galaxies_images()

"""
__Wrap Up__

This script has shown how to model CCD imaging data of strong lenses, and simulate your own strong lens images.

Details of the **PyAutoLens** API and how lens modeling and simulations actually work were omitted for simplicity,
but everything you need to know is described throughout the main workspace documentation. You should check it out,
but maybe you want to try and model your own lens first!

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/*/modeling/imaging`: A full description of the lens modeling API and how to customize your model-fits.
- `autolens_workspace/*/simulators/imaging`: A full description of the lens simulation API and how to customize your simulations.
- `autolens_workspace/*/data_preparation/imaging`: How to load and prepare your own imaging data for lens modeling.
- `autolens_workspace/results`: How to load and analyze the results of your lens model fits, including tools for large samples.
- `autolens_workspace/guides`: A complete description of the API and information on lensing calculations and units.
- `autolens_workspace/feature`: A description of advanced features for lens modeling, for example pixellated source reconstructions, read this once you're confident with the basics!
"""
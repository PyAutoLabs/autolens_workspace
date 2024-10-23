"""
Simulator: Extra Galaxies
=========================

Certain lenses have small galaxies within their Einstein radius, or nearby the lensed source emission. 

The emission of these galaxies may overlap the lensed source emission, and their mass may contribute to the lensing 
of the source.

We therefore will need to mask the emission of these extra galaxies or include them in the model as light profiles which
fit and subtract the emission. We may also include these galaxies as mass profiles in the lens model, accounting for
their lensing effects via ray-tracing.

This uses the modeling API, which is illustrated in 
the script `autolens_workspace/*/modeling/imaging/features/extra_galaxies.py`.

This script simulates an imaging dataset which includes extra galaxies near the lens and source
galaxies. This is used to illustrate the extra galaxies API in the script above.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source galaxy's light is an `Sersic`.
 - There are two extra galaxies whose light is near the strong lens and their mass perturbs the lensed source's emission.

__Other Scripts__

This dataset is used in the following scripts:

 `autolens_workspace/*/data_preparation/imaging/examples/optional/scaled_dataset.ipynb`

To illustrate how to subtract and remove the light of extra galaxies in real strong lensing data, so that it does
not impact the lens model.

 `autolens_workspace/*/data_preparation/imaging/examples/optional/extra_galaxies_centres.ipynb`

To illustrate how mark extra galaxy centres on a dataset so they can be used in the lens model.

 `autolens_workspace/*/modeling/imaging/features/extra_galaxies.ipynb`

To illustrate how compose and fit a lens model which includes the extra galaxies as light and mass profiles.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_name = "extra_galaxies"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using a `Grid2D` with the `OverSamplingIterate` object.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.1,
    over_sampling=al.OverSamplingIterate(
        fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    ),
)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Galaxies__

Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.
"""
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
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


"""
__Extra Galaxies__

Includes two extra galaxies, which must be modeled or masked to ensure they do not impact the fit.

Note that their redshift is the same as the main galaxy, which is not necessarily the case in real observations. 

If they are at a different redshift, the tools for masking or modeling the luminous emission of the extra galaxies 
are equipped to handle this.

For mass modeling, their redshifts being different to the main galaxy will lead to multi-plane ray-tracing being
performed.
"""
extra_galaxy_0_centre = (1.0, 3.5)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    light=al.lp.ExponentialSph(
        centre=extra_galaxy_0_centre, intensity=2.0, effective_radius=0.5
    ),
    mass=al.mp.IsothermalSph(centre=extra_galaxy_0_centre, einstein_radius=0.1),
)

extra_galaxy_1_centre = (-2.0, -3.5)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    light=al.lp.ExponentialSph(
        centre=extra_galaxy_1_centre, intensity=2.0, effective_radius=0.8
    ),
    mass=al.mp.IsothermalSph(centre=extra_galaxy_1_centre, einstein_radius=0.2),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.

For a faster run time, the tracer visualization uses the binned grid instead of the iterative grid.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=path.join(dataset_path, "tracer.json"),
)

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/extra_galaxies`.
"""

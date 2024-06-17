"""
Simulator: Wavelength Dependent
===============================

This script simulates multiple `Imaging` datasets of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`, which has a different `intensity` at each wavelength.

Unlike other `multi` simulators, all datasets are at the same wavelength and therefore the source does not change
its appearance in each dataset.

This dataset demonstrates how PyAutoLens's multi-dataset modeling tools can also simultaneously analyse datasets
observed at the same wavelength.

An example use case might be analysing undithered HST images before they are combined via the multidrizzing process,
to remove correlated noise in the data.

TODO: NEED TO INCLUDE DIFFERENT POINTING / CENTERINGS.
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
__Dataset Paths__
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "same_wavelength"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

If observed at the same wavelength, it is likely the datasets have the same pixel-scale.

Nevertheless, we specify this as a list as there could be an exception.
"""
pixel_scales_list = [0.1, 0.1]

grid_list = [
    al.Grid2D.uniform(
        shape_native=(150, 150),
        pixel_scales=pixel_scales,
        over_sampling=al.OverSamplingIterate(
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16],
        ),
    )
    for pixel_scales in pixel_scales_list
]

"""
Simulate simple Gaussian PSFs for the images, which we assume slightly vary (e.g. due to different bserving conditions
for each image)
"""
sigma_list = [0.09, 0.11]

psf_list = [
    al.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=sigma, pixel_scales=grid.pixel_scales
    )
    for grid, sigma in zip(grid_list, sigma_list)
]

"""
Create separate simulators for the images, which we will assume have slightly different exposure times and background
sky levels.
"""
exposure_time_list = [300.0, 350.0]
background_sky_level_list = [0.1, 0.12]

simulator_list = [
    al.SimulatorImaging(
        exposure_time=exposure_time,
        psf=psf,
        background_sky_level=background_sky_level,
        add_poisson_noise=True,
    )
    for psf, exposure_time, background_sky_level in zip(
        psf_list, exposure_time_list, background_sky_level_list
    )
]


"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE+Shear) for this simulated lens.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Ray Tracing__

The source galaxy is observed att he same wavelength in each image thus its intensity does not vary across the datasets.
"""
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

"""
Use these galaxies to setup tracers at each waveband, which will generate each image for the simulated `Imaging` 
dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
for grid in grid_list:
    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset_list = [
    simulator.via_tracer_from(tracer=tracer, grid=grid)
    for grid, simulator in zip(grid_list, simulator_list)
]

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Output__

Output each simulated dataset to the dataset path as .fits files, with a tag describing its color.
"""
for i, dataset in enumerate(dataset_list):
    dataset.output_to_fits(
        data_path=path.join(dataset_path, f"image_{i}.fits"),
        psf_path=path.join(dataset_path, f"psf_{i}.fits"),
        noise_map_path=path.join(dataset_path, f"noise_map_{i}.fits"),
        overwrite=True,
    )

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.

For a faster run time, the tracer visualization uses the binned grid instead of the iterative grid.
"""
for i, dataset in enumerate(dataset_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, suffix=f"_{i}", format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)

for i, grid in enumerate(grid_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, suffix=f"_{i}", format="png")
    )

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
The dataset can be viewed in the folder `autolens_workspace/imaging/multi/same_wavelength/simple__no_lens_light`.
"""

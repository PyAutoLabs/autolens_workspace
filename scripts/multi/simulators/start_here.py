"""
Simulator: SIE
==============

This script simulates multi-wavelength `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's light profile is an `Sersic`, which has a different `intensity` at each wavelength.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`, which has a different `intensity` at each wavelength.

Two images are simulated, corresponding to a greener ('g' band) redder image (`r` band).

This is an advanced script and assumes previous knowledge of the core **PyAutoLens** API for simulating images. Thus,
certain parts of code are not documented to ensure the script is concise.
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
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for naming the datasets on output.
"""
color_list = ["g", "r"]

"""
__Dataset Paths__
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "lens_sersic"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

The pixel-scale of each color image is different meaning we make a list of grids for the simulation.
"""
pixel_scales_list = [0.08, 0.12]

grid_list = [
    al.Grid2D.uniform(
        shape_native=(150, 150),
        pixel_scales=pixel_scales,
        over_sampling=al.OverSamplingIterate(
            fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
        ),
    )
    for pixel_scales in pixel_scales_list
]

"""
Simulate simple Gaussian PSFs for the images in the r and g bands.
"""
sigma_list = [0.1, 0.2]

psf_list = [
    al.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=sigma, pixel_scales=grid.pixel_scales
    )
    for grid, sigma in zip(grid_list, sigma_list)
]

"""
Create separate simulators for the g and r bands.
"""
background_sky_level_list = [0.1, 0.15]

simulator_list = [
    al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=background_sky_level,
        add_poisson_noise=True,
    )
    for psf, background_sky_level in zip(psf_list, background_sky_level_list)
]

"""
__Ray Tracing__

The lens galaxy light at each wavelength has a different intensity, thus we create two lens galaxies for each waveband. 

The lens galaxy's mass (SIE+Shear) is identical for each waveband and included in both lens galaxies in the list..
"""
intensity_list = [0.05, 1.5]

bulge_list = [
    al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=intensity,
        effective_radius=0.8,
        sersic_index=4.0,
    )
    for intensity in intensity_list
]

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

lens_galaxy_list = [
    al.Galaxy(
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )
    for bulge in bulge_list
]

"""
__Ray Tracing__

The source galaxy at each wavelength has a different intensity, thus we create two source galaxies for each waveband.
"""
intensity_list = [0.5, 0.7]

source_galaxy_list = [
    al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=intensity,
            effective_radius=0.1,
            sersic_index=1.0,
        ),
    )
    for intensity in intensity_list
]

"""
Use these galaxies to setup tracers at each waveband, which will generate each image for the simulated `Imaging` 
dataset.
"""
tracer_list = [
    al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    for lens_galaxy, source_galaxy in zip(lens_galaxy_list, source_galaxy_list)
]

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
for tracer, grid in zip(tracer_list, grid_list):
    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset_list = [
    simulator.via_tracer_from(tracer=tracer, grid=grid)
    for grid, simulator, tracer in zip(grid_list, simulator_list, tracer_list)
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
for color, dataset in zip(color_list, dataset_list):
    dataset.output_to_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        overwrite=True,
    )

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.

For a faster run time, the tracer visualization uses the binned grid instead of the iterative grid.
"""
for color, dataset in zip(color_list, dataset_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{color}_", format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)


for color, grid, tracer in zip(color_list, grid_list, tracer_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{color}_", format="png")
    )

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
    tracer_plotter.subplot_tracer()
    tracer_plotter.subplot_galaxies_images()

    mat_plot = aplt.MatPlot2D(
        title=aplt.Title(label=f"Lens and Source {color}-band Images"),
        output=aplt.Output(
            path=dataset_path, filename=f"{color}_tracer_image", format="png"
        ),
    )

    tracer_plotter.figures_2d(image=True)

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
[
    al.output_to_json(
        obj=tracer, file_path=path.join(dataset_path, f"{color}_tracer.json")
    )
    for color, tracer in zip(color_list, tracer_list)
]

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/multi/lens_sersic`.
"""

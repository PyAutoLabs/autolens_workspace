"""
Simulator: Source Complex
=========================

This script simulates `Imaging` of a 'galaxy-scale' strong lens where the source galaxy's light is more complex than
other examples, being composed of 4 Sersics.

It is used to illustrate features which reconstruct the source galaxy's light using a pixelizaiton, for example
in the script `autolens_workspace/notebooks/modeling/imaging/features/pixelization.ipynb`.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is four `Sersic`s.

This produces a very complex lensed source galaxy, which is used to illustrate source reconstructions on pixel-grids
using an `Inversion`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
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

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_name = "source_complex"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using a `Grid2D` with the `OverSamplingIterate` object.
"""
grid = al.Grid2D.uniform(
    shape_native=(170, 170),
    pixel_scales=0.05,
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
__Ray Tracing__

Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.

the `source_galaxy` has four separate light components each of which are `Sersic` profiles.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light_0=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.1,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
    light_1=al.lp.SersicCore(
        centre=(0.8, 0.6),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.5, angle=30.0),
        intensity=0.2,
        effective_radius=0.3,
        sersic_index=3.0,
    ),
    light_2=al.lp.SersicCore(
        centre=(-0.3, 0.6),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.3, angle=120.0),
        intensity=0.6,
        effective_radius=0.5,
        sersic_index=1.5,
    ),
    light_3=al.lp.SersicCore(
        centre=(-0.3, -0.3),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=85.0),
        intensity=0.4,
        effective_radius=0.1,
        sersic_index=2.0,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

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
The dataset can be viewed in the folder `autolens_workspace/imaging/source_complex`.
"""

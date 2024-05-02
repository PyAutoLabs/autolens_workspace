"""
Simulator: Manual Signal to Noise Ratio
=======================================

When simulating `Imaging` of a strong lens, one is often not concerned with the actual units of the light (e.g.
electrons per second, counts, etc.) but instead simple wants the data to correspond to a certain signal to noise
value.

This can be difficult to achieve when specifying the `intensity` of the input light profiles, especially given the
unknown contribution of the mass model's magnification.

This script illustrates the `lp_snr` light profiles, which when used to simulate a dataset via a tracer, set the
signal to noise of each light profile to an input value. This uses the `exposure_time` and `background_sky_level`
of the `SimulatorImaging` object to choose the `intensity` of each light profile such that the input signal to
noise is used.

For normal light profiles, the `intensity` is defined in units of electrons per second, meaning that the
`exposure_time` and `background_sky_level` are used to convert this to counts when adding noise. When the `lp_snr`
profiles are used, the `exposure_time` and `background_sky_level` are instead used to set its S/N, meaning their input
values do not set the S/N.

However, the ratio of `exposure_time` and `background_sky_level` does set how much noise is due to Poisson count
statistics in the CCD imaging detector relative to the background sky. If one doubles the `exposure_time`, the
Poisson count component will contribute more compared to the background sky component. For detailed scientific
analysis, one should therefore make sure their values are chosen to produce images with realistic noise properties.

The use of the `light_snr` profiles changes the meaning of `exposure_time` and `background_sky_level`.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's bulge is an `Sersic` with a S/N of 50.0.
 - The lens galaxy's disk is an `Exponential` with a S/N of 20.0.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is two `Sersic`;s with S/N of 20.0 and 10.0.

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

The path where the dataset will be output.
"""
dataset_type = "imaging"
dataset_label = "misc"
dataset_name = "manual_signal_to_noise_ratio"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

Simulate the image using a `Grid2D` with the `OverSamplingIterate` object.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
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
__Ray Tracing__

Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.

the `lens_galaxy` uses light profile signal-to-noise objects (`lp_snr`).
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_snr.Sersic(
        signal_to_noise_ratio=50.0,
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp_snr.Exponential(
        signal_to_noise_ratio=20.0,
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        effective_radius=1.6,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_snr.Sersic(
        signal_to_noise_ratio=20.0,
        centre=(0.25, 0.15),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        effective_radius=0.7,
        sersic_index=1.0,
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_snr.Sersic(
        signal_to_noise_ratio=10.0,
        centre=(0.7, -0.5),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=60.0),
        effective_radius=1.6,
        sersic_index=3.0,
    ),
)


"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

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
The dataset can be viewed in the folder `autolens_workspace/imaging/misc/manual_signal_to_noise_ratio`.
"""

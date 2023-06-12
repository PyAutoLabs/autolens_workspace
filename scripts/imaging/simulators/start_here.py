"""
Simulator: Start Here
=====================

This script is the starting point for simulating galaxy-galaxy strong lenses as CCD imaging data (E.g. Hubble Space
Telescope, Euclid) and it provides an overview of the lens simulation API.

After reading this script, the `examples` folder provide examples for simulating more complex lenses in different ways.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's light profiles are an `Sersic` and `Exponential`.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.

__Plotters__

To output images of the simulated data, `Plotter` objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of strong lenses.

The `PLotter` API is described in the `autolens_workspace/*/plot/start_here.py` script.
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
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "imaging"
dataset_name = "simple"

"""
The path where the dataset will be output, which in this case is:

`/autolens_workspace/dataset/imaging/simple`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Simulate__

When simulating the amount of emission in each image pixel from the lens and source galaxies, a two dimensional
line integral of all of the emission within the area of that pixel should be performed. However, for complex lens
models this can be difficult to analytically compute and can lead to slow run times.

Instead, an iterative ray-tracing algorithm is used to approximate the line integral. Grids of increasing resolution
are used to evaluate the flux in each pixel from the lens and source galaxies. Grids of higher resolution are used
until the fractional accuracy of the flux in each pixel meets a certain threshold, which we set below to 99.99%

This uses the `Grid2DIterate` object, which is identical to the `Grid2D` object you may have seen in other example 
scripts, however it additional performs the iterative ray-tracing described above.

The grid is also created from:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.
"""
grid = al.Grid2DIterate.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 8, 16, 24],
)

"""
All CCD imaging data (e.g. Hubble Space Telescope, Euclid) are blurred by the telescope optics when they are imaged.

The Point Spread Function (PSF) describes the blurring of the image by the telescope optics, in the form of a
two dimensional convolution kernel. The lens modeling scripts use this PSF when fitting the data, to account for
this blurring of the image.

In this example, use a simple 2D Gaussian PSF, which is convolved with the image of the lens and source galaxies 
when simulating the dataset.
"""
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
To simulate the `Imaging` dataset we first create a simulator, which includes:

 - The exposure time of the simulated dataset, increasing this will increase the signal-to-noise of the simulated data.
 - The PSF of the simulated dataset, which is convolved with the image of the lens and source galaxies.
 - The background sky level of the simulated dataset, which is added to the image of the lens and source galaxies and
  leads to a higher level of Poisson noise.
 - Whether the simulated dataset includes Poisson noise.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

"""
__Ray Tracing__

We now define the lens galaxy's light (elliptical Sersic + Exponential), mass (SIE+Shear) and source galaxy light
(elliptical Sersic) for this simulated lens.

The following should be noted about the parameters below:

 - The native units of light and mass profiles distance parameters (e.g. centres, effective_radius) are arc-seconds. 
 - The intensity of the light profiles is in units of electrons per second per arc-second squared.
 - The ellipticity of light and mass profiles are defined using the `ell_comps` parameter, however we below use
   the convert module to input the `axis-ratio` (semi-major axis / semi-minor axis = b/a) and positive 
   angle (degrees defined counter clockwise from the positive x-axis).
 - The external shear is defined using the (gamma_1, gamma_2) convention.
 - The input redshifts are used to determine which galaxy is the lens (e.g. lower redshift) and which is the 
   source (e.g. higher redshift).
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
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
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)


"""
We now pass these galaxies to a `Tracer`, which performs the ray-tracing calculations they describe and returns
the image of the strong lens system they produce.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
We can plot the `Tracer``s image, which is the image we'll next simulate as CCD imaging data.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
By passing the `Tracer` and grid to the simulator, we create the simulated CCD imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
We now plot the simulated `Imaging` dataset before outputting it to fits.

Note how unlike the `Tracer` image above, the simulated `Imaging` dataset includes the blurring effects of the 
telescope's PSF and also has noise.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.

If you are unfamiliar with .fits files, this is the standard file format of astronomical data and you can open 
them using the software ds9 (https://sites.google.com/cfa.harvard.edu/saoimageds9/home).
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
__Visualize__

In the same folder as the .fits files, we also output subplots of the simulated dataset in .png format, as well as 
other images which summarize the dataset.

Having .png files like this is useful, as they can be opened quickly and easily by the user to check the dataset.

For a faster run time, this visualization uses a regular grid which does not perferm the iterative ray-tracing.
"""
grid = al.Grid2D.uniform(
    shape_native=grid.shape_native,
    pixel_scales=grid.pixel_scales,
)

mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.figures_2d(data=True)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_plane_images()

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `Tracer.from_json`.
"""
tracer.output_to_json(file_path=path.join(dataset_path, "tracer.json"))

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/simple`.
"""

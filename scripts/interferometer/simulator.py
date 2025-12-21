"""
Simulator: SIE
==============

This script simulates `Interferometer` data of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

""" 
The `dataset_type` describes the type of data being simulated (in this case, `Interferometer` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "interferometer"
dataset_name = "simple"

"""
The path where the dataset will be output.

In this example, this is: `/autolens_workspace/dataset/interferometer/simple`
"""
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2d grid of (y,x) coordinates that the galaxy images are evaluated and therefore simulated on, via
the inputs:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.

For interferometer data, this image is evaluate in real-space and then transformed to Fourier space.

__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called
over sampling is used, which evaluates light profiles on a higher resolution grid than the image data to ensure the
calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.
"""
grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

"""
To perform the Fourier transform we need the wavelengths of the baselines, which we'll load from the fits file below.

By default we use baselines from the Square Mile Array (SMA), which produces low resolution interferometer data that
can be fitted extremely efficiently. The `autolens_workspace` includes ALMA uv_wavelengths files for simulating
much high resolution datasets (which can be performed by replacing "sma.fits" below with "alma.fits").
"""
uv_wavelengths_path = Path("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=Path(uv_wavelengths_path, "sma.fits"), hdu=0
)

"""
To simulate the interferometer dataset we first create a simulator, which defines the exposure time, noise levels 
and Fourier transform method used in the simulation.
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerDFT,
)

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.

The following should be noted about the parameters below:

 - The native units of light and mass profiles distance parameters (e.g. centres, effective_radius) are arc-seconds. 
 - The intensity of the light profiles is in units of electrons per second per arc-second squared.
 - The ellipticity of light and mass profiles are defined using the `ell_comps` parameter, however we below use
   the convert module to input the `axis-ratio` (semi-major axis / semi-minor axis = b/a) and positive 
   angle (degrees defined counter clockwise from the positive x-axis).
 - The external shear is defined using the (gamma_1, gamma_2) convention.
 - The input redshifts are used to determine which galaxy is the lens (e.g. lower redshift) and which is the 
   source (e.g. higher redshift).
 - The source uses a cored Sersic with a radius half the pixel-scale, ensuring that over-sampling is not necessary.
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

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated interferometer dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
interferometer dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Lets plot the simulated interferometer dataset before we output it to fits.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(dirty_image=True)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
dataset.output_to_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""
mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()
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
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Multiple Images__

Lens modeling can use a "positions likelihood penalty", whereby mass models which traces the (y,x) 
coordinates of multiple images of a source galaxy to positions which are far apart from one another 
in the source plane are penalized in the lens model's overall likelihood.

This speeds up lens modeling, helps the non-linear search avoid local maxima and is vital for inferred 
accurate solutions when using pixelized source reconstructions.

For real data, the multiple image positions are determined by eye from the data, for example
using a Graphical User Interface (GUI) to mark them with mouse clicks. For simulated data, we can save
ourselves time by using the `PointSolver` to determine the multiple image positions automatically and
output to a .json file.

If you have not looked in the `point_source` package, the point solver is the core tool used to find
multiple image positions for point source lens modeling (e.g. lensed quasars).
"""
solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

al.output_to_json(
    file_path=dataset_path / "positions.json",
    obj=positions,
)

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/simple`.

__Many Visibilities__

Simulating interferometer datasets with many visibilities can be computationally expensive
if a direct Fourier transform is used. 

Therefore, for simulating high-resolution datasets with many visibilities (e.g. > 10000), we recommend using the
`TransformerNUFFT` transformer, which uses a non-uniform fast Fourier transform via the library pynufft to 
perform the Fourier transform efficiently.

Higher resolution datasets also require a higher resolution real space grid, which also increases the computational
costs of simulating the dataset.

The code below loads a `uv_wavelengths` file with many over 1 million visibilities and simulates the dataset using 
the `TransformerNUFFT`.

__High Resolution Dataset__

A high-resolution `uv_wavelengths` file for ALMA is available in a separate repository that hosts large files which
are too big to include in the main `autolens_workspace` repository:

https://github.com/Jammy2211/autolens_workspace_large_files

After downloading the file, place it in the directory:

`autolens_workspace/dataset/interferometer/alma`

You can then simulate and fit this high-resolution ALMA dataset by uncommenting the 
line `dataset_name = "alma"` below.

This dataset is particularly useful for testing performance, memory usage, and accuracy when modeling realistic
ALMA uv-coverage with a very large number of visibilities.
"""
dataset_type = "interferometer"
# dataset_name = "alma"

dataset_path = Path("dataset", dataset_type, dataset_name)

grid = al.Grid2D.uniform(shape_native=(800, 800), pixel_scales=0.01)

uv_wavelengths_path = Path("dataset", dataset_type, dataset_name)
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=Path(uv_wavelengths_path, "uv_wavelengths.fits"), hdu=0
)

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerNUFFT,
)

"""
The code below is identical to above, outputting images, data, tracer and multiple image 
positions to the dataset folder.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.figures_2d(dirty_image=True)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

dataset.output_to_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

mat_plot = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset, mat_plot_2d=mat_plot)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()
dataset_plotter.figures_2d(data=True)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
tracer_plotter.subplot_tracer()
tracer_plotter.subplot_galaxies_images()

al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

al.output_to_json(
    file_path=dataset_path / "positions.json",
    obj=positions,
)

"""
Finish.
"""

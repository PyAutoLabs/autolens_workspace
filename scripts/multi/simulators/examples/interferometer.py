"""
Simulator: Mutli Interferometer
===============================

This script simulates `Interferometer` data of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.

This dataset is paired with the script `multi/simulators/lens_sersic.py` and therefore
provides interferometer observations of the same strong lens.

It is used to demonstrate the combination of imaging and interferometer datasets.
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
The `dataset_type` describes the type of data being simulated (in this case, `Interferometer` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/psf.fits`.
"""
dataset_type = "multi"
dataset_label = "interferometer"
dataset_name = "simple__no_lens_light"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
The path where the dataset will be output, which in this case is
`/autolens_workspace/dataset/interferometer/simple__no_lens_light`
"""
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

For simulating interferometer data of a strong lens, we recommend using a Grid2D object with a `sub_size` of 1. This
simplifies the generation of the strong lens image in real space before it is transformed to Fourier space.
"""
grid = al.Grid2D.uniform(shape_native=(800, 800), pixel_scales=0.05, sub_size=1)

"""
To perform the Fourier transform we need the wavelengths of the baselines, which we'll load from the fits file below.

By default we use baselines from the Square Mile Array (SMA), which produces low resolution interferometer data that
can be fitted extremely efficiently. The `autolens_workspace` includes ALMA uv_wavelengths files for simulating
much high resolution datasets (which can be performed by replacing "sma.fits" below with "alma.fits").
"""
uv_wavelengths_path = path.join("dataset", "interferometer", "uv_wavelengths")
uv_wavelengths = al.util.array_1d.numpy_array_1d_via_fits_from(
    file_path=path.join(uv_wavelengths_path, "sma.fits"), hdu=0
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

The lens galaxy's mass model is identical to that of the script `multi/simulators/simple__no_lens_light.py`, because
we will use this dataset to demonstrate the combination of imaging and interferometer datasets.

the source galaxy morphology is very different to that script, because the sub-milimeter observations observe 
the galaxy's warm dust which is therefore clumpy and irregular.
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
    bulge=al.lp.Sersic(
        centre=(0.2, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=90.0),
        intensity=0.2,
        effective_radius=0.5,
        sersic_index=2.0,
    ),
    clump_0=al.lp.SersicSph(
        centre=(0.3, 0.4), intensity=0.1, effective_radius=0.3, sersic_index=2.5
    ),
    clump_1=al.lp.SersicSph(
        centre=(0.0, 0.05), intensity=0.15, effective_radius=0.2, sersic_index=3.0
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated interferometer dataset.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

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
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
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
dataset_plotter.figures_2d(data=True, dirty_image=True)

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
The dataset can be viewed in the folder `autolens_workspace/imaging/simple__no_lens_light`.
"""

"""
Simulator: Manual Signal to Noise Ratio
=======================================

Build controlled imaging simulations to test lens-light subtraction and recovery of faint lensed sources as data quality changes.
The `lp_snr` profiles let you request a signal-to-noise ratio instead of manually tuning each profile's intensity.
This is useful for comparing model recovery across a range of brightnesses at fixed observing conditions.

__What the Requested S/N Means__

The target is the expected signal-to-noise ratio at the brightest pixel of each component, evaluated on
the simulation grid with the supplied PSF. It is not an integrated S/N for the galaxy. Each component is
normalized separately, so overlapping components and the final random noise realization need not reproduce
those targets in the combined image.

The normalization solves S/N = S / sqrt(S + B), where S is the component's expected counts in its brightest
pixel and B is the sky counts there. `exposure_time` still has its usual meaning, and `background_sky_level`
is still a count rate: together they determine the intensity required to reach the requested S/N.
Changing either while keeping the target S/N fixed changes the simulated object's brightness as well.

Choose exposure, sky, PSF and sampling to represent the observations you want to study, then inspect the
resulting image and noise map. An S/N target alone does not specify all of the image's noise properties.

For background sources the profiles are evaluated on ray-traced grids, so their normalization responds to
the lensed image and PSF. This does not prescribe a total magnification or an integrated source S/N.

__Contents__

- **Model:** Define the lens and source components and their target S/N values.
- **Dataset Paths:** Choose where the simulated data are saved.
- **Simulate:** Set the grid, PSF, exposure time and sky level.
- **Ray Tracing:** Construct the lens and source galaxies and generate the lensed image.
- **Output:** Save the data, PSF and noise map to FITS files.
- **Visualize:** Inspect the simulated dataset and lens components.
- **Tracer json:** Save the simulation's true model for later comparison.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's bulge is an `Sersic` with a S/N of 50.0.
 - The lens galaxy's disk is an `Exponential` with a S/N of 20.0.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The external shear is an `ExternalShear` held in a `MassField`.
 - The source galaxy's light is two `Sersic` components with S/N of 20.0 and 10.0.

__Start Here Notebook__

If any code in this script is unclear, refer to the `imaging/simulator.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_label = "misc"
dataset_name = "manual_signal_to_noise_ratio"
dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

"""
__Simulate__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.
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
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
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
)

field = al.MassField(
    redshift=0.5,
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
tracer = al.Tracer(
    galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1], fields=[field]
)

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""

aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

aplt.subplot_tracer(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies_images(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)

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
The dataset can be viewed in the folder `autolens_workspace/dataset/imaging/misc/manual_signal_to_noise_ratio`.
"""

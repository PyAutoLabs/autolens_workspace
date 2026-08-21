"""
Simulator: Light Operated (Interferometer)
==========================================

It is common for galaxies to have point-source emission, for example bright emission right at their centre due
to an active galactic nuclei or a compact knot of star formation.

For interferometer data there is no Point Spread Function: the visibilities are the Fourier transform of the
sky emission, and the synthesized beam only enters when a dirty image is formed. The operated `Gaussian` in
this script therefore represents compact nuclear emission whose image-plane shape is specified directly, and
it is Fourier transformed to the visibility plane like every other light profile.

This script simulates an `Interferometer` dataset of a 'galaxy-scale' strong lens which has this point-source
emission in the centre of its lens galaxy.

This dataset is used in `interferometer/features/advanced/operated_light_profile/modeling.py` to demonstrate
how to fit this point-source emission using an operated light profile.

__Contents__

- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a name.
- **Grid:** Real-space grid the strong lens image is evaluated on.
- **uv-wavelengths:** Load the uv baselines used to NUFFT the image to the visibility plane.
- **Simulator:** `SimulatorInterferometer` (no PSF; uv-plane noise instead of image-plane Poisson noise).
- **Ray Tracing:** Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.
- **Output:** Output the simulated dataset to the dataset path as .fits files.
- **Visualize:** Output a subplot of the simulated dataset and the tracer's quantities to the dataset path.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file.

__Model__

This script simulates `Interferometer` data of a 'galaxy-scale' strong lens where:

 - The lens galaxy's light profile is an `Sersic` bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The lens galaxy has a point source of emission at its centre which is modeled as a operated `Gaussian`.
 - The source galaxy's light is an `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `interferometer/simulator.ipynb` notebook.
"""

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_type = "interferometer"
dataset_name = "light_operated"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Simulate the image using a (y,x) grid. Over-sampling is an imaging-only technique and is not used for
interferometer data.
"""
grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

"""
__uv-wavelengths__

To perform the Fourier transform we need the wavelengths of the baselines.
"""
uv_wavelengths_path = Path("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=Path(uv_wavelengths_path, "sma.fits"), hdu=0
)

"""
__Simulator__

Create the simulator for the interferometer data, which defines the exposure time, visibility-plane
noise sigma, and transformer.
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerDFT,
)

"""
__Ray Tracing__

Setup the lens galaxy's light (elliptical Sersic bulge + operated Gaussian point source), mass (Isothermal
and ExternalShear) and source galaxy light (cored elliptical Sersic) for this simulated lens.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    psf=al.lp_operated.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=100.0, sigma=0.1
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
        intensity=10.0,
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
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator the tracer, which creates the ray-traced image and NUFFTs it to visibilities.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Interferometer` dataset before outputting it to fits.
"""
aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_interferometer(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset and the tracer's quantities to the dataset path as .png files.
"""
aplt.subplot_interferometer_dirty_images(
    dataset=dataset, output_path=dataset_path, output_format="png"
)
aplt.subplot_tracer(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and
galaxies are safely stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
The dataset can be viewed in the folder `autolens_workspace/dataset/interferometer/light_operated`.
"""

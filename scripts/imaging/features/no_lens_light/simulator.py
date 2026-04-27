"""
Simulator: No Lens Light
========================

This script simulates `Imaging` of a 'galaxy-scale' which is identical to the `simple` simulated in the `start_here.py`
script, but where the lens galaxy's light is omitted.

It is used in `autolens_workspace/notebooks/modeling/features/no_lens_light.ipynb` to illustrate how to fit a
lens model to data where the lens galaxy's light is not present (e.g. because it is too faint to be detected).

__Contents__

**Model:** Compose the lens model fitted to the data.
**Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a.
**Simulate:** Simulate the image using a (y,x) grid with the adaptive over sampling scheme.
**Ray Tracing:** Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.
**Output:** Output the simulated dataset to the dataset path as .fits files.
**Visualize:** Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset.
**Mask Extra Galaxies:** Save an empty `mask_extra_galaxies.fits` so noise-scaling tutorials can load it.
**Tracer json:** Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. 
"""
dataset_type = "imaging"
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset", dataset_type, dataset_name)

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

the `lens_galaxy` below does not include a `bulge` or `disk` component and therefore has no lens light.
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
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

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
__Mask Extra Galaxies__

This dataset has no extra galaxies, but pixelization tutorials that load it (e.g.
`imaging/features/pixelization/modeling.py`, `imaging/features/pixelization/fit.py`) demonstrate the
noise-scaling API by applying a `mask_extra_galaxies` mask to the dataset. Output an empty (all-False,
no-pixels-masked) mask so those tutorials can call `apply_noise_scaling(mask=...)` without crashing on a
missing FITS file. The mask shape tracks `dataset.shape_native`, so `PYAUTO_SMALL_DATASETS=1` is honoured
automatically.
"""
mask_extra_galaxies = al.Mask2D.all_false(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
)

aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
    overwrite=True,
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
The dataset can be viewed in the folder `autolens_workspace/imaging/simple__no_lens_light`.
"""

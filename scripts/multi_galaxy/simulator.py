"""
Simulator: Multi Galaxy
=======================

This script simulates an example strong lens on the 'multi galaxy' scale, where two (or more) galaxies of
comparable mass both contribute significantly to the lensing of a single background source. Neither galaxy is a
minor perturber: they are **co-dominant deflectors**, and every one of them is modeled individually with its own
free light and mass profiles.

This is the defining feature of the multi-galaxy regime, and what separates it from the regimes either side of it:

 - Below it, `imaging/` lenses have a single dominant lens galaxy (any nearby galaxies are minor perturbers).
 - Above it, `group/` and `cluster/` lenses add a dominant dark-matter halo shared by the galaxies and tiers of
   member galaxies on scaling relations. **All groups and clusters are multi-galaxy systems, but not vice versa** —
   the multi-galaxy regime is the base rung of the ladder, with no host halo and no tiered member populations.

The simulated system is modeled on **SDSS J1011+0143** (Shu et al. 2016, ApJ 820, 43, arXiv:1602.02927): a close
merging pair of early-type galaxies (projected separation ~4.2 kpc, or ~0.9" at the lens redshift z=0.331) lensing
a Lyman-alpha emitter at z=2.701 into a wide (Einstein radius ~1.8") cross / arc configuration. It is one of the
cleanest known examples of two co-dominant deflectors: the published mass model is exactly two isothermal profiles,
and comparing the two mass centres with the two light centres revealed kiloparsec-scale mass/light offsets — a
result a single-galaxy lens model physically cannot produce.

This script simulates `Imaging` of a 'multi-galaxy' strong lens where:

 - The lens is a pair of galaxies of comparable mass, whose light distributions are `Sersic` profiles and whose
   total mass distributions are `Isothermal` profiles.
 - A single source galaxy is observed, whose `LightProfile` is a `SersicCore`.

__Contents__

- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a name.
- **Grid:** Define the 2d grid of (y,x) coordinates that the lens and source galaxy images are evaluated on.
- **Galaxy Centres:** Define the centres of the two main lens galaxies.
- **Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
- **PSF Convolution:** Define the Point Spread Function (PSF) that blurs the simulated image.
- **Main Lens Galaxies:** The two co-dominant lens galaxies of the merging pair.
- **Source Galaxy:** The source galaxy whose lensed images we simulate.
- **Ray Tracing:** Use all galaxies to setup a tracer, which generates the image of the simulated `Imaging`.
- **Dataset:** Simulate and plot the strong lens dataset.
- **Visualize:** Output a subplot of the simulated dataset to the dataset folder.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file.
- **Centre JSON Files:** Save the centres of the main lens galaxies as a JSON file.
- **Positions:** Solve for and save the lensed positions of the source.
- **JAX Variant:** Pointer to the JAX-jitted simulator pattern.

__Main Lens Galaxies__

Note on redshifts: this example uses the workspace-standard z_lens = 0.5 / z_source = 1.0 rather than the
real system's z = 0.331 / 2.701 — the pair separation and Einstein radius are chosen so the lensing
CONFIGURATION mirrors SDSS J1011+0143, not its exact distances (0.86" is ~5.3 kpc at z = 0.5).

In the group and cluster regimes, galaxies are split into tiers (main galaxies, extra galaxies, scaling galaxies)
because there are too many of them to model each one freely. The multi-galaxy regime needs no tiers: there are only
a handful of deflectors and each contributes comparably, so **every galaxy is a main lens galaxy** with its own
free light and mass model. The centres are saved to `main_lens_centres.json` so the modeling scripts can load them.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They
define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/data.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""
dataset_type = "multi_galaxy"
dataset_name = "simple"

"""
The path where the dataset will be output.

In this example, this is: `/autolens_workspace/dataset/multi_galaxy/simple`
"""
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2d grid of (y,x) coordinates that the lens and source galaxy images are evaluated and therefore simulated
on, via the inputs:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.

The 0.05" / pixel resolution matches Hubble Space Telescope ACS imaging, the data the real SDSS J1011+0143 pair
was modeled with.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

"""
__Galaxy Centres__

Define the centres of the two main lens galaxies. Their separation of ~0.9" matches the ~4.2 kpc projected
separation of the SDSS J1011+0143 merging pair at z=0.331. Both centres sit well inside the Einstein radius of the
combined mass distribution (~1.8"), which is what makes them co-dominant: the lensed images wrap around the pair
as a whole, and the data constrains both galaxies' masses.
"""
main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

An adaptive oversampling scheme is used, evaluating the central regions of each lens galaxy's light profile at a
resolution of 32x32, transitioning to 8x8 in intermediate areas, and 2x2 in the outskirts. This ensures precise and
accurate image simulation while focusing computational resources on the bright regions that demand higher
oversampling. The adaptive grid is centred on both main lens galaxies.

An adaptive oversampling grid cannot be defined for the lensed source because its light appears in different regions
of the image plane for each dataset. For this reason, most workspace examples use cored light profiles for the
source galaxy, whose gradual central variation is evaluated accurately without over-sampling.

Once you are more experienced, you should read up on over-sampling in more detail via
the `autolens_workspace/*/guides/advanced/over_sampling.ipynb` notebook.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
__PSF Convolution__

All CCD imaging data (e.g. Hubble Space Telescope, Euclid) are blurred by the telescope optics when they are imaged.

The Point Spread Function (PSF) describes the blurring of the image by the telescope optics, in the form of a
two dimensional convolution kernel. The lens modeling scripts use this PSF when fitting the data, to account for
this blurring of the image.

In this example, we use a simple 2D Gaussian PSF, which is convolved with the image of the lens and source galaxies
when simulating the dataset.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.08,
    pixel_scales=grid.pixel_scales,
    convolve_over_sample_size=1,  # Increase for PSF Oversampling
)

"""
To simulate the `Imaging` dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=900.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxies__

The two co-dominant lens galaxies of the merging pair. Their Einstein radii (1.0" and 0.8") are comparable — the
defining property of the multi-galaxy regime. Contrast this with the `imaging/` simulators (a single lens whose
companions, if any, are minor) and the `group/` simulator (one dominant lens plus smaller extra galaxies whose
Einstein radii are several times smaller).

Both galaxies use elliptical `Sersic` light and `Isothermal` mass profiles. Note the small offsets between each
galaxy's light centre and mass centre — kiloparsec-scale mass/light offsets in an interacting pair are exactly the
science the real SDSS J1011+0143 system delivered (Shu et al. 2016), and something only a model with two free mass
profiles can measure.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

main_lens_galaxies = [lens_0, lens_1]

"""
__Source Galaxy__

The source galaxy whose lensed images we simulate. It uses a cored Sersic profile so that adaptive over-sampling
is not required for the source.

The real SDSS J1011+0143 source is a Lyman-alpha emitter at z=2.701 which resolves into multiple star-forming
knots under lensing magnification; a single compact cored Sersic is the simple stand-in for it here.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

"""
__Ray Tracing__

Use all galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.

The tracer combines the two main lens galaxies and the source galaxy. Because both deflectors are at the same
redshift, this is single-plane ray tracing — the two galaxies' deflection fields simply add. (Two deflectors at
*different* redshifts is compound, multi-plane lensing — see the cluster package, where multi-plane tracing is the
default.)
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + [source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Dataset__

Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Lets plot the simulated `Imaging` dataset before we output it to fits.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
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
__Centre JSON Files__

Save the centres of the main lens galaxies as a JSON file. These are loaded by the multi-galaxy modeling scripts to
set up the lens model (initializing the centre priors of each galaxy's light and mass profiles).

Note there is no `extra_galaxies_centres.json` and no scaling-galaxy catalogue: in the multi-galaxy regime every
deflector is a main lens galaxy. The extra and scaling tiers only enter as feature extensions (see
`multi_galaxy/features`) and become the default at group and cluster scale.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

"""
__Positions__

Solve for the lensed positions of the source galaxy, which can be used as input for the modeling scripts to help
the non-linear search converge.
"""
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.05),
    pixel_scale_precision=0.001,
    magnification_threshold=0.01,
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

al.output_to_json(
    obj=positions,
    file_path=dataset_path / "positions.json",
)

"""
Finished.

__JAX Variant__

Same pattern as `scripts/imaging/simulator.py` `__JAX Variant__`:
instantiate `al.SimulatorImaging(use_jax=True)` and wrap
`via_tracer_from` in `@jax.jit`. The simulator handles pytree
registration internally.

See `scripts/imaging/simulator.py` for the runnable variant block.
"""

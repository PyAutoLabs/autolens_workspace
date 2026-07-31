"""
Simulator: No Lens Light (Multi Galaxy)
=======================================

This script simulates `Imaging` of a 'multi-galaxy' strong lens where neither co-dominant deflector has visible
light emission — only their mass profiles contribute to the ray-tracing. The source galaxy still has light.

This is the multi-galaxy analogue of `imaging/features/no_lens_light/simulator.py`. The mass model is unchanged
from `multi_galaxy/simulator.py`: the same pair of co-dominant `Isothermal` deflectors, the same Einstein radii
(1.0" and 0.8"), the same external shear and the same source. Only the light is removed. That makes the two
datasets directly comparable, which is the point — everything the modeling example says about what you lose is
measured against a dataset that differs in exactly one respect.

This script simulates `Imaging` of a 'multi-galaxy' strong lens where:

 - The lens is a pair of co-dominant galaxies whose total mass distributions are `Isothermal` profiles and which
   have no light profiles at all.
 - The system has a single overall `ExternalShear`, held at the system centre (0.0", 0.0").
 - A single source galaxy is observed, whose `LightProfile` is a `SersicCore`.

__Contents__

- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a name.
- **Grid:** Define the 2d grid of (y,x) coordinates the galaxy images are evaluated on.
- **Galaxy Centres:** The centres of the two deflectors — which are now mass centres, not light centres.
- **Over Sampling:** Over-sampling for the lensed arcs, with no galaxy light to resolve.
- **PSF Convolution:** Define the Point Spread Function (PSF) that blurs the simulated image.
- **Main Lens Galaxies:** The two co-dominant deflectors, mass only, no light.
- **External Shear:** The system's overall external shear, held at the system centre.
- **Source Galaxy:** The source galaxy whose lensed images we simulate.
- **Ray Tracing:** Use all galaxies to setup a tracer, which generates the image of the simulated `Imaging`.
- **Dataset:** Simulate and plot the strong lens dataset.
- **Visualize:** Output a subplot of the simulated dataset to the dataset folder.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file.
- **Centre JSON Files:** Save the centres of the two deflectors as a JSON file.
- **Positions:** Solve for and save the lensed positions of the source.

__Start Here Notebook__

If any code in this script is unclear, refer to the `multi_galaxy/simulator.ipynb` notebook.

__No Extra Galaxy__

Unlike the `simple` dataset, no faint contaminating galaxy is included here. In `simple` the extra galaxy exists so
the modeling examples can demonstrate the `__Extra Galaxies Noise Scaling__` step; with the lens light removed it
would be the only foreground emission in the image and would dominate the visual impression of a dataset whose
whole point is that the foreground is empty. The noise-scaling lever is orthogonal to this feature and is taught
where it belongs, in `multi_galaxy/modeling.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name. They
define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/multi_galaxy/simple__no_lens_light/data.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/multi_galaxy/simple__no_lens_light/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/multi_galaxy/simple__no_lens_light/psf.fits`.
"""
dataset_type = "multi_galaxy"
dataset_name = "simple__no_lens_light"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2d grid of (y,x) coordinates that the source galaxy's lensed image is evaluated and therefore simulated
on, via the inputs:

 - `shape_native`: The (y_pixels, x_pixels) 2D shape of the grid defining the shape of the data that is simulated.
 - `pixel_scales`: The arc-second to pixel conversion factor of the grid and data.

The 0.05" / pixel resolution matches Hubble Space Telescope ACS imaging, and is identical to
`multi_galaxy/simulator.py` so the two datasets can be compared pixel for pixel.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

"""
__Galaxy Centres__

Define the centres of the two co-dominant deflectors.

These are the **mass** centres. In `multi_galaxy/simulator.py` each galaxy's light and mass centres are deliberately
offset by ~0.05" from each other, reproducing the kiloparsec-scale mass/light offsets Shu et al. (2016) measured in
SDSS J1011+0143. With no lens light there is no light centre to offset from, so the values written to
`main_lens_centres.json` below are the mass centres themselves.

That is a bigger difference than it looks, and the modeling script builds on it: at galaxy scale the centre of a
lens is conventionally assumed to be near (0.0", 0.0") and the light tells you if it is not. Here there are two
centres, neither at the origin, and nothing in the image marks where they are.
"""
main_lens_centres = [(0.30, 0.28), (-0.31, -0.22)]

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

Because neither deflector has a light profile, there is no steep central emission needing adaptive over-sampling at
the galaxy centres — the usual reason for centring the adaptive scheme on every deflector does not apply. This is
the one place where removing the lens light makes the *numerics* simpler as well as the model smaller.

The lensed source light still requires accurate evaluation, so we apply over-sampling at the system centre, around
which the arcs wrap at the combined Einstein radius (~1.8").

An adaptive oversampling grid cannot be defined for the lensed source because its light appears in different regions
of the image plane for each dataset. For this reason we use a cored light profile for the source galaxy
(`SersicCore`), whose gradual central variation is evaluated accurately without heavy over-sampling.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
__PSF Convolution__

All CCD imaging data (e.g. Hubble Space Telescope, Euclid) are blurred by the telescope optics when they are imaged.

The Point Spread Function (PSF) describes the blurring of the image by the telescope optics, in the form of a
two dimensional convolution kernel. The lens modeling scripts use this PSF when fitting the data, to account for
this blurring of the image.
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

The two co-dominant deflectors. Their `Isothermal` mass profiles are identical to those in
`multi_galaxy/simulator.py` — same centres, same ellipticities, same Einstein radii (1.0" and 0.8") — but neither
galaxy is given a `bulge`.

In the list-based API used by the multi-galaxy modeling scripts, these are the `lens_0`, `lens_1`, ... entries of
the model, built in a loop over `main_lens_centres.json`. Removing the light changes what goes inside each entry,
not how many entries there are: every deflector is still a main lens galaxy, because co-dominance is a statement
about mass and mass is exactly what remains.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

main_lens_galaxies = [lens_0, lens_1]

"""
__External Shear__

The `ExternalShear` describes the tidal gravitational field of structure *outside* the system being simulated. It is
a property of the system as a whole rather than of any individual galaxy, so we give it its own entry at the system
centre (0.0", 0.0") instead of attaching it to one of the deflectors.

`ExternalShear` takes no `centre` argument because it is a uniform field defined about the coordinate origin, which
for this dataset is the centre of the lens pair.
"""
shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Source Galaxy__

The source galaxy whose lensed images we simulate. It uses a cored Sersic profile so that adaptive over-sampling
is not required for the source, and is identical to the source of `multi_galaxy/simulator.py`.
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

Because both deflectors are at the same redshift this is single-plane ray tracing — the two galaxies' deflection
fields, and the shear's, simply add. The simulated image contains only the lensed source emission, since no
foreground galaxy has light.
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + [shear_galaxy, source_galaxy])

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

Save the `Tracer` in the dataset folder as a .json file, ensuring the true mass profiles and galaxies are safely
stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the centres of the two deflectors as a JSON file, loaded by `modeling.py` and `slam.py` to initialize each
galaxy's mass centre prior.

Writing this file is not a convenience here in the way it is for `simple` — it is the only record of where the
deflectors are. For real data with no lens light in the observed band, the equivalent information has to come from
somewhere outside the image: a different band in which the galaxies are detected, a catalogue position, or the
lens light subtraction that produced the image in the first place. `modeling.py` discusses what to do when it is
uncertain.
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

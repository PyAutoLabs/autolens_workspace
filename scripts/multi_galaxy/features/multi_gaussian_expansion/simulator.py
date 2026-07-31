"""
Simulator: Multi Gaussian Expansion (Multi Galaxy)
==================================================

This script simulates `Imaging` of a 'multi-galaxy' strong lens whose co-dominant deflectors have **disturbed,
asymmetric light** that a single symmetric `Sersic` cannot represent.

That is the point of the dataset. The `simple` dataset of `multi_galaxy/simulator.py` gives each deflector one
elliptical `Sersic`, so an MGE fitted to it has nothing to demonstrate — a Sersic model would fit it exactly,
because a Sersic is what made it. Here each deflector's light is the sum of two offset, differently-oriented
components, producing isophotal twists and a radially varying ellipticity. Those are precisely the features an MGE
captures and a Sersic cannot.

This is not a contrived choice. The multi-galaxy regime is populated by interacting systems — the pair modelled by
this package is based on SDSS J1011+0143, a *merging* pair (Shu et al. 2016) — and tidally disturbed morphology is
the norm rather than the exception.

This script simulates `Imaging` of a 'multi-galaxy' strong lens where:

 - The lens is a pair of co-dominant galaxies, each of whose light is two offset `Sersic` components with different
   position angles, and whose total mass distributions are `Isothermal` profiles.
 - The system has a single overall `ExternalShear`, held at the system centre (0.0", 0.0").
 - A single source galaxy is observed, whose `LightProfile` is a `SersicCore`.

__Contents__

- **Dataset Paths:** Where the simulated dataset is written.
- **Grid:** The 2d grid of (y,x) coordinates the galaxy images are evaluated on.
- **Galaxy Centres:** The centres of the two co-dominant deflectors.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **PSF Convolution:** The Point Spread Function that blurs the simulated image.
- **Main Lens Galaxies:** The two deflectors, each with a twisted two-component light distribution.
- **External Shear:** The system's overall external shear.
- **Source Galaxy:** The source galaxy whose lensed images we simulate.
- **Ray Tracing:** Combine the galaxies into a tracer.
- **Dataset:** Simulate and plot the dataset.
- **Visualize:** Output a subplot of the simulated dataset.
- **Tracer json:** Save the `Tracer` as a .json file.
- **Centre JSON Files:** Save the deflector centres.
- **Positions:** Solve for the lensed source positions.

__Start Here Notebook__

If any code in this script is unclear, refer to the `multi_galaxy/simulator.ipynb` notebook.

__Why Two Components And Not A Truly Irregular Profile__

The truth here is still a sum of analytic profiles, because a dataset whose truth cannot be written down is a poor
teaching example — you could never say what the model *should* recover. Two offset Sersics at different angles is
the simplest truth that a single Sersic provably cannot fit, which is all this dataset needs to be.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

 - The image will be output to `/autolens_workspace/dataset/multi_galaxy/mge/data.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/multi_galaxy/mge/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/multi_galaxy/mge/psf.fits`.
"""
dataset_type = "multi_galaxy"
dataset_name = "mge"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

The 0.05" / pixel resolution matches Hubble Space Telescope ACS imaging, and is identical to
`multi_galaxy/simulator.py`.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

"""
__Galaxy Centres__

The centres of the two co-dominant deflectors, matching `multi_galaxy/simulator.py`'s ~0.9" separation.
"""
main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

"""
__Over Sampling__

Adaptive over-sampling centred on both deflectors, evaluating their bright central regions at 32x32 and falling
back to 2x2 in the outskirts.
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

All CCD imaging data are blurred by the telescope optics. The Point Spread Function describes that blurring as a
two dimensional convolution kernel.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.08,
    pixel_scales=grid.pixel_scales,
    convolve_over_sample_size=1,
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

Each deflector's light is **two** `Sersic` components rather than one:

 - An inner `bulge`, compact and round-ish.
 - An outer `disk`, larger, flatter, offset by ~0.05" and rotated by ~35 degrees from the bulge.

The offset and rotation are what make this dataset worth having. A single elliptical Sersic has one centre, one
axis ratio and one position angle at all radii; this galaxy's isophotes shift and twist as you move outward. That
is the "isophotal twist / radially varying ellipticity" an MGE is designed to capture.

The mass profiles are unchanged from `multi_galaxy/simulator.py` — same centres, same ellipticities, same Einstein
radii (1.0" and 0.8"). Only the light is made harder, so any difference in the inferred mass model between fitting
this dataset with a Sersic and with an MGE is attributable to the light model alone.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.2,
        effective_radius=0.3,
        sersic_index=4.0,
    ),
    disk=al.lp.Sersic(
        centre=(0.40, 0.30),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.6, angle=80.0),
        intensity=0.4,
        effective_radius=0.9,
        sersic_index=1.0,
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
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=120.0),
        intensity=1.0,
        effective_radius=0.25,
        sersic_index=4.0,
    ),
    disk=al.lp.Sersic(
        centre=(-0.40, -0.30),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.55, angle=155.0),
        intensity=0.35,
        effective_radius=0.8,
        sersic_index=1.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

main_lens_galaxies = [lens_0, lens_1]

"""
__External Shear__

Held in its own galaxy at the system centre (0.0", 0.0"), for the reasons given in `multi_galaxy/simulator.py`.
"""
shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Source Galaxy__

Identical to the source of `multi_galaxy/simulator.py`, so this dataset differs from `simple` in the deflectors'
light and nothing else.
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

Both deflectors are at the same redshift, so this is single-plane ray tracing — their deflection fields, and the
shear's, simply add.
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

Output a subplot of the simulated dataset and the image to the dataset path as .png files.
"""
aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, so the true light and mass profiles are available to
check how the dataset was simulated.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the centres of the co-dominant deflectors, loaded by the modeling scripts to build the `lens_i` loop.

Note these are the **bulge** centres. Each galaxy's disk is offset from its bulge by ~0.05", which is part of what
makes the morphology twisted — so "the centre of this galaxy" is already slightly ambiguous in this dataset, as it
is in real interacting systems.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

"""
__Positions__

Solve for the lensed positions of the source galaxy, which can be used to help the non-linear search converge.
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

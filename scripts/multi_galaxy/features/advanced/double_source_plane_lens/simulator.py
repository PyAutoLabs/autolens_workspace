"""
Simulator: DSPL (Multi Galaxy)
==============================

A double source-plane lens (DSPL) is a strong lens with **two source galaxies at different redshifts** behind the
same deflector. They appear as two distinct Einstein rings in the image plane.

This script simulates one where the deflector is a pair of co-dominant galaxies — the `simple` pair of
`multi_galaxy/simulator.py`, with a second source plane added behind the first.

The system has three planes rather than two, so ray tracing is multi-plane: light from the second source is
deflected by both co-dominant galaxies at z=0.5 **and** by the first source's own mass at z=1.0.

__Contents__

- **Dataset Paths:** Where the dataset is written.
- **Grid:** The 2d grid of (y,x) coordinates the galaxies are evaluated on.
- **Galaxy Centres:** The centres of the two co-dominant deflectors.
- **Over Sampling:** The adaptive over-sampling grid, centred on every deflector.
- **PSF Convolution:** The PSF that blurs the simulated image.
- **Main Lens Galaxies:** The two deflectors at z=0.5.
- **External Shear:** The system's overall shear, at the system centre.
- **Source Galaxies:** `source_0` at z=1.0 (light and mass), `source_1` at z=2.0 (light only).
- **Ray Tracing:** Build the multi-plane tracer.
- **Dataset:** Simulate and output the dataset.
- **Visualize:** Output the subplot.
- **Tracer json:** Save the truth tracer.
- **Centre JSON Files:** Save the deflector centres.

__Why source_1 Is Offset__

`source_1`'s centre is deliberately offset from `source_0`'s, rather than sitting behind it.

That offset is what makes this dataset useful for a multi-galaxy lens. The two sources are lensed by the same
deflection field, so a source directly behind the first would produce a second ring at essentially the same
image-plane positions — the same places the field is already being measured. Offsetting it puts the second ring's
images somewhere else, so the field is sampled where the first source's images do not reach.

`modeling.py` says what that buys.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/simulator.py`, which this script is a variation of, and
`imaging/features/advanced/double_source_plane_lens/simulator.py` for the single-deflector DSPL.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is output to `dataset/multi_galaxy/dspl`.
"""
dataset_type = "multi_galaxy"
dataset_name = "dspl"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

The same 200 x 200 grid at 0.05" / pixel used by `multi_galaxy/simulator.py`.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

"""
__Galaxy Centres__

The centres of the two co-dominant deflectors, unchanged from the `simple` dataset.
"""
main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

"""
__Over Sampling__

Adaptive over-sampling centred on every deflector.
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
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.08,
    pixel_scales=grid.pixel_scales,
    convolve_over_sample_size=1,
)

simulator = al.SimulatorImaging(
    exposure_time=900.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxies__

The two co-dominant deflectors, unchanged from the `simple` dataset.
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
__External Shear__
"""
shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Source Galaxies__

`source_0` at z=1.0 carries **both** light and mass. Its light forms the first Einstein ring; its mass deflects
the light of `source_1`, which is what makes the ray tracing multi-plane rather than two independent single-plane
problems.

`source_1` at z=2.0 carries light only, and is offset from `source_0` so its ring's images land in different
places — see `__Why source_1 Is Offset__` at the top of this script.

Both use cored profiles, so adaptive over-sampling is not needed for either.
"""
source_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
    mass=al.mp.IsothermalSph(
        centre=(0.0, 0.03),
        einstein_radius=0.15,
    ),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.25, 0.28),
        intensity=1.5,
        effective_radius=0.08,
    ),
)

"""
__Ray Tracing__

The tracer holds galaxies at three redshifts. PyAutoLens orders them internally, so the deflection chain runs:

 - z=0.5 — both co-dominant deflectors and the shear, whose deflection fields sum.
 - z=1.0 — `source_0`, whose light is traced back through the z=0.5 plane and whose mass deflects what passes it.
 - z=2.0 — `source_1`, traced back through both planes above.

`multi_galaxy/simulator.py`'s tracer has two planes; this one has three, and that is the whole structural
difference.
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + [shear_galaxy, source_0, source_1])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Dataset__
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__
"""
aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

"""
__Tracer json__

Save the truth `Tracer`, including both source planes' profiles and redshifts.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the deflector centres, which the modeling scripts load to drive their `lens_i` loop.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

"""
The dataset can be viewed in the folder `autolens_workspace/dataset/multi_galaxy/dspl`.
"""

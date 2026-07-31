"""
Simulator: Operated Light Profiles (Multi Galaxy)
=================================================

This script simulates a multi-galaxy strong lens where **each co-dominant deflector hosts a compact nuclear
point source** — an AGN — on top of its extended stellar light.

An operated light profile is one that is assumed to have already been convolved with the PSF, so the simulator
does not convolve it again. That is the right description of a point source: what the telescope records of an AGN
*is* the PSF, scaled by the source's brightness, so the profile written into the data should already carry that
convolution.

The dataset this writes is the `simple` pair with one extra component per deflector. Everything else — the
centres, the mass profiles, the shear, the source — is unchanged, so a fit to this dataset can be compared
directly against `multi_galaxy/modeling.py`'s.

__Contents__

- **Dataset Paths:** Where the dataset is written.
- **Grid:** The 2d grid of (y,x) coordinates the galaxies are evaluated on.
- **Galaxy Centres:** The centres of the two co-dominant deflectors.
- **Over Sampling:** The adaptive over-sampling grid, centred on every deflector.
- **PSF Convolution:** The PSF that blurs the simulated image.
- **Main Lens Galaxies:** The two deflectors, each with extended light, a nuclear point source and mass.
- **External Shear:** The system's overall shear, at the system centre.
- **Source Galaxy:** The lensed source.
- **Ray Tracing:** Build the tracer.
- **Dataset:** Simulate and output the dataset.
- **Visualize:** Output the subplot.
- **Tracer json:** Save the truth tracer.
- **Centre JSON Files:** Save the deflector centres.
- **Positions:** Solve for and save the source's multiple images.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/simulator.py`, which this script is a variation of and which
documents the simulation API in full.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is output to `dataset/multi_galaxy/operated`.
"""
dataset_type = "multi_galaxy"
dataset_name = "operated"

dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

The same 200 x 200 grid at 0.05" / pixel used by `multi_galaxy/simulator.py`, matching Hubble ACS imaging.
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

Adaptive over-sampling centred on every deflector. This matters more here than in `multi_galaxy/simulator.py`:
a point source is the steepest thing in the image, and there is one at each centre.
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

The PSF blurs everything in the image except the operated profiles, which are taken to carry it already.
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

Each deflector carries three components:

 - `bulge`: the extended stellar light, a `Sersic`, convolved with the PSF like any ordinary profile.
 - `point`: the nuclear point source, an `lp_operated.Gaussian`. Because it is operated, the simulator adds it to
   the image *after* convolution rather than before.
 - `mass`: the `Isothermal` total mass, unchanged from the `simple` dataset.

The two point sources are given different intensities. A pair of interacting galaxies is not required to host
equally active nuclei, and a model that assumes they do will push the difference into the extended light.

The `sigma` of each operated Gaussian matches the PSF's, because that is what an unresolved source looks like once
the optics have blurred it. Setting it wider would describe a marginally resolved nucleus instead.
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
    point=al.lp_operated.Gaussian(
        centre=(0.35, 0.25),
        ell_comps=(0.0, 0.0),
        intensity=2.5,
        sigma=0.08,
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
    point=al.lp_operated.Gaussian(
        centre=(-0.35, -0.25),
        ell_comps=(0.0, 0.0),
        intensity=1.2,
        sigma=0.08,
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

The system's overall shear, in its own galaxy at the system centre rather than attached to either deflector, as
`multi_galaxy/simulator.py` explains.
"""
shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Source Galaxy__

The lensed source, unchanged from the `simple` dataset. Its light is not operated — it is genuine extended
emission, which the telescope blurs like everything else.
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

Both deflectors are at the same redshift, so this is single-plane tracing and their deflection fields simply add.
The point sources contribute light only — they carry no mass of their own.
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + [shear_galaxy, source_galaxy])

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

Save the truth `Tracer`, so the profiles the dataset was simulated with can be checked later.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the deflector centres, which `modeling.py` loads to drive its `lens_i` loop.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

"""
__Positions__

Solve for the source's multiple images, used by the modeling script to constrain the mass model.
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
The dataset can be viewed in the folder `autolens_workspace/dataset/multi_galaxy/operated`.
"""

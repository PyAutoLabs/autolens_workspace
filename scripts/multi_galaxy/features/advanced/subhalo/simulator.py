"""
Simulator: Subhalo (Multi Galaxy)
=================================

This script simulates a multi-galaxy strong lens containing a **dark matter subhalo** — a compact, dark
perturber that distorts the lensed source's arcs without producing any light of its own.

The dataset is the `simple` pair with one extra mass component. Everything else — the deflector centres, their
masses, the shear, the source — is unchanged, so the perturbation the detection pipeline is looking for is the
only difference from `multi_galaxy/modeling.py`'s dataset.

__Contents__

- **Dataset Paths:** Where the dataset is written.
- **Grid:** The 2d grid of (y,x) coordinates the galaxies are evaluated on.
- **Galaxy Centres:** The centres of the two co-dominant deflectors.
- **Over Sampling:** The adaptive over-sampling grid, centred on every deflector.
- **PSF Convolution:** The PSF that blurs the simulated image.
- **Main Lens Galaxies:** The two deflectors.
- **External Shear:** The system's overall shear, at the system centre.
- **Subhalo:** The dark perturber.
- **Source Galaxy:** The lensed source.
- **Ray Tracing:** Build the tracer.
- **Dataset:** Simulate and output the dataset.
- **Visualize:** Output the subplot.
- **Tracer json:** Save the truth tracer.
- **Centre JSON Files:** Save the deflector centres.

__Where The Subhalo Is Placed__

The subhalo sits **on the arcs**, not at the system centre.

A subhalo is detectable only where it perturbs light that reaches the observer, which means near an image of the
source. One placed in an empty part of the image plane deflects nothing you can see, and the detection pipeline
would correctly find nothing.

__What This Dataset Is And Is Not For__

It is for demonstrating that the detection pipeline finds a subhalo that is genuinely there.

It is **not** a test of whether the pipeline avoids false positives, and `detect/start_here.py` is explicit
about why that matters more in this regime than at galaxy scale: a mis-split between two co-dominant deflectors
produces residuals in the same place and of the same character as a subhalo does. Establishing a false-positive
rate needs datasets simulated *without* a subhalo, which this folder does not provide.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/simulator.py`, which this script is a variation of, and
`imaging/features/advanced/subhalo/simulator.py` for the single-deflector version.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is output to `dataset/multi_galaxy/subhalo`.
"""
dataset_type = "multi_galaxy"
dataset_name = "subhalo"

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
__Subhalo__

An `NFWMCRLudlowSph` at the lens redshift, carrying mass and no light.

`mass_at_200` is the halo's virial mass. `NFWMCRLudlowSph` derives the concentration from it using the
Ludlow et al. mass-concentration relation, so the halo is described by a mass and a position rather than by
independent profile parameters — which is what makes it usable as a grid-search model.

`redshift_object` and `redshift_source` are required for that derivation: the concentration depends on the
halo's redshift, and converting it to a deflection depends on the source's.
"""
subhalo = al.Galaxy(
    redshift=0.5,
    mass=al.mp.NFWMCRLudlowSph(
        centre=(1.1, 0.6),
        mass_at_200=1.0e10,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

"""
__Source Galaxy__
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

The subhalo is at the same redshift as the deflectors, so this is still single-plane tracing — the deflection
field simply has one more term.
"""
tracer = al.Tracer(
    galaxies=main_lens_galaxies + [shear_galaxy, subhalo, source_galaxy]
)

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

Save the truth `Tracer`, including the subhalo's true mass and position — the values the detection pipeline is
trying to recover.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the deflector centres, which the detection pipeline loads to drive its `lens_i` loop.

The subhalo's position is deliberately **not** saved here. It is what the pipeline is searching for, and a
detection script that loaded the answer would not be demonstrating anything.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

"""
The dataset can be viewed in the folder `autolens_workspace/dataset/multi_galaxy/subhalo`.
"""

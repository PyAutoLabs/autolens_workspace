"""
Simulator: Stellar and Dark Mass (Multi Galaxy)
===============================================

This script simulates a multi-galaxy strong lens where **each co-dominant deflector's mass is decomposed into a
stellar component and a dark matter halo**, rather than described by a single total mass profile.

The stellar component is a light-and-mass profile: one `Sersic` describes both the galaxy's light and the mass
that traces it, tied together by a `mass_to_light_ratio`. The dark component is an `NFWSph` halo, which has no
light at all.

The dataset is otherwise the `simple` pair: the same centres, the same source, the same shear. Only the mass
parameterisation changes.

__Contents__

- **Dataset Paths:** Where the dataset is written.
- **Grid:** The 2d grid of (y,x) coordinates the galaxies are evaluated on.
- **Galaxy Centres:** The centres of the two co-dominant deflectors.
- **Over Sampling:** The adaptive over-sampling grid, centred on every deflector.
- **PSF Convolution:** The PSF that blurs the simulated image.
- **Main Lens Galaxies:** The two deflectors, each with a stellar component and a dark halo.
- **External Shear:** The system's overall shear, at the system centre.
- **Source Galaxy:** The lensed source.
- **Ray Tracing:** Build the tracer.
- **Dataset:** Simulate and output the dataset.
- **Visualize:** Output the subplot.
- **Tracer json:** Save the truth tracer.
- **Centre JSON Files:** Save the deflector centres.

__Why The Two Galaxies Get Different Mass-To-Light Ratios__

The two deflectors are given **different** `mass_to_light_ratio` values.

That choice is what makes this dataset able to say anything about `modeling.py`'s central question — whether to
tie the ratio across the two galaxies or fit it per galaxy. A dataset simulated with a shared ratio would make
the tied model correct by construction and teach nothing about the cost of assuming it.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/simulator.py`, which this script is a variation of, and
`imaging/features/advanced/mass_stellar_dark/simulator.py` for the single-deflector version.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The dataset is output to `dataset/multi_galaxy/mass_stellar_dark`.
"""
dataset_type = "multi_galaxy"
dataset_name = "mass_stellar_dark"

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

Each deflector carries two mass components instead of one:

 - `bulge`: an `lmp.Sersic` — a light **and** mass profile. Its light parameters are the same ones
   `multi_galaxy/simulator.py` uses, plus a `mass_to_light_ratio` converting that light into stellar mass.
 - `dark`: an `NFWSph` halo, centred on the galaxy, contributing mass and no light.

The two galaxies are given different mass-to-light ratios, for the reason at the top of this script.

Their dark halos differ too. A halo is not required to scale with the stellar component, and giving them
identical halos would build in an assumption the modelling scripts are meant to be able to test.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
        mass_to_light_ratio=0.6,
    ),
    dark=al.mp.NFWSph(
        centre=(0.35, 0.25),
        kappa_s=0.08,
        scale_radius=15.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
        mass_to_light_ratio=0.4,
    ),
    dark=al.mp.NFWSph(
        centre=(-0.35, -0.25),
        kappa_s=0.06,
        scale_radius=15.0,
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

Both deflectors are at the same redshift, so this is single-plane tracing. The deflection field now sums four
mass profiles rather than two — a stellar and a dark component per galaxy — plus the shear.
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

Save the truth `Tracer`, including both galaxies' mass-to-light ratios and halo parameters.
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
The dataset can be viewed in the folder `autolens_workspace/dataset/multi_galaxy/mass_stellar_dark`.
"""

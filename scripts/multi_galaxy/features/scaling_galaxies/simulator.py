"""
Simulator: Multi Galaxy Scaling Galaxies
========================================

Simulates the dataset for the `multi_galaxy/features/scaling_galaxies` feature: the co-dominant pair of
`multi_galaxy/simulator.py`, plus five faint galaxies scattered FAR from the lens whose masses follow a
luminosity scaling relation.

The tier's framing at this scale matters (see `features/README.md`): with no host halo there is no bound
member population — the scaling tier here is "a load of galaxies far from the lens", a weak collective
correction to the deflection field, not a standard model ingredient. Their mass profiles are
**untruncated** `IsothermalSph` (truncation encodes tidal stripping by a host halo, which this regime
lacks by definition; truncated dPIE members belong to the group/cluster workflows).

__Contents__

- **Dataset Paths / Grid / PSF / Simulator:** Standard imaging simulation setup.
- **Main Lens Galaxies:** The co-dominant pair (identical to `multi_galaxy/simulator.py`).
- **Scaling Galaxies:** Five faint, distant galaxies on an einstein_radius ~ L^0.5 relation.
- **Source / Dataset / Records:** Simulate, write the data, the centres JSONs and the scaling CSV.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths / Grid / PSF / Simulator__
"""
dataset_type = "multi_galaxy"
dataset_name = "scaling_galaxies"

dataset_path = Path("dataset", dataset_type, dataset_name)

grid = al.Grid2D.uniform(
    shape_native=(300, 300),
    pixel_scales=0.05,
)

main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

scaling_galaxies_centres = [
    (5.5, -4.5),
    (-5.0, 4.0),
    (3.5, 6.0),
    (-6.0, -3.5),
    (6.5, 2.5),
]
scaling_galaxies_luminosities = [0.40, 0.28, 0.20, 0.12, 0.08]

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres + scaling_galaxies_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.08,
    pixel_scales=grid.pixel_scales,
)

simulator = al.SimulatorImaging(
    exposure_time=900.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxies__

The co-dominant pair, identical to `multi_galaxy/simulator.py` (SDSS J1011+0143-like configuration).
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

"""
__Scaling Galaxies__

Five faint galaxies 4-7" from the pair, on an untruncated-isothermal luminosity relation:

    einstein_radius_i = einstein_radius_ref * (L_i / L_ref) ** 0.5

with truth `einstein_radius_ref = 0.15"` at `L_ref = 1.0`. An isothermal's deflection magnitude is constant
(its einstein radius, here 0.03-0.10"), but a constant deflection is degenerate with source position — what
matters is the *differential* deflection across the ring, roughly a percent-level shear
(gamma ~ theta_E / 2d ~ 0.01 for the closest galaxy). That is the regime where the tier is a refinement,
not a necessity.
"""
scaling_einstein_radius_ref_truth = 0.15
reference_luminosity = 1.0

scaling_galaxies = []
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    luminosity_ratio = luminosity / reference_luminosity
    scaling_galaxies.append(
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.SersicSph(
                centre=centre,
                intensity=luminosity,
                effective_radius=0.4,
                sersic_index=3.0,
            ),
            mass=al.mp.IsothermalSph(
                centre=centre,
                einstein_radius=scaling_einstein_radius_ref_truth
                * luminosity_ratio**0.5,
            ),
        )
    )

"""
__Source / Dataset / Records__
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

tracer = al.Tracer(galaxies=[lens_0, lens_1] + scaling_galaxies + [source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_imaging_dataset(dataset=dataset)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

al.output_to_json(
    obj=tracer,
    file_path=dataset_path / "tracer.json",
)

al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=dataset_path / "main_lens_centres.json",
)

al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=dataset_path / "scaling_galaxies.csv",
)

"""
Finished.
"""

"""
Simulator: Group Scaling Relation
=================================

This script simulates a group-scale strong lens with three populations of galaxies in the foreground, designed to
exercise the three-tier modeling API used by `group/features/scaling_relation/modeling.py`:

 - One **main lens galaxy** at the origin, which dominates the light and mass of the system.
 - Two **extra galaxies** offset from the lens, modelled individually in the fit (one free `sigma` per galaxy).
 - Two **scaling galaxies** further out, modelled via a shared luminosity-mass scaling relation.

Each population's centres are saved to a separate JSON file (`main_lens_centres.json`, `extra_galaxies_centres.json`,
`scaling_galaxies_centres.json`) so the modeling script can load them directly.

This dataset is independent of `dataset/group/simple` so the existing group examples are unaffected.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__
"""
dataset_type = "group"
dataset_name = "scaling_relation"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

A 250x250 0.1"/pixel grid wide enough to contain all five galaxies plus the lensed source.
"""
grid = al.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

Three populations:

 - `main_lens_centres`: primary lens at the origin.
 - `extra_galaxies_centres`: closer companions modelled individually in the fit.
 - `scaling_galaxies_centres`: further-out, fainter companions modelled via a shared scaling relation.

The scaling galaxies are deliberately placed further out and given fainter light profiles below — this matches the
typical observational scenario in which the scaling-relation tier is reserved for galaxies that contribute only a small
amount of lensing individually but matter collectively.
"""
main_lens_centres = [(0.0, 0.0)]
extra_galaxies_centres = [(3.5, 2.5), (-4.4, -5.0)]
scaling_galaxies_centres = [(6.5, 0.0), (-1.0, 7.0)]

all_galaxy_centres = (
    main_lens_centres + extra_galaxies_centres + scaling_galaxies_centres
)

"""
__Over Sampling__

Adaptive over-sampling is applied at every galaxy centre (main + extras + scaling) for accurate light evaluation.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_galaxy_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
A simple Gaussian PSF.
"""
psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxy__

A bright spherical Sersic + Isothermal mass at the origin.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

main_lens_galaxies = [lens_0]

"""
__Extra Galaxies__

Two companion galaxies modelled with their own light + mass.  These are the brighter, closer-in tier; in the modeling
script they receive their own free `sigma` parameter each.

Following the group and cluster convention, their mass profiles are tidally truncated `dPIEMassSph` profiles
(vanishing core `r_core = 0`, truncation radius `r_cut = 10.0`): members orbiting in the shared group potential
have their outer dark matter stripped by tides. The `sigma` values give Einstein radii of ~0.8" and ~1.0".
"""
extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(3.5, 2.5),
        sigma=212.0,
        r_core=0.0,
        r_cut=10.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(-4.4, -5.0),
        sigma=239.0,
        r_core=0.0,
        r_cut=10.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

extra_galaxies = [extra_galaxy_0, extra_galaxy_1]

"""
__Scaling Galaxies__

Two further-out, fainter companions whose true masses are placed EXACTLY on the reference-anchored tied
scaling relation the modeling script fits — truncated ``dPIEMassSph`` profiles (Bergamini et al. 2019
convention) with

    sigma = sigma_ref * (luminosity / reference_luminosity) ** 0.25
    r_cut = r_cut_ref * (luminosity / reference_luminosity) ** 0.7

with ``sigma_ref = 106.0`` km/s, ``r_cut_ref = 5.0`` and a fiducial reference luminosity
``reference_luminosity = 1.0`` (an explicit fixed constant — Lenstool's ``mag0`` — not the sample max; both
members share luminosity 0.45, so both get sigma = 86.8 km/s and r_cut = 2.86", an Einstein radius of ~0.136").
``r_core`` is fixed to zero — the vanishing-core standard, never scaled. Modelling these individually would add
2 free parameters; on a scaling relation they add zero (the single free normalization is shared across all
scaling galaxies, so adding more does not grow the model).
"""
scaling_sigma_ref_truth = 106.0
scaling_sigma_exponent = 0.25  # alpha
scaling_gamma = 0.2  # M/L tilt, universally fixed
scaling_rcut_exponent = 1.0 + scaling_gamma - 2.0 * scaling_sigma_exponent  # 0.7
scaling_r_cut_ref = 5.0
reference_luminosity = 1.0
scaling_truth_luminosities = [0.45, 0.45]

scaling_galaxies = []
for centre, luminosity in zip([(6.5, 0.0), (-1.0, 7.0)], scaling_truth_luminosities):
    luminosity_ratio = luminosity / reference_luminosity
    scaling_galaxies.append(
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.SersicSph(
                centre=centre, intensity=0.45, effective_radius=0.6, sersic_index=2.5
            ),
            mass=al.mp.dPIEMassSph(
                centre=centre,
                sigma=scaling_sigma_ref_truth * luminosity_ratio**scaling_sigma_exponent,
                r_core=0.0,
                r_cut=scaling_r_cut_ref * luminosity_ratio**scaling_rcut_exponent,
                redshift_object=0.5,
                redshift_source=1.0,
            ),
        )
    )

"""
__Source Galaxy__
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

"""
__Ray Tracing__

Tracer order: main lenses, extras, scaling galaxies, source.
"""
tracer = al.Tracer(
    galaxies=main_lens_galaxies + extra_galaxies + scaling_galaxies + [source_galaxy]
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

Save the truth tracer for later inspection.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

One JSON per population, loaded individually by the modeling script.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

al.output_to_json(
    obj=al.Grid2DIrregular(extra_galaxies_centres),
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

al.output_to_json(
    obj=al.Grid2DIrregular(scaling_galaxies_centres),
    file_path=Path(dataset_path, "scaling_galaxies_centres.json"),
)

"""
__Galaxy Population CSVs__

The modeling script loads luminosities (and centres) for both the extras and the scaling tier from CSVs written here.
The simulator knows the truth values of the per-galaxy luminosities so we write them out alongside the centre JSONs.

The CSV schema is `y, x, luminosity, redshift?` — see `al.galaxy_table_from_csv` /
`al.galaxy_table_to_csv` (`autogalaxy/galaxy/galaxy_table.py`). Centre JSONs above are kept for backward compatibility;
new consumers should prefer the CSVs.
"""
extra_galaxies_luminosities = [0.9, 0.9]
scaling_galaxies_luminosities = [0.45, 0.45]

al.galaxy_table_to_csv(
    centres=extra_galaxies_centres,
    luminosities=extra_galaxies_luminosities,
    file_path=Path(dataset_path, "extra_galaxies.csv"),
)

al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=Path(dataset_path, "scaling_galaxies.csv"),
)

"""
__Positions__

Solve for the lensed source positions; written for the SLaM-style pipelines that consume this dataset.
"""
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.1),
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

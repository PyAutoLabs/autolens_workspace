"""
Simulator: Multi Galaxy Scaling Relation
========================================

Simulates the dataset for the `multi_galaxy/features/scaling_relation` feature: the co-dominant pair of
`multi_galaxy/simulator.py`, plus five faint galaxies scattered FAR from the lens whose masses are tied to the
**brightest** of the pair by a Faber-Jackson relation.

    einstein_radius_i = einstein_radius_brightest * (L_i / L_brightest) ** 0.5

The anchor is the brighter of the co-dominant pair. That is the one structural difference from the
galaxy-scale version of this feature (`imaging/features/scaling_relation`), where there is only one lens and so no
choice to make. Here the anchor has to be *identified*, and `modeling.py` does so by luminosity, not by which
galaxy happens to be called `lens_0`.

Truth masses are derived from the relation rather than typed in: each galaxy's light profile is defined first, its
luminosity integrated from it with `luminosity_within_circle_from`, and its Einstein radius computed from the
relation. The fainter co-dominant galaxy is placed on the relation too — physically natural, though `modeling.py`
still frees its mass, because a co-dominant deflector is exactly the kind of galaxy you do not want to constrain by
a scaling law.

__Framing At This Scale__

The tier's framing here matters (see `features/README.md`): with no host halo there is no bound member population —
the scaling tier at multi-galaxy scale is "a load of galaxies far from the lens", a weak collective correction to
the deflection field, not a standard model ingredient. Fit without it first; add it if residuals or your science
case demand it.

The members sit at 5.5-7", well OUTSIDE the 3.0" mask used by `modeling.py`, so their light never enters that fit —
only their mass, whose deflections reach inside the mask. `slam.py` takes the other approach, fitting their light on
a deliberately enlarged mask in order to *measure* the luminosities this relation needs.

__Untruncated Profiles__

Their mass profiles are **untruncated** `IsothermalSph`. Truncation encodes tidal stripping by a host halo's
potential, which this regime lacks by definition; truncated `dPIEMass` members belong to the group and cluster
workflows.

__Contents__

- **Dataset Paths / Grid / PSF / Simulator:** Standard imaging simulation setup.
- **Luminosity Convention:** What the luminosity numbers mean.
- **Main Lens Galaxies:** The co-dominant pair, with the brightest galaxy identified by luminosity.
- **Scaling Galaxies:** Five faint, distant galaxies tied to the brightest galaxy.
- **Source / Dataset / Records:** Simulate, write the data, the centre JSONs and the CSVs.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths / Grid / PSF / Simulator__
"""
dataset_type = "multi_galaxy"
dataset_name = "scaling_relation"

dataset_path = Path("dataset", dataset_type, dataset_name)

grid = al.Grid2D.uniform(
    shape_native=(300, 300),
    pixel_scales=0.05,
)

main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

# The tier is placed as a fraction of the simulated grid's half-width rather than in absolute arcseconds, so it
# stays inside the frame when the grid shrinks (a capped smoke run, `PYAUTO_SMALL_DATASETS=1`, remakes the grid
# as 16 x 16 at 0.6", a half-width of 4.8", where the arcsecond positions below would all fall off the image and
# every measured luminosity would be zero). At the full resolution above the half-width is 7.5", which reproduces
# the tier's design positions exactly: (5.5, -4.5), (-5.0, 4.0), (3.5, 6.0), (-6.0, -3.5), (6.5, 2.5) arcseconds.
image_half_width = 0.5 * min(grid.shape_native) * grid.pixel_scales[0]

scaling_galaxies_centres = [
    (5.5 / 7.5 * image_half_width, -4.5 / 7.5 * image_half_width),
    (-5.0 / 7.5 * image_half_width, 4.0 / 7.5 * image_half_width),
    (3.5 / 7.5 * image_half_width, 6.0 / 7.5 * image_half_width),
    (-6.0 / 7.5 * image_half_width, -3.5 / 7.5 * image_half_width),
    (6.5 / 7.5 * image_half_width, 2.5 / 7.5 * image_half_width),
]

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
__Luminosity Convention__

Luminosities are integrated to a radius far larger than any galaxy here, so they are effectively total
luminosities. Only ratios to the brightest galaxy enter the relation, so the units are irrelevant.
"""
luminosity_radius = 100.0

"""
__Main Lens Galaxies__

The co-dominant pair, following `multi_galaxy/simulator.py` (an SDSS J1011+0143-like configuration). The first is
the brighter, so it is the anchor.

The brightest galaxy's Einstein radius is set directly; the second galaxy's is derived from the relation, which puts
the pair on a consistent Faber-Jackson locus.
"""
einstein_radius_brightest = 1.0

lens_0_bulge = al.lp.Sersic(
    centre=(0.35, 0.25),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.2,
    effective_radius=0.6,
    sersic_index=4.0,
)

lens_1_bulge = al.lp.Sersic(
    centre=(-0.35, -0.25),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
    intensity=1.0,
    effective_radius=0.5,
    sersic_index=4.0,
)

main_lens_luminosities = [
    bulge.luminosity_within_circle_from(radius=luminosity_radius)
    for bulge in [lens_0_bulge, lens_1_bulge]
]

brightest_index = int(np.argmax(main_lens_luminosities))
luminosity_brightest = main_lens_luminosities[brightest_index]

print(f"Main lens luminosities = {main_lens_luminosities}")
print(
    f"Brightest galaxy is lens_{brightest_index}, L_brightest = {luminosity_brightest:.4f}"
)


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the brightest galaxy.
    """
    return einstein_radius_brightest * (luminosity / luminosity_brightest) ** 0.5


lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=lens_0_bulge,
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=einstein_radius_from(main_lens_luminosities[0]),
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=lens_1_bulge,
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=einstein_radius_from(main_lens_luminosities[1]),
    ),
)

print(f"lens_0 einstein_radius = {lens_0.mass.einstein_radius:.4f}")
print(f"lens_1 einstein_radius = {lens_1.mass.einstein_radius:.4f}")

"""
__Scaling Galaxies__

Five faint galaxies 5.5-7" from the pair, on the same relation anchored to the brightest galaxy.

An isothermal's deflection magnitude is constant and equal to its Einstein radius, so these members deflect by
0.16-0.36" everywhere — not negligible next to the brightest galaxy's 1.0". A nearly uniform deflection is degenerate
with source position, though, so what actually matters is the *differential* deflection across the ring: a shear of
roughly
`gamma ~ theta_E / 2d`, which is ~2.5% for the closest member (0.36" at 7.1") and ~1% for the faintest. That is the
regime where this tier is a refinement rather than a necessity.
"""
scaling_galaxies_intensities = [0.40, 0.28, 0.20, 0.12, 0.08]

scaling_galaxies = []
scaling_galaxies_luminosities = []

for centre, intensity in zip(scaling_galaxies_centres, scaling_galaxies_intensities):
    bulge = al.lp.SersicSph(
        centre=centre, intensity=intensity, effective_radius=0.4, sersic_index=3.0
    )

    luminosity = bulge.luminosity_within_circle_from(radius=luminosity_radius)
    scaling_galaxies_luminosities.append(luminosity)

    scaling_galaxies.append(
        al.Galaxy(
            redshift=0.5,
            bulge=bulge,
            mass=al.mp.IsothermalSph(
                centre=centre, einstein_radius=einstein_radius_from(luminosity)
            ),
        )
    )

print("\nScaling tier (einstein_radius tied to the brightest galaxy):")
for centre, luminosity, galaxy in zip(
    scaling_galaxies_centres, scaling_galaxies_luminosities, scaling_galaxies
):
    print(
        f"  {str(centre):>16}  L = {luminosity:8.4f}  L/L_brightest = {luminosity / luminosity_brightest:7.5f}  "
        f"einstein_radius = {galaxy.mass.einstein_radius:.4f}"
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

al.output_to_json(
    obj=al.Grid2DIrregular(scaling_galaxies_centres),
    file_path=dataset_path / "scaling_galaxies_centres.json",
)

"""
__Luminosity CSVs__

Centres plus luminosities in the `y, x, luminosity` schema `al.galaxy_table_from_csv` reads. The modeling scripts
document the explicit-Python-list interface first and this CSV interface at the end; both are supported.

The main lenses get a CSV too, because the relation needs `L_brightest` — and, at this scale, needs to work out *which*
galaxy the brightest one is.
"""
al.galaxy_table_to_csv(
    centres=main_lens_centres,
    luminosities=main_lens_luminosities,
    file_path=dataset_path / "main_lens_galaxies.csv",
)

al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=dataset_path / "scaling_galaxies.csv",
)

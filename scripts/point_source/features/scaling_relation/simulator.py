"""
Simulator: Scaling Relation (Point Source)
==========================================

Simulates the dataset for the `point_source/features/scaling_relation` feature: a quadruply imaged point source
behind a lens with five foreground companions whose Einstein radii are tied to the lens's own by a Faber-Jackson
relation.

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

__Two Datasets, And Why That Is The Point__

This script writes **two** things, and the split is the whole lesson:

 - `point_dataset.json` — the positions and fluxes that are actually fitted. It contains no light whatsoever.
 - `data.fits` — accompanying CCD imaging of the same field, in which the lens and all five companions are visible.

A `PointDataset` cannot tell you where a companion sits, and it certainly cannot tell you how bright one is. Both
the centres and the luminosities the relation needs are measured from the accompanying imaging. By giving the
companions real light profiles here and integrating **those** profiles for the luminosities, this simulator plays the
part of that ancillary measurement explicitly rather than asserting some numbers.

So the truth chain runs: companion light (visible in `data.fits`) -> integrated luminosity -> Einstein radius via the
relation -> perturbed multiple image positions (in `point_dataset.json`). The modeling script walks back up it.

__Why The Relation Earns Its Keep Here More Than Anywhere Else__

A quad gives 8 positional data points; adding fluxes brings it to 12. Model each companion's `einstein_radius`
individually and five companions cost 5 free parameters on top of the lens's own — against a 12-point budget that is
close to unconstrained. Tie them and they cost **zero**. Point-source data is information-poor, so the saving is not
a convenience here, it is what makes the model fittable at all.

__Untruncated Profiles__

The companions are **untruncated** `IsothermalSph`. Truncation encodes tidal stripping by a host halo's potential and
a galaxy-scale lens has no host halo; truncated `dPIEMass` members belong to the group and cluster workflows.

__Contents__

- **Dataset Paths / Luminosity Convention:** Setup and what the luminosity numbers mean.
- **Main Lens Galaxy:** The anchor, with light (for the imaging) and mass.
- **Scaling Companions:** Five companions whose light sets their mass through the relation.
- **Source / Tracer:** The point source and the full tracer.
- **Point Solver:** Locate the multiple images, which already carry the tier's perturbation.
- **Positions / Fluxes / Point Dataset:** Add noise and write the point data.
- **Accompanying Imaging:** The CCD image the centres and luminosities are measured from.
- **Records:** Centre JSONs and luminosity CSVs.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths / Luminosity Convention__

Only ratios to the anchor enter the relation, so the units are irrelevant.
"""
dataset_type = "point_source"
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / dataset_type / dataset_name

luminosity_radius = 100.0

einstein_radius_anchor = 1.6

"""
__Main Lens Galaxy__

The anchor. Its light exists so it appears in the accompanying imaging and so its luminosity — the denominator of
every luminosity ratio — is a measured quantity rather than an assumed one.
"""
main_lens_centres = [(0.0, 0.0)]

main_lens_bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=1.0,
    effective_radius=0.8,
    sersic_index=4.0,
)

luminosity_anchor = main_lens_bulge.luminosity_within_circle_from(
    radius=luminosity_radius
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=main_lens_bulge,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=einstein_radius_anchor,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

print(f"Anchor luminosity      = {luminosity_anchor:.4f}")
print(f"Anchor einstein_radius = {einstein_radius_anchor:.4f}")


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the main lens.
    """
    return einstein_radius_anchor * (luminosity / luminosity_anchor) ** 0.5


"""
__Scaling Companions__

Five companions placed well outside the 1.6" Einstein radius, at radii of ~4.3-4.9". Each gets a light profile — this is
what a real observer sees and measures — and its Einstein radius is then *derived* from that light through the
relation.

They are far enough from the ring that they do not produce multiple images of their own, so the system remains a quad.
That constraint is what sets their distance: an earlier draft of this dataset placed them at 2-3", where their derived
Einstein radii reached 0.51" and the solver returned **six** images instead of four. Faber-Jackson is steep enough that
a companion only ~10x fainter than the lens is a strong perturber in its own right, so "faint neighbour" and
"harmless" are not the same statement.

Their effect on the four images is very large. Re-solving this system with the tier removed moves them by
**182, 398, 1596 and 1633 mas** — 36x to 327x the 5 mas astrometric precision adopted below. The reason is that a
multiple image position is the *solution* of the lens equation, not a linear readout of the deflection field: near the
ring the images lie along a nearly-degenerate direction, so a 0.2" deflection does not move an image by 0.2", it
slides it until ray-tracing rebalances. A model that omits this tier cannot fit these positions, and will distort the
main lens's mass distribution trying to absorb them.
"""
scaling_galaxies_centres = [
    (3.5, 2.5),
    (-2.8, -3.5),
    (4.2, -1.8),
    (-3.6, 3.2),
    (0.8, 4.8),
]

scaling_galaxies_intensities = [0.113, 0.085, 0.060, 0.040, 0.028]

scaling_galaxies = []
scaling_galaxies_luminosities = []

for centre, intensity in zip(scaling_galaxies_centres, scaling_galaxies_intensities):
    bulge = al.lp.SersicSph(
        centre=centre, intensity=intensity, effective_radius=0.5, sersic_index=2.5
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

print("\nScaling tier (einstein_radius tied to the anchor):")
for centre, luminosity, galaxy in zip(
    scaling_galaxies_centres, scaling_galaxies_luminosities, scaling_galaxies
):
    print(
        f"  {str(centre):>16}  L = {luminosity:8.4f}  L/L_anchor = {luminosity / luminosity_anchor:7.5f}  "
        f"einstein_radius = {galaxy.mass.einstein_radius:.4f}"
    )

"""
__Source / Tracer__

The source carries a faint extended light profile for visualization (so the multiple images are visible in the
accompanying imaging) and a `PointFlux` for the point data.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialCore(
        centre=(0.07, 0.07), intensity=0.1, effective_radius=0.02, radius_break=0.025
    ),
    point_0=al.ps.PointFlux(centre=(0.07, 0.07), flux=1.0),
)

tracer = al.Tracer(galaxies=[lens_galaxy] + scaling_galaxies + [source_galaxy])

"""
__Point Solver__

Locates the multiple images by ray tracing triangles from the image plane back to the source plane. The solver is
passed the tracer above, which includes the tier — so the positions it returns already carry the tier's perturbation.
That is the signal `modeling.py` recovers.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

print(f"\nMultiple images found: {len(positions)}")
print(positions)

"""
__Positions / Fluxes / Point Dataset__

A positional uncertainty of 0.005" (5 mas), representative of HST point-source astrometry, and a 5% relative flux
error — both discussed in full in `point_source/simulator.py`.

The fluxes matter here for the reason given in the header: they add 4 data points to the 8 the positions provide. With
the tier tied those 12 points support the model comfortably; with five free companion Einstein radii they would not.
"""
position_noise = 0.005

positions_with_noise = al.Grid2DIrregular(
    values=positions
    + np.random.normal(loc=0.0, scale=position_noise, size=positions.shape)
)

magnifications = al.LensCalc.from_tracer(
    tracer=tracer
).magnification_2d_via_hessian_from(grid=positions)

flux = 1.0
fluxes = al.ArrayIrregular(
    values=[flux * np.abs(magnification) for magnification in magnifications]
)

flux_rel_noise = 0.05

fluxes_with_noise = al.ArrayIrregular(
    values=fluxes
    + np.random.normal(
        loc=0.0, scale=flux_rel_noise * np.asarray(fluxes), size=len(fluxes)
    )
)

fluxes_noise_map = al.ArrayIrregular(values=flux_rel_noise * np.asarray(fluxes))

dataset = al.PointDataset(
    name="point_0",
    positions=positions_with_noise,
    positions_noise_map=position_noise,
    fluxes=fluxes_with_noise,
    fluxes_noise_map=fluxes_noise_map,
)

al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset.json",
)

"""
__Accompanying Imaging__

The CCD image a real observer would use, in which the lens and all five companions are visible. This is where the
centres and the luminosities come from — the point data cannot supply either.
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

imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.fits_imaging(
    dataset=imaging,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

aplt.subplot_point_dataset(dataset=dataset)
aplt.plot_array(array=imaging.data, title="Accompanying Imaging (companions visible)")

"""
__Records__

Centres per tier, and luminosities in the `y, x, luminosity` schema `al.galaxy_table_from_csv` reads. The modeling
scripts document the explicit-Python-list interface first and this CSV interface at the end.

For this regime the CSV is the more faithful representation, because both columns genuinely originate in a
photometric measurement on the imaging above rather than in anything the point-source fit computes.
"""
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

al.galaxy_table_to_csv(
    centres=main_lens_centres,
    luminosities=[luminosity_anchor],
    file_path=dataset_path / "main_lens_galaxies.csv",
)

al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=dataset_path / "scaling_galaxies.csv",
)

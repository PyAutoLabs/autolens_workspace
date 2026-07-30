"""
Simulator: Scaling Relation (Interferometer)
===========================================

Simulates the dataset for the `interferometer/features/scaling_relation` feature: a lens plus a tier of foreground
companions whose Einstein radii are tied to the main lens's own by a Faber-Jackson relation.

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

__Mass Only, And Why That Matters Here__

Foreground galaxy light is rarely detected at the long wavelengths an interferometer probes, so every interferometer
`extra_galaxies` example in this workspace is mass-only — and so is this one. The companions below have **no light
profile at all**.

That has a consequence the CCD-imaging version of this feature does not face: **the luminosities cannot come from
this dataset.** There is no foreground light in the data to fit, so there is nothing to integrate. They must come
from ancillary optical or near-infrared imaging of the same field — the same imaging you would use to locate the
companions in the first place.

This simulator therefore plays the role of that ancillary catalogue: it *defines* the luminosities, derives each
Einstein radius from them through the relation, and writes them to a CSV. In a real analysis that CSV is the output
of a photometric measurement on other data, not of anything in the interferometer pipeline. `slam.py` in this folder
is explicit about the same point.

__Untruncated Profiles__

The companions are **untruncated** `IsothermalSph`. Truncation encodes tidal stripping by a host halo's potential
and a galaxy-scale lens has no host halo; the truncated `dPIEMass` form of this tier belongs to the group- and
cluster-scale workflows.

__Contents__

- **Dataset Paths / Grid / uv-Wavelengths / Simulator:** Standard interferometer simulation setup.
- **Luminosity Convention:** What the luminosity numbers mean and where they would really come from.
- **Main Lens Galaxy:** The anchor.
- **Scaling Companions:** Five companions on the relation, mass only.
- **Source / Dataset / Records:** Simulate, write the data, the centre JSON and the luminosity CSVs.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths / Grid / uv-Wavelengths / Simulator__
"""
dataset_type = "interferometer"
dataset_name = "scaling_relation"

dataset_path = Path("dataset", dataset_type, dataset_name)

grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

uv_wavelengths_path = Path("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = al.ndarray_via_fits_from(
    file_path=Path(uv_wavelengths_path, "sma.fits"), hdu=0
)

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerDFT,
)

"""
__Luminosity Convention__

Only *ratios* to the anchor enter the relation, so the units are irrelevant — a magnitude catalogue converts via
`L / L_ref = 10 ** (0.4 * (m_ref - m))`.

The anchor's luminosity is given as a number rather than integrated from a light profile, because the lens has no
light profile in this dataset either. Both it and the companions' luminosities stand in for measurements made on
ancillary imaging.
"""
einstein_radius_anchor = 1.6

luminosity_anchor = 31.0962

"""
__Main Lens Galaxy__

Mass only, as in every interferometer example. This is the anchor: the relation is expressed relative to its
Einstein radius and its luminosity.
"""
main_lens_centres = [(0.0, 0.0)]

main_lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=einstein_radius_anchor,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

"""
__Faber-Jackson Relation__

One function, so the simulated truth and the modeling scripts cannot drift apart.
"""


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the main lens.
    """
    return einstein_radius_anchor * (luminosity / luminosity_anchor) ** 0.5


"""
__Scaling Companions__

Five companions 2-4" from the lens, on the relation. Their Einstein radii are *derived*, not typed in.

They sit closer than the galaxy-scale imaging example's tier because an interferometer's constraint comes from the
uv-plane visibilities of the lensed source, so what matters is how much a companion distorts the arcs — not whether
it happens to fall inside a chosen image mask.
"""
scaling_galaxies_centres = [
    (1.0, 3.5),
    (-2.0, -3.5),
    (3.0, -1.5),
    (-3.2, 1.2),
    (2.2, 2.4),
]

scaling_galaxies_luminosities = [1.4939, 1.0865, 0.7696, 0.4980, 0.2716]

scaling_galaxies = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(
            centre=centre, einstein_radius=einstein_radius_from(luminosity)
        ),
    )
    for centre, luminosity in zip(
        scaling_galaxies_centres, scaling_galaxies_luminosities
    )
]

print(f"Anchor: einstein_radius = {einstein_radius_anchor}, L = {luminosity_anchor}")
print("\nScaling tier (einstein_radius tied to the anchor):")
for centre, luminosity, galaxy in zip(
    scaling_galaxies_centres, scaling_galaxies_luminosities, scaling_galaxies
):
    print(
        f"  {str(centre):>16}  L = {luminosity:8.4f}  L/L_anchor = {luminosity / luminosity_anchor:7.5f}  "
        f"einstein_radius = {galaxy.mass.einstein_radius:.4f}"
    )

"""
__Source / Dataset / Records__
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[main_lens_galaxy] + scaling_galaxies + [source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

aplt.fits_interferometer(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

al.output_to_json(
    obj=tracer,
    file_path=dataset_path / "tracer.json",
)

"""
__Centre JSON__

The tier's centres. In a real analysis these come from ancillary imaging, not from the interferometer data.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(scaling_galaxies_centres),
    file_path=dataset_path / "scaling_galaxies_centres.json",
)

al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=dataset_path / "main_lens_centres.json",
)

"""
__Luminosity CSVs__

Centres plus luminosities in the `y, x, luminosity` schema `al.galaxy_table_from_csv` reads. The modeling scripts
document the explicit-Python-list interface first and this CSV interface at the end.

For this regime the CSV is the more realistic of the two, because the numbers genuinely come from a separate
photometric catalogue rather than from anything the pipeline computes.
"""
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

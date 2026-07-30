"""
Simulator: Scaling Relation
===========================

Simulates the dataset for the `imaging/features/scaling_relation` feature: a galaxy-scale lens plus two tiers of
foreground companions.

 - Two **bounded** companions close to the lens, whose Einstein radii the model frees individually within a
   luminosity-derived bound.
 - Five **scaling** companions further out, whose Einstein radii the model ties to the main lens's own Einstein
   radius through a Faber-Jackson relation, adding zero free parameters.

The dataset's defining property is that the truth Einstein radii are **derived from the relation rather than
typed in**. Light is the input, because light is what you observe: each companion gets a light profile, its
luminosity is integrated from that profile with `luminosity_within_circle_from`, and its Einstein radius then
follows

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

with the main lens as the anchor. The modeling, fit and likelihood scripts recover exactly this relation, so a
mismatch between them and this file is a real error rather than a mistuned constant.

__Faber-Jackson Is Steep__

`einstein_radius ~ sigma^2` and `L ~ sigma^4` together give `L ~ einstein_radius^2`, so the relation is steep: a
companion with a 0.35" Einstein radius beside a 1.6" main lens is already ~20x less luminous, and the faintest
member here is ~115x fainter. It is worth checking that against your intuition before reading the numbers printed
below — a galaxy that looks negligible in an image is not necessarily negligible in mass.

Note also what "small perturbation" does and does not mean here. An isothermal's deflection magnitude is constant
and equal to its Einstein radius, so these members deflect by 0.15-0.35" everywhere, which is not tiny next to the
anchor's 1.6". A nearly uniform deflection is degenerate with the source position, though, so the quantity that
actually matters is the **differential** deflection across the lensed images — a shear of roughly
`theta_E / 2d`, i.e. a few percent for members ~5" out. `fit.py` shows this explicitly.

__Untruncated Profiles__

Every mass profile here is an **untruncated** `IsothermalSph`. Truncation encodes tidal stripping by a host halo's
potential, and a galaxy-scale lens has no host halo. The truncated `dPIEMass` form of this same tier belongs to
the group- and cluster-scale workflows (`group/features/group_halo`, `cluster/modeling.py`), where a host
potential does exist.

__Contents__

- **Dataset Paths / Grid / PSF / Simulator:** Standard imaging simulation setup.
- **Luminosity Convention:** What the luminosity numbers mean and why only ratios matter.
- **Main Lens Galaxy:** The anchor — its light sets `L_anchor`, its mass sets `einstein_radius_anchor`.
- **Faber-Jackson Relation:** The one function both tiers derive their masses from.
- **Bounded Companions:** Two close companions, still placed on the relation.
- **Scaling Companions:** Five fainter, further-out companions on the relation.
- **Source / Dataset / Records:** Simulate, write the data, the centre JSONs and the luminosity CSVs.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths / Grid / PSF / Simulator__

A galaxy-scale field of view: 130x130 pixels at 0.1"/pixel = 13" wide, enclosing the lens, both companion tiers
and the lensed source.
"""
dataset_type = "imaging"
dataset_name = "scaling_relation"

dataset_path = Path("dataset", dataset_type, dataset_name)

grid = al.Grid2D.uniform(
    shape_native=(130, 130),
    pixel_scales=0.1,
)

main_lens_centres = [(0.0, 0.0)]

bounded_galaxies_centres = [(2.0, 1.5), (-1.5, -2.0)]

scaling_galaxies_centres = [
    (5.0, -1.0),
    (-1.0, 5.0),
    (3.5, 3.5),
    (-4.0, -2.5),
    (1.5, -4.5),
]

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres + bounded_galaxies_centres + scaling_galaxies_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

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
__Luminosity Convention__

Luminosities are integrated out to a radius far larger than any galaxy here, so they are effectively total
luminosities. Only *ratios* to the anchor enter the relation, so the units are irrelevant — a magnitude catalogue
converts via `L / L_ref = 10 ** (0.4 * (m_ref - m))`.
"""
luminosity_radius = 100.0

"""
__Main Lens Galaxy__

The anchor. Its integrated luminosity is `L_anchor` and its Einstein radius is `einstein_radius_anchor`; every
other galaxy's mass below is expressed relative to these two numbers.
"""
einstein_radius_anchor = 1.6

main_lens_bulge = al.lp.SersicSph(
    centre=(0.0, 0.0), intensity=0.7, effective_radius=1.5, sersic_index=3.0
)

luminosity_anchor = main_lens_bulge.luminosity_within_circle_from(
    radius=luminosity_radius
)

main_lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=main_lens_bulge,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=einstein_radius_anchor),
)

print(f"Anchor luminosity      = {luminosity_anchor:.4f}")
print(f"Anchor einstein_radius = {einstein_radius_anchor:.4f}")

"""
__Faber-Jackson Relation__

One function, used for both tiers, so the simulated truth and the modeling scripts cannot drift apart.
"""


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the main lens.
    """
    return einstein_radius_anchor * (luminosity / luminosity_anchor) ** 0.5


"""
__Bounded Companions__

Two close companions, bright enough that the model gives each its own free Einstein radius (bounded by its
luminosity) rather than tying it. They are still *placed* on the relation here, so the bound in `modeling.py`
brackets the truth.
"""
bounded_galaxies_intensities = [0.5, 0.4]

bounded_galaxies = []
bounded_galaxies_luminosities = []

for centre, intensity in zip(bounded_galaxies_centres, bounded_galaxies_intensities):
    bulge = al.lp.SersicSph(
        centre=centre, intensity=intensity, effective_radius=0.6, sersic_index=2.5
    )

    luminosity = bulge.luminosity_within_circle_from(radius=luminosity_radius)
    bounded_galaxies_luminosities.append(luminosity)

    bounded_galaxies.append(
        al.Galaxy(
            redshift=0.5,
            bulge=bulge,
            mass=al.mp.IsothermalSph(
                centre=centre, einstein_radius=einstein_radius_from(luminosity)
            ),
        )
    )

"""
__Scaling Companions__

Five fainter companions ~5" from the lens, on the same relation. In `modeling.py` this tier costs zero free
parameters no matter how many rows are added here.
"""
scaling_galaxies_intensities = [0.33, 0.24, 0.17, 0.11, 0.06]

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

"""
The truth table. These are the numbers `modeling.py`, `fit.py` and `likelihood_function.py` consume — note how
far the luminosity ratios fall for a modest fall in Einstein radius, which is the steepness discussed in the
header.
"""
print("\nBounded tier (free einstein_radius, luminosity-bounded):")
for centre, luminosity, galaxy in zip(
    bounded_galaxies_centres, bounded_galaxies_luminosities, bounded_galaxies
):
    print(
        f"  {str(centre):>16}  L = {luminosity:8.4f}  L/L_anchor = {luminosity / luminosity_anchor:7.5f}  "
        f"einstein_radius = {galaxy.mass.einstein_radius:.4f}"
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
__Source / Dataset / Records__

Tracer order: main lens, bounded tier, scaling tier, source.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.2,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(
    galaxies=[main_lens_galaxy] + bounded_galaxies + scaling_galaxies + [source_galaxy]
)

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

"""
__Centre JSONs__

One file per tier — which file a galaxy's centre appears in is what decides how the model treats it.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=dataset_path / "main_lens_centres.json",
)

al.output_to_json(
    obj=al.Grid2DIrregular(bounded_galaxies_centres),
    file_path=dataset_path / "extra_galaxies_centres.json",
)

al.output_to_json(
    obj=al.Grid2DIrregular(scaling_galaxies_centres),
    file_path=dataset_path / "scaling_galaxies_centres.json",
)

"""
__Luminosity CSVs__

The same centres plus their luminosities, in the `y, x, luminosity` schema `al.galaxy_table_from_csv` reads. The
modeling scripts document the explicit-Python-list interface first and this CSV interface at the end; both are
supported, and this file writes the inputs for both.

The main lens gets a CSV too, because the relation needs `L_anchor` just as much as it needs the member
luminosities.
"""
al.galaxy_table_to_csv(
    centres=main_lens_centres,
    luminosities=[luminosity_anchor],
    file_path=dataset_path / "main_lens_galaxies.csv",
)

al.galaxy_table_to_csv(
    centres=bounded_galaxies_centres,
    luminosities=bounded_galaxies_luminosities,
    file_path=dataset_path / "extra_galaxies.csv",
)

al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=dataset_path / "scaling_galaxies.csv",
)

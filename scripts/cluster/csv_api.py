"""
CSV API: Cluster
================

Cluster lens modelling uses a small family of CSV files as its canonical
input format. Rather than composing a tracer and a model inline in Python,
the cluster workflow stores every component — main galaxies, dark-matter
halo, source light, source point components, scaling-tier members — in
spreadsheet-editable CSVs that are then loaded into PyAutoLens objects.

This guide walks through every CSV in the cluster surface, in the order
they are usually consumed:

 1. **Point dataset CSV** — ``point_datasets.csv`` — the observational data
    (multiple-image positions per source). Loaded *before* modelling, as is
    the autolens convention. Produced by ``simulator.py`` for the bundled
    cluster dataset; this guide builds an illustrative one to show the
    schema.
 2. **Model CSVs (this PR)** — ``mass.csv``, ``light.csv``, ``point.csv``
    — one CSV per profile family, each row carrying a galaxy name, an
    attribute name, a profile class, and the constructor parameters of
    that profile. The lens galaxies, the host dark-matter halo, and the
    background sources all share the same three CSVs (grouping is by
    profile family, not by tier). Joined across the family CSVs to
    produce a full ``Galaxy`` (or ``af.Model[Galaxy]``).
 3. **Scaling galaxies CSV** — ``scaling_galaxies.csv`` — the legacy
    3-column ``y, x, luminosity`` schema for the scaling-relation tier
    that drove the original CSV-first cluster work. Kept as-is because
    naming each scaling-tier member would add overhead with no signal.

Everything this guide writes goes into ``dataset/cluster/csv_api_example/``
(a scratch folder dedicated to this script). The canonical cluster dataset
at ``dataset/cluster/simple/`` is produced by ``simulator.py`` and is what
``modeling.py`` and ``start_here.py`` consume.

__Why CSVs?__

At cluster scale you might have tens of main galaxies, hundreds of member
galaxies, and a dozen background sources. Building all of that inline in
Python becomes unwieldy and error-prone. CSVs make every component:

 - **Spreadsheet-editable** — open in Excel / LibreOffice / any text
   editor, tweak a value, save, re-run. No Python edit needed.
 - **Diff-friendly** — git diff on a CSV is line-by-line, so a change to
   one galaxy's mass is one line in the diff.
 - **Easy to scale up** — adding more galaxies is a row append, not a
   Python loop edit. The number of free parameters in the model does not
   grow with the population size (see scaling galaxies below).
 - **Round-trippable** — write a Python model out, read it back, get the
   same Python model. This guide demonstrates that explicitly.

__Contents__

- **Imports & Output Path** — set up the scratch folder.
- **Point Dataset CSV** — write and load ``point_datasets.csv``.
- **Galaxy Model CSV API** — write, read, and round-trip family CSVs.
- **Scaling Galaxies CSV** — the legacy scaling-tier schema.
- **af.Model[Galaxy] from CSVs** — building modelling-ready objects.
- **Wrap Up** — where each CSV lives in the canonical cluster workflow.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al

"""
__Imports & Output Path__

Everything this guide writes goes into ``dataset/cluster/csv_api_example/``,
a scratch folder dedicated to this script. The canonical cluster dataset
lives at ``dataset/cluster/simple/`` and is produced by ``simulator.py``;
this guide is illustrative and does not pair to that dataset.
"""
output_path = Path("dataset") / "cluster" / "csv_api_example"
output_path.mkdir(exist_ok=True, parents=True)


"""
__Point Dataset CSV__

We begin with the observational data CSV because convention is "load data
before modelling". For point-source cluster lensing the observational data
is the image-plane positions of each multiply-imaged source.

``al.PointDataset`` carries a source's ``name`` (which pairs to a ``Point``
component in the model), its ``positions`` (image-plane (y, x) coordinates
of the multiple images), a ``positions_noise_map`` (per-position
positional uncertainty), and an optional ``redshift``. A real cluster
typically has one ``PointDataset`` per background source.

The CSV schema is one row per observed image, grouped by ``name``:

  ``name, y, x, positions_noise, redshift?``

All rows sharing a ``name`` belong to the same source. The ``redshift``
column is consistent within a group (validated on load).
"""
point_dataset_0 = al.PointDataset(
    name="point_0",
    positions=al.Grid2DIrregular(
        [
            (-9.0, -19.3),
            (0.04, -0.08),
            (1.84, 23.3),
        ]
    ),
    positions_noise_map=0.005,
    redshift=1.0,
)

point_dataset_1 = al.PointDataset(
    name="point_1",
    positions=al.Grid2DIrregular(
        [
            (-15.96, 18.99),
            (0.68, -0.51),
            (13.65, -14.32),
        ]
    ),
    positions_noise_map=0.005,
    redshift=2.0,
)

"""
Write the combined dataset to a single CSV.
"""
al.output_to_csv(
    datasets=[point_dataset_0, point_dataset_1],
    file_path=output_path / "point_datasets.csv",
)

"""
Read it back. ``al.list_from_csv`` returns a ``List[PointDataset]`` with
each entry carrying the per-source positions, noise, and redshift.
"""
loaded_point_datasets = al.list_from_csv(file_path=output_path / "point_datasets.csv")

print("=" * 64)
print("Point datasets loaded from point_datasets.csv:")
print("=" * 64)
for dataset in loaded_point_datasets:
    print(f"  name={dataset.name!r}  redshift={dataset.redshift}")
    print(f"  positions={dataset.positions.in_list}")
    print(f"  noise={dataset.positions_noise_map}")
    print()


"""
__Galaxy Model CSV API__

The model CSV API stores every galaxy component (main galaxies, the dark
matter halo, sources) in three family-level CSVs:

 - ``mass.csv``   — all mass profiles (``dPIEMassSph``, ``NFWMCRLudlowSph``, etc.)
 - ``light.csv``  — all light profiles (``SersicSph``, ``SersicCore``, etc.)
 - ``point.csv``  — all point-source components (``Point``)

Each row carries:

 - ``galaxy``       — galaxy name. Rows sharing this name compose into one
                      ``Galaxy``.
 - ``attr_name``    — attribute name to bind under on the Galaxy (e.g.
                      ``mass``, ``bulge``, ``point_0``).
 - ``profile_class``— the concrete class name (looked up via ``getattr``
                      against the family namespace ``al.mp`` / ``al.lp`` /
                      ``al.ps``).
 - ``<params>``     — the profile's constructor arguments. Tuples like
                      ``centre`` split into ``y`` / ``x`` columns; other
                      tuples (e.g. ``ell_comps``) into ``<name>_0`` /
                      ``<name>_1``.
 - ``redshift``     — optional per-row Galaxy redshift (consistent across
                      every row for the same galaxy or all blank).

**Sparse columns are supported.** Different profile classes inside the
same family CSV can use disjoint parameter columns; cells unused by a
row's class are simply blank. This is what lets one ``main_lens_mass.csv``
carry both 2 ``dPIEMassSph`` cluster members and 1 ``NFWMCRLudlowSph``
host halo even though their parameter sets don't overlap.

We start by building a small illustrative cluster in Python.
"""
redshift_lens = 0.5
source_redshifts = [1.0, 2.0]

# One dict per profile family. Each maps {galaxy_name: {attr_name: profile_instance}}.
# Mass family carries the 2 main-lens dPIE profiles + the host-halo NFW; light family
# carries the 2 main-lens Sersic bulges + the 2 source-galaxy SersicCore bulges; point
# family carries the 2 source-galaxy Point components.

mass_profiles = {
    "lens_0": {
        "mass": al.mp.dPIEMassSph(
            centre=(0.0, 0.0),
            sigma=330.0,
            r_core=8.0,
            r_cut=20.0,
            redshift_object=redshift_lens,
            redshift_source=max(source_redshifts),
        )
    },
    "lens_1": {
        "mass": al.mp.dPIEMassSph(
            centre=(10.0, 8.0),
            sigma=210.0,
            r_core=5.0,
            r_cut=12.0,
            redshift_object=redshift_lens,
            redshift_source=max(source_redshifts),
        )
    },
    "host_halo": {
        "dark": al.mp.NFWMCRLudlowSph(
            centre=(0.0, 0.0),
            mass_at_200=10**15.3,
            redshift_object=redshift_lens,
            redshift_source=max(source_redshifts),
        )
    },
}

light_profiles = {
    "lens_0": {
        "bulge": al.lp.SersicSph(
            centre=(0.0, 0.0), intensity=1.5, effective_radius=3.0, sersic_index=4.0
        )
    },
    "lens_1": {
        "bulge": al.lp.SersicSph(
            centre=(10.0, 8.0), intensity=0.8, effective_radius=1.5, sersic_index=3.5
        )
    },
    "source_0": {
        "bulge": al.lp.SersicCore(
            centre=(0.3, 0.5),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        )
    },
    "source_1": {
        "bulge": al.lp.SersicCore(
            centre=(-0.8, 1.2),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=90.0),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        )
    },
}

point_profiles = {
    "source_0": {"point_0": al.ps.Point(centre=(0.3, 0.5))},
    "source_1": {"point_1": al.ps.Point(centre=(-0.8, 1.2))},
}

"""
We now have a complete cluster model expressed as nested dicts. Time to
write each profile family to its own CSV.

The ``redshifts`` argument is a flat ``{galaxy_name: redshift}`` mapping
so the writer can emit the per-row ``redshift`` column. Mains + halo sit
at the lens redshift; sources carry their per-source redshift.
"""
mass_csv = output_path / "mass.csv"
light_csv = output_path / "light.csv"
point_csv = output_path / "point.csv"

redshifts_by_galaxy = {
    "lens_0": redshift_lens,
    "lens_1": redshift_lens,
    "host_halo": redshift_lens,
    "source_0": source_redshifts[0],
    "source_1": source_redshifts[1],
}

al.galaxy_models_to_csv(
    profiles_by_galaxy=mass_profiles,
    file_path=mass_csv,
    family="mass",
    redshifts=redshifts_by_galaxy,
)

al.galaxy_models_to_csv(
    profiles_by_galaxy=light_profiles,
    file_path=light_csv,
    family="light",
    redshifts=redshifts_by_galaxy,
)

al.galaxy_models_to_csv(
    profiles_by_galaxy=point_profiles,
    file_path=point_csv,
    family="point",
    redshifts=redshifts_by_galaxy,
)

"""
Read each family CSV back. ``al.galaxy_models_from_csv`` returns a typed
``GalaxyModelTable`` with one ``GalaxyModelRow`` per CSV row: each row
carries the resolved ``profile_class`` (now a Python class, not a string)
and the parameter dict it needs.
"""
mass_table = al.galaxy_models_from_csv(mass_csv, family="mass")
light_table = al.galaxy_models_from_csv(light_csv, family="light")
point_table = al.galaxy_models_from_csv(point_csv, family="point")

for label, table in [
    ("mass.csv", mass_table),
    ("light.csv", light_table),
    ("point.csv", point_table),
]:
    print("=" * 64)
    print(f"{label}:")
    print("=" * 64)
    for row in table.rows:
        print(
            f"  galaxy={row.galaxy!r}  attr_name={row.attr_name!r}  "
            f"class={row.profile_class.__name__}  redshift={row.redshift}"
        )
        print(f"    params={row.params}")
    print()


"""
Join the family tables into ``Galaxy`` instances. ``al.galaxies_from_csv_tables``
takes one or more ``GalaxyModelTable``s and groups rows by the ``galaxy``
column, attaching each profile under its ``attr_name``. Per-galaxy redshift
consistency is enforced across every family CSV that mentions the galaxy.
"""
galaxies = al.galaxies_from_csv_tables(
    mass_table,
    light_table,
    point_table,
)

print("=" * 64)
print("Galaxies built from CSVs (al.galaxies_from_csv_tables):")
print("=" * 64)
for name, galaxy in galaxies.items():
    attr_summary = ", ".join(
        f"{a}={type(getattr(galaxy, a)).__name__}"
        for a in vars(galaxy)
        if not a.startswith("_") and a != "redshift"
    )
    print(f"  {name}: redshift={galaxy.redshift}  attrs=[{attr_summary}]")
print()


"""
__af.Model[Galaxy] from CSVs__

The concrete ``Galaxy`` instances above are useful for visualisation and
for the simulator (which needs concrete parameter values to produce a
truth tracer). For *modelling*, what you want is an ``af.Model[Galaxy]``
where some parameters are free (priors) and others are fixed.

``al.galaxy_af_models_from_csv_tables`` returns the same dict keyed by
galaxy name, but each value is an ``af.Model[Galaxy]`` with concrete CSV
values as fixed defaults. To free a parameter for the non-linear search,
mutate the returned model:

```python
galaxy_models["lens_0"].mass.sigma = af.UniformPrior(lower_limit=50.0, upper_limit=600.0)
galaxy_models["lens_0"].mass.r_core = af.UniformPrior(lower_limit=1.0, upper_limit=15.0)
galaxy_models["lens_0"].mass.r_cut = af.UniformPrior(lower_limit=5.0, upper_limit=40.0)
```

This is the same composition pattern as ``af.Model(al.Galaxy, mass=...)``
elsewhere in the workspace — only the construction step is replaced by
loading from CSVs.
"""
galaxy_models = al.galaxy_af_models_from_csv_tables(
    mass_table,
    light_table,
    point_table,
)

# Mutate selected params on the main-lens mass profiles into priors (sigma is
# Lenstool's fiducial v_disp in km/s; radii in arcsec).
for name in ("lens_0", "lens_1"):
    galaxy_models[name].mass.sigma = af.UniformPrior(lower_limit=50.0, upper_limit=600.0)
    galaxy_models[name].mass.r_core = af.UniformPrior(lower_limit=1.0, upper_limit=15.0)
    galaxy_models[name].mass.r_cut = af.UniformPrior(lower_limit=5.0, upper_limit=40.0)

# Host halo: free mass_at_200, keep centre + redshifts fixed.
galaxy_models["host_halo"].dark.mass_at_200 = af.LogUniformPrior(
    lower_limit=10**14.5, upper_limit=10**16.0
)

model = af.Collection(galaxies=af.Collection(**galaxy_models))

print("=" * 64)
print("af.Collection built from CSVs (modelling-ready):")
print("=" * 64)
print(model.info)


"""
__Scaling Galaxies CSV__

The scaling-tier CSV format predates the named-galaxy model CSV API and
keeps its narrow 3-column schema: ``y, x, luminosity, redshift?``. The
scaling tier is implicitly one profile class per member, so naming each
member and emitting an ``attr_name`` column would be more overhead than
signal — every row uses the same ``dPIEMassSph`` mass profile with
parameters derived from the reference-anchored scaling relation's shared
``sigma_ref`` normalization (see ``modeling.py``).

``al.galaxy_table_to_csv`` and ``al.galaxy_table_from_csv`` are the
schema-specific writers/readers. The simulator emits 10 scaling members
into ``dataset/cluster/simple/scaling_galaxies.csv``; we write an
illustrative 3-member version here.
"""
scaling_centres = [(5.5, -6.5), (-7.5, 3.0), (12.0, -5.0)]
scaling_luminosities = [0.40, 0.32, 0.25]

scaling_csv = output_path / "scaling_galaxies.csv"

al.galaxy_table_to_csv(
    centres=scaling_centres,
    luminosities=scaling_luminosities,
    file_path=scaling_csv,
)

scaling_table = al.galaxy_table_from_csv(scaling_csv)

print("=" * 64)
print("scaling_galaxies.csv (legacy schema):")
print("=" * 64)
print(f"  centres={[tuple(c) for c in scaling_table.centres.in_list]}")
print(f"  luminosities={scaling_table.luminosities}")
print(f"  redshifts={scaling_table.redshifts}")
print()


"""
__Wrap Up__

The four CSVs above are the canonical input format for cluster lens
modelling in PyAutoLens. Where they live in the workflow:

 - ``dataset/cluster/<dataset_name>/point_datasets.csv`` — produced by
   ``simulator.py``; consumed by ``modeling.py`` and ``start_here.py``
   via ``al.list_from_csv``.
 - ``dataset/cluster/<dataset_name>/mass.csv`` + ``light.csv`` +
   ``point.csv`` — produced by ``simulator.py`` (truth values); consumed
   by ``modeling.py`` and ``start_here.py`` via
   ``al.galaxy_models_from_csv`` + ``al.galaxy_af_models_from_csv_tables``.
 - ``dataset/cluster/<dataset_name>/scaling_galaxies.csv`` — produced by
   ``simulator.py``; consumed by ``modeling.py`` and ``start_here.py``
   via ``al.galaxy_table_from_csv``.

__Lenstool-Parameterized Rows__

The default dPIE classes ARE Lenstool's native parameterization, so an elliptical ``dPIEMass``
row's columns are the ``.par``-file keywords verbatim::

    galaxy,attr_name,profile_class,y,x,ellipticity,angle_pos,sigma,r_core,r_cut,redshift_object,redshift_source,H0,Om0,redshift
    O1,mass,dPIEMass,1.479,-2.997,0.678,8.971,987.34,18.96,283.54,0.39,11.76,70.0,0.3,0.39

``sigma`` is Lenstool's fiducial ``v_disp`` (sigma_LT), radii are in arcsec, and the run's own
cosmology travels as the flat ``H0`` / ``Om0`` columns. ``scripts/cluster/lenstool/`` builds its
entire 149-component published model this way — the ``.par`` file becomes one canonical CSV. Note
the multi-plane convention: ``redshift_source`` must be the tracer's *final* (highest) source
plane. (The internal ``(ra, rs, b0)`` parameterization remains available for CSV rows via the
non-standard ``dPIEMassB0`` / ``dPIEMassB0Sph`` classes.)

Light-profile CSVs (``light.csv``) support the linear / operated variants with qualified class
names (``linear.Sersic``, ``operated.Gaussian``); plain names resolve to the standard profiles.

__Member Catalogues With Properties__

``scaling_galaxies.csv`` / ``al.galaxy_table_from_csv`` accept any extra per-galaxy columns
beyond ``y, x, luminosity[, redshift]`` — numeric columns (``ellipticity``, ``angle_pos``,
``mag``) load as float lists in ``GalaxyTable.properties``, strings (names, notes) as string
lists. Nothing is silently dropped, and two loud guards protect the model CSVs: a typo'd
parameter column raises (instead of silently leaving the profile at its default), as does a
duplicate ``(galaxy, attr_name)`` row pair.

To start modelling your own cluster:

 1. Edit (or generate from a light-only fit) the model CSVs and
    ``scaling_galaxies.csv`` for your cluster's main + scaling tiers.
 2. Edit ``point_datasets.csv`` with your measured multiple-image
    positions per source.
 3. Drop the CSVs into ``dataset/cluster/<your_name>/``.
 4. Set ``dataset_name = "<your_name>"`` in ``modeling.py`` and run.

For a deeper view of what ``simulator.py`` and ``modeling.py`` do with
these CSVs end-to-end, follow the next files in this folder.
"""

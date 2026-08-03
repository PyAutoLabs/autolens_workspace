"""
Modeling Features (Point Source): Scaling Relation
=================================================

Ties a population of foreground companions to the lens's own Einstein radius by a Faber-Jackson relation, so the whole
tier costs **zero free parameters**:

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

`einstein_radius_anchor` is the lens's `einstein_radius`, which the model already fits.

__This Is Where The Relation Matters Most__

Point-source data is information-poor. A quadruply imaged source gives 8 positional data points; adding fluxes brings
it to 12. That is the entire budget.

Model five companions' Einstein radii individually and you spend 5 of those 12 points on nuisance parameters. Tie them
and you spend none. The relation is not a convenience here — it is the difference between a model this dataset can
constrain and one it cannot. Every other regime can afford to be relaxed about this; point sources cannot.

And the companions cannot be ignored either. Re-solving the simulated system with the tier removed moves the four
images by **182, 398, 1596 and 1633 mas**, against 5 mas astrometry. A multiple image position is the *solution* of
the lens equation, not a linear readout of the deflection field, so a 0.2" deflection near the ring slides an image far
further than 0.2". Omit the tier and the fit will distort the lens's mass distribution trying to absorb it.

__Mass Only, And Where The Luminosities Come From__

A `PointDataset` is positions and fluxes — not an image. There is no companion light in it to blend with anything,
nothing to mask, nothing to noise-scale. This example is therefore mass-only, and the only question it answers is
whether the companions' mass is in the model.

It follows that neither the centres nor the luminosities can come from the point data. Both are measured from the
**accompanying imaging** the positions were extracted from; the simulator writes that imaging to `data.fits` and it is
loaded and plotted below so you can see the companions the numbers refer to. There is no `slam.py` in this folder for
the same reason — with no light in the data there is no light stage to measure anything in.

__Prerequisites__

 - `autolens_workspace/scripts/point_source/modeling.py` — the canonical point-source modeling workflow, including
   the `PointSolver` and name pairing.
 - `autolens_workspace/scripts/point_source/features/extra_galaxies/modeling.py` — companions modelled individually,
   and the fuller discussion of the information budget.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the fuller walkthrough of the relation
   itself.

__Untruncated Profiles__

The tier uses **untruncated** `IsothermalSph`. Truncation encodes tidal stripping by a host halo's potential and a
galaxy-scale lens has no host halo; the truncated `dPIEMass` form belongs to the group- and cluster-scale workflows.

__Contents__

- **Dataset:** The point dataset and its accompanying imaging.
- **Point Solver:** Locating the multiple images of a mass model.
- **Centres:** From the accompanying imaging, not the point data.
- **Luminosities:** Likewise external, and why.
- **Main Lens & Source:** The anchor and the `PointFlux` source.
- **Scaling Tier:** Einstein radii tied to the anchor.
- **Model:** Two top-level collections.
- **Zero Free Parameters:** Proof by parameter count, against the 12-point budget.
- **Search / Analysis / Fit / Result.**
- **CSV Interface.**
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "point_source" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/point_source/features/scaling_relation/simulator.py",
        ],
        check=True,
    )

dataset = al.from_json(file_path=dataset_path / "point_dataset.json")

print("Point Dataset Info:")
print(dataset.info)

aplt.subplot_point_dataset(dataset=dataset)

"""
The accompanying imaging. For most point-source examples this is optional context; here it is doing real work — it is
where the companions' centres *and* luminosities were measured, and plotting it is the quickest way to confirm the
numbers loaded below land on actual galaxies.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)

aplt.plot_array(array=data, title="Accompanying Imaging (companions visible)")

"""
__Point Solver__

Determines the multiple images of the mass model by ray tracing progressively smaller triangles from the image plane to
the source plane. A full description is in `point_source/modeling.py`.

No special settings are needed for a scaling tier: the solver operates on whatever tracer the analysis builds, and the
tier's galaxies are simply more galaxies in it.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

"""
__Centres__

From the accompanying imaging. A `PointDataset` holds positions and fluxes and nothing else — it contains no
information about where a faint companion sits.

The tutorial `autolens_workspace/*/imaging/data_preparation/examples/optional/extra_galaxies_centres.py` shows how to
mark centres on an image and write the JSON.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

print(f"Scaling galaxies centres: {scaling_galaxies_centres}")

"""
__Luminosities__

Explicit Python lists — the simplest interface, and the one worth reading first. The CSV alternative is at the end.

**These are measurements, and they cannot come from the data being fitted.** They are integrated from the companions'
light in the accompanying imaging. In the CCD-imaging version of this feature a prior light-only fit (or `slam.py`)
supplies them; that route does not exist here, because there is no light in a `PointDataset` to fit.

Only ratios to the anchor enter the relation, so units are irrelevant — a magnitude catalogue converts via
`L / L_ref = 10 ** (0.4 * (m_ref - m))`.
"""
luminosity_anchor = 14.5055

scaling_galaxies_luminosities = [0.5116, 0.3848, 0.2716, 0.1811, 0.1268]

assert len(scaling_galaxies_luminosities) == len(list(scaling_galaxies_centres))

"""
__Main Lens & Source__

The lens is mass only — its light is in the accompanying imaging but plays no part in a point-source fit. Its
`einstein_radius` is what the tier hangs off.

The source is a `PointFlux` rather than a `Point` because this dataset includes fluxes; the `point_0` name pairs it to
the dataset of the same name, as described in `point_source/modeling.py`.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.Isothermal))

source = af.Model(al.Galaxy, redshift=1.0, point_0=af.Model(al.ps.PointFlux))

"""
__Scaling Tier__

The relation. `lens.mass.einstein_radius` is the model's own free parameter, so multiplying it by each companion's
luminosity ratio produces a derived quantity rather than a new one.

Mass only, centres fixed to the imaging-measured positions — so each member contributes exactly zero free parameters.
Fixing the centres is not merely stylistic here: with 12 data points, five free centres would be 10 more parameters
and the degeneracy between a free-centre perturber and the lens's own mass is not something a handful of image
positions can break.

The exponent is fixed at the Faber-Jackson value of 0.5 (`einstein_radius ~ sigma^2`, `sigma ~ L^0.25`).
"""
scaling_exponent = 0.5

scaling_galaxies_list = []

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = (
        lens.mass.einstein_radius * (luminosity / luminosity_anchor) ** scaling_exponent
    )

    scaling_galaxies_list.append(af.Model(al.Galaxy, redshift=0.5, mass=mass))

scaling_galaxies = af.Collection(scaling_galaxies_list)

"""
__Model__

Two top-level collections. `scaling_galaxies` is a first-class collection alongside `galaxies`: `AnalysisPoint`
appends it to the tracer it builds from each model instance, so the tier contributes to the ray-tracing the
`PointSolver` performs with no further wiring.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Zero Free Parameters__

Worth checking rather than believing — and worth comparing against the data budget, which is the part that matters in
this regime.
"""
scaling_galaxies_free_list = []

for centre in scaling_galaxies_centres:
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    scaling_galaxies_free_list.append(af.Model(al.Galaxy, redshift=0.5, mass=mass))

model_free = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    scaling_galaxies=af.Collection(scaling_galaxies_free_list),
)

n_data = 2 * len(dataset.positions) + len(dataset.fluxes)

print(f"\nData points (2 per position + 1 per flux): {n_data}")
print(f"Scaling galaxies in the tier:              {len(scaling_galaxies_list)}")
print(f"Free parameters, tier tied:                {model.prior_count}")
print(f"Free parameters, tier freed:               {model_free.prior_count}")
print(f"Parameters saved by the relation:          {model_free.prior_count - model.prior_count}")

assert model_free.prior_count - model.prior_count == len(scaling_galaxies_list)

"""
Read those four numbers together, because they make the argument better than any prose can: **12** data points,
**8** free parameters with the tier tied, **13** with it freed.

Freeing the tier gives the model more free parameters than the data has points. That is not a fit with weak
constraints, it is an under-determined system — the posterior would be shaped by the priors rather than the data. The
relation is what keeps this model on the right side of that line, and it does so while still letting all five
companions perturb the image positions by the hundreds of milliarcseconds they physically do.

Add a sixth companion and the tied count stays at 8.

__Search / Analysis / Fit__
"""
search = af.Nautilus(
    path_prefix=Path("point_source") / "features",
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    # `PointFlux` carries a free source centre (needed here because fluxes are fitted), so the
    # solved-centre default fit cannot be used — we pass the free-centre all-to-all chi-squared.
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

`result.info` lists both collections separately. The tier's members have no `einstein_radius` entry of their own —
each is reported as a derived function of the lens's.
"""
print(result.info)

"""
__CSV Interface__

For larger populations, `al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV (with optional `redshift`) and
returns a `GalaxyTable` with `.centres`, `.luminosities` and `.redshifts`:

    main_lens_table = al.galaxy_table_from_csv(file_path=dataset_path / "main_lens_galaxies.csv")
    luminosity_anchor = main_lens_table.luminosities[0]

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

For this regime the CSV is the truer representation of the workflow: centres and luminosities are two columns of one
photometric measurement on the accompanying imaging, so they belong in one file that cannot fall out of order.
"""
scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")

print(f"\nTier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

The relation turns N free Einstein radii into zero by expressing them as a fixed function of a parameter the model
already has. For point sources that saving is decisive, because the data budget is 12 numbers.

Where to go next:

 - `fit.py` and `likelihood_function.py` in this folder — the composition without a search, and where the tier enters
   the point-source chi-squared.
 - `point_source/features/extra_galaxies/modeling.py` — companions given individual freedom, and when that is
   affordable.
 - `imaging/features/scaling_relation` — the CCD-imaging version, which models companion light and measures its own
   luminosities in `slam.py`.
 - `group/features/scaling_relation` and `cluster/modeling.py` — the reference-magnitude normalisation, and the
   truncated `dPIEMass` profiles appropriate once a host halo exists. Cluster-scale analyses fit many point-source
   families at once and lean on exactly this kind of relation.
"""

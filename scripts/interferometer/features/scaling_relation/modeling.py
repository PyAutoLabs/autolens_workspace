"""
Modeling Features (Interferometer): Scaling Relation
===================================================

Ties a population of foreground companions to the main lens's own Einstein radius by a Faber-Jackson relation, so
the whole tier costs **zero free parameters**:

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

`einstein_radius_anchor` is the main lens's `einstein_radius`, which the model already fits.

__Mass Only, And Where The Luminosities Come From__

Foreground galaxy light is rarely detected at the wavelengths an interferometer probes, so this example — like every
interferometer `extra_galaxies` example in the workspace — models companion **mass only**. There is no companion
light in the model and none in the data.

That settles the luminosity question in a way the CCD-imaging version cannot: the luminosities **cannot** be measured
from this dataset. There is no foreground light here to fit. They come from ancillary optical or near-infrared
imaging of the same field — the same imaging that gave you the companions' centres, since the interferometer data
cannot give you those either.

This is the one substantive difference from `imaging/features/scaling_relation`, whose `slam.py` measures its own
luminosities from a light stage. `slam.py` in *this* folder cannot: an interferometer SLaM pipeline has no `light_lp`
stage at all (see `interferometer/features/extra_galaxies/slam.py`). It threads externally-measured luminosities
through the pipeline instead.

__Prerequisites__

 - `autolens_workspace/scripts/interferometer/modeling.py` — the canonical interferometer modeling workflow.
 - `autolens_workspace/scripts/interferometer/features/extra_galaxies/modeling.py` — companions modelled
   individually, the tier above this one.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the fuller walkthrough of the
   relation itself, including the bounded tier and the light-modelling case.

__Untruncated Profiles__

The tier uses **untruncated** `IsothermalSph`. Truncation encodes tidal stripping by a host halo's potential and a
galaxy-scale lens has no host halo; the truncated `dPIEMass` form belongs to the group- and cluster-scale workflows
(`group/features/group_halo`, `cluster/modeling.py`). `group/features/scaling_relation` also uses a different
normalisation — a free `einstein_radius_ref` at a fixed reference magnitude (Lenstool's `mag0`) — which costs one
parameter but does not assume a single anchoring galaxy.

__Contents__

- **Real Space Mask / Dataset:** Standard interferometer set up (auto-simulating if absent).
- **Centres:** The tier's centres, from ancillary imaging.
- **Luminosities:** The measured luminosities, and why they are necessarily external here.
- **Main Lens & Source:** The anchor and the source.
- **Scaling Tier:** Einstein radii tied to the anchor.
- **Model:** Two top-level collections.
- **Zero Free Parameters:** Proof by parameter count.
- **Search / Analysis / Fit / Result.**
- **CSV Interface:** Loading centres and luminosities from a CSV instead.
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Real Space Mask__

Defines the real-space grid the lens model is evaluated on before being Fourier transformed to the uv-plane.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "interferometer" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/features/scaling_relation/simulator.py",
        ],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Centres__

The tier's centres, and the main lens's. Note where these come from: **not** the interferometer data. A visibility
dataset does not tell you where a faint foreground galaxy sits; the centres are measured from the ancillary imaging
of the same field.

The data-preparation tutorial
`autolens_workspace/*/imaging/data_preparation/examples/optional/extra_galaxies_centres.py` shows how to mark them
on an image and write the JSON.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

"""
__Luminosities__

Given here as explicit Python lists — the simplest interface, and the one worth reading first. The CSV alternative
is at the end of this script, and for this regime it is arguably the more realistic of the two, because the numbers
come from a separate photometric catalogue rather than from anything the pipeline computes.

**These luminosities must be measured, and they cannot be measured here.** In the imaging version of this feature a
prior light-only fit supplies them; that route does not exist for an interferometer, because the foreground light is
not in the data. Use ancillary optical/NIR photometry of the same field.

Only ratios to the anchor enter the relation, so units are irrelevant — a magnitude catalogue converts via
`L / L_ref = 10 ** (0.4 * (m_ref - m))`.
"""
luminosity_anchor = 31.0962

scaling_galaxies_luminosities = [1.4939, 1.0865, 0.7696, 0.4980, 0.2716]

assert len(scaling_galaxies_luminosities) == len(list(scaling_galaxies_centres))

"""
__Main Lens & Source__

Mass only for the lens, as in every interferometer example. Its `einstein_radius` is what the tier hangs off.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.SersicCore))

"""
__Scaling Tier__

The relation. `lens.mass.einstein_radius` is the model's own free parameter, so multiplying it by each companion's
luminosity ratio produces a derived quantity rather than a new one.

Mass only, and centres fixed to the ancillary-imaging positions — so each member contributes exactly zero free
parameters.

The exponent is fixed at the Faber-Jackson value of 0.5 (`einstein_radius ~ sigma^2`, `sigma ~ L^0.25`), which avoids
a normalisation-slope degeneracy.
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

Two top-level collections. `scaling_galaxies` is a first-class collection alongside `galaxies`: the analysis appends
it to the tracer's galaxy list and the aggregator restores it when results are loaded back.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Zero Free Parameters__

Worth checking rather than believing: the same model with every member's `einstein_radius` freed instead of tied.
"""
scaling_galaxies_free_list = []

for centre in scaling_galaxies_centres:
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    scaling_galaxies_free_list.append(af.Model(al.Galaxy, redshift=0.5, mass=mass))

model_free = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    scaling_galaxies=af.Collection(scaling_galaxies_free_list),
)

print(f"\nScaling galaxies in the tier:      {len(scaling_galaxies_list)}")
print(f"Free parameters, tier tied:        {model.prior_count}")
print(f"Free parameters, tier freed:       {model_free.prior_count}")
print(f"Parameters saved by the relation:  {model_free.prior_count - model.prior_count}")

assert model_free.prior_count - model.prior_count == len(scaling_galaxies_list)

"""
__Search / Analysis / Fit__
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "features",
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisInterferometer(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

`result.info` lists both collections separately. The tier's members have no `einstein_radius` entry of their own —
each is reported as a derived function of the lens's, which is what a tied parameter looks like in the output.
"""
print(result.info)

aplt.subplot_fit_interferometer(fit=result.max_log_likelihood_fit)

"""
__CSV Interface__

For larger populations, `al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV (with optional `redshift`) and
returns a `GalaxyTable` with `.centres`, `.luminosities` and `.redshifts`:

    main_lens_table = al.galaxy_table_from_csv(file_path=dataset_path / "main_lens_galaxies.csv")
    luminosity_anchor = main_lens_table.luminosities[0]

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

Keeping centres and luminosities in one file means they cannot fall out of order, which the two-list interface above
cannot enforce for you. For this regime it is also the honest representation of the workflow: both quantities come
from the same external photometric catalogue, so they belong in the same file.
"""
scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")

print(f"\nTier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

The tier turns N free Einstein radii into zero by expressing them as a fixed function of a parameter the model
already has. Adding more rows to `scaling_galaxies_centres.json` does not grow the model.

Where to go next:

 - `slam.py` in this folder — the same relation inside a SLaM pipeline, and why that pipeline still cannot measure
   the luminosities.
 - `fit.py` and `likelihood_function.py` in this folder — the composition without a search, and where the tier
   enters the uv-plane likelihood.
 - `interferometer/features/extra_galaxies/modeling.py` — companions given individual freedom.
 - `imaging/features/scaling_relation` — the CCD-imaging version, which models companion light and measures its own
   luminosities.
 - `group/features/scaling_relation` and `cluster/modeling.py` — the reference-magnitude normalisation and the
   truncated `dPIEMass` profiles appropriate once a host halo exists.
"""

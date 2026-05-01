"""
Simulator: Cluster
==================

This script simulates an example strong lens on the 'cluster' scale: a small cluster consisting of 2 main
lens galaxies (a brightest cluster galaxy + a single satellite), a single host dark matter halo not tied to
any individual galaxy, and 2 multiply-imaged background source galaxies sitting at *different* redshifts
(``z = 1.0`` and ``z = 2.0``) — making this a genuine multi-plane lens.

This is a deliberately small cluster — real clusters can have tens or hundreds of member galaxies and
several background sources. Building the example at this scale lets us exercise every piece of the cluster
workflow (scaling, CSV I/O, visualization, modeling) end-to-end before scaling up to the full problem. A
follow-up example will populate this same simulator with many more cluster members on a luminosity-mass
scaling relation, so 2 main galaxies is the deliberate floor for this baseline.

Modeling at cluster scale almost always uses the *point source* API: rather than fitting the extended arc
light of a lensed source, we fit only the image-plane positions of the brightest pixels of each multiple
image. This script simulates that point-source data alongside CCD imaging — the imaging is used to
*measure* the point positions in real datasets and to visually confirm the lens configuration.

__Contents__

**Multi-Plane Setup:** Why the two sources sit at different redshifts and what that buys the example.
**Main Lens Galaxies vs Host Halo vs Source Galaxies:** Galaxies are organized into three categories.
**Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a name.
**Imaging and Visualization Grids:** Define the high-res rendering grid and a coarse viz grid.
**Galaxy Centres:** Define the centres of the main lens galaxies and sources; used for over-sampling and JSON output.
**Over Sampling:** Adaptive over-sampling grid for accurate light profile evaluation near galaxy centres.
**Main Lens Galaxies:** The 2 cluster member galaxies — each has a `SersicSph` light profile and a `dPIEMassSph` mass.
**Host Dark Matter Halo:** A standalone `NFWMCRLudlowSph` halo with `mass_at_200 = 10^15.3` at z=0.5.
**Source Galaxies:** The 2 multi-plane background sources, each a `SersicCore` light + a `Point` model.
**Ray Tracing:** Combine all galaxies into a single `Tracer` capable of multi-plane ray tracing.
**JAX JIT:** Register the tracer's underlying classes as JAX pytrees and compile the point solver.
**Point Solver:** Solve for image-plane multiple-image positions of each source.
**Point Datasets:** Collect per-source image positions (with noise) into `PointDataset` objects, one per source.
**Combined CSV:** Write *all* datasets to a single CSV so a user can hand-edit positions and noise in a spreadsheet.
**Manual CSV Editing:** Instructions for editing the combined CSV by hand, which is the preferred cluster workflow.
**Tracer JSON:** Save the true `Tracer` for future inspection.
**Centre JSON Files:** Save the main lens and source centres as JSON.
**Imaging:** Simulate CCD imaging of the cluster (used to measure positions in real datasets and for visualization).
**Visualize:** Plot the point-source dataset, tracer, and imaging.

__Multi-Plane Setup__

The two background sources sit at distinct redshifts: source 0 at ``z = 1.0`` and source 1 at ``z = 2.0``.
A real cluster lenses many sources at many different redshifts simultaneously; restricting an example to
a single source plane is a galaxy-scale approximation that hides multi-plane ray-tracing entirely. By
choosing two distinct redshifts here we get a concrete multi-plane testbed with the smallest possible
configuration — the `Tracer` ray-traces through *both* source planes when solving for the image positions
of the further source, exercising the multi-plane code path.

The host halo's ``redshift_source`` parameter is anchored to the *furthest* source (``z = 2.0``) so its
``NFWMCRLudlow`` concentration is set against the deepest light cone in the system. The halo mass
``10^15.3 M_sun`` is large enough that *both* sources end up multiply-imaged.

__Main Lens Galaxies vs Host Halo vs Source Galaxies__

- `main_lens_galaxies`: The 2 cluster member galaxies that dominate the light and contribute to the mass
  through individual `dPIEMassSph` profiles. Each is modeled individually with light and mass profiles in
  modeling scripts.

- `host_halo_galaxy`: A standalone `Galaxy` holding the cluster's `NFWMCRLudlowSph` dark matter halo. It
  is not tied to any individual member galaxy — the halo is a separate mass component sitting "on top of"
  the members.

- `source_galaxies`: The 2 background sources at *different* redshifts. Each carries both a `SersicCore`
  light profile (for visualization of the lensed arcs) and a `Point` model component (used during
  point-source modeling).

Centres of each category are saved to separate JSON files so the modeling scripts can load them directly.

__dPIE Mass Profile__

The cluster member galaxies use the dual Pseudo-Isothermal Elliptical (dPIE) mass profile introduced in
Eliasdottir 2007 (https://arxiv.org/abs/0710.5636), the de facto standard for cluster strong lens modeling.
In spherical form (`dPIEMassSph`), its parameters are:

 - `ra` (arcsec): the core radius, below which the density profile flattens (kept small, ~0.05–0.1" at z=0.5).
 - `rs` (arcsec): the truncation radius, above which the density falls as R^-4 (kept ~10–30" for cluster members).
 - `b0` (arcsec): the mass normalization, roughly setting the galaxy-scale Einstein radius.

These per-galaxy values are hand-tuned below for a physically plausible small cluster. In real analyses a
scaling relation is commonly used to derive `ra`, `rs`, and `b0` from each galaxy's luminosity — that
variant is shown in a separate example.

__NFWMCRLudlow Host Halo__

The host dark matter halo uses `NFWMCRLudlowSph`, which parameterises an NFW profile by the physical mass
within r_200 (`mass_at_200`) and the lens and source redshifts. Internally the concentration-mass
relation of Ludlow et al. (2016) sets the concentration, which together with the cosmology determines
``kappa_s`` and ``scale_radius``. ``mass_at_200 = 10^15.3`` (~2e15 M_sun) is chosen so the combined
halo + member lensing produces genuinely multiply-imaged sources within the field — lighter halos
(``10^14.5``) would only weakly lens these source positions and give a single image each, which is not
useful as a modeling testbed.

__JAX JIT__

Solving the lens equation for the image-plane positions of a point source is iterative and numerically
expensive — at cluster scale (many lens galaxies, multi-plane ray tracing) it dominates the simulator's
runtime by an order of magnitude. We accelerate it with JAX, jitting the ``PointSolver.solve`` call so
the triangle-refinement loop compiles to a single fused kernel.

The pattern mirrors the canonical example in
``autolens_workspace_developer/jax_profiling/point_source/image_plane.py``: every concrete class reachable
from the `Tracer` (i.e. `Galaxy`, light/mass profiles, `Point`) is registered with ``jax.tree_util`` so
that JAX can flatten and unflatten a `Tracer` argument. PyAutoFit's
``autofit.jax.register_model(model)`` walks an ``af.Model`` and registers each ``cls`` it sees;
PyAutoArray's ``register_instance_pytree(Tracer, ...)`` registers the `Tracer` container itself.

The result is that ``jax.jit(solver.solve)`` accepts the concrete `Tracer` directly, traces it once,
caches the compiled triangle-refinement kernel, and reuses it for the second source — turning what was
~5 minutes of Python-loop overhead into a few seconds of compiled JAX execution.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

from autofit.jax import register_model as _register_model_pytrees
from autoarray.abstract_ndarray import register_instance_pytree
from autolens.lens.tracer import Tracer

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive
name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/cluster/simple/data.fits`.
 - The point datasets will be written to `/autolens_workspace/dataset/cluster/simple/point_datasets.csv`.
"""
dataset_type = "cluster"
dataset_name = "simple"

dataset_path = Path("dataset") / dataset_type / dataset_name

"""
__Redshifts__

All main lens galaxies and the host dark matter halo sit at the same lens redshift ``z = 0.5``. The two
sources sit at *different* redshifts (``z = 1.0`` and ``z = 2.0``); see the ``__Multi-Plane Setup__``
section in the module docstring for the rationale.
"""
redshift_lens = 0.5
source_redshifts = [1.0, 2.0]

"""
__Galaxy Centres__

Define the centres of the main lens galaxies, the host halo, and the sources. The host halo is anchored
at the cluster centre (the origin); the two cluster members are placed at the centre and a single
satellite location offset to the upper-right. Source centres are chosen so that both sources land in the
strongly-lensed region for the given halo + member configuration, producing genuine multiple images.
"""
main_lens_centres = [
    (0.0, 0.0),  # BCG at cluster centre
    (10.0, 8.0),  # satellite member
]

host_halo_centre = (0.0, 0.0)

source_centres = [
    (0.3, 0.5),
    (-0.8, 1.2),
]

"""
__Imaging and Visualization Grids__

Two grids are used for image rendering and one for visualization plotting:

 - ``imaging_grid``: a high-resolution (1000x1000 @ 0.1"/px) grid with adaptive over-sampling around each
   cluster member. This is the grid passed to ``SimulatorImaging.via_tracer_from`` and gives an accurate
   simulated CCD image.
 - ``viz_grid``: a coarse (200x200 @ 0.5"/px), un-over-sampled grid passed only to the visualization
   plotters at the end of the script. Visualization plots are illustrative — they don't need the same
   resolution or sub-sampling as the rendered data, and using the imaging grid for them dominated the
   simulator's runtime in earlier versions of this script.

Both grids span the same 100"x100" field — the typical Einstein radius of a ``10^15`` M_sun halo is
~20–30" and the member galaxies span ~30" across, so the field has to be large to capture the multiple
images and extended arc light. The PointSolver builds *its own* internal grid for triangle root-finding
(see the ``__Point Solver__`` section below); that grid is independent of these rendering grids by design.
"""
imaging_grid = al.Grid2D.uniform(
    shape_native=(1000, 1000),
    pixel_scales=0.1,
)

viz_grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.5)

"""
__Over Sampling__

Over sampling evaluates light profiles on a higher-resolution sub-grid in bright central regions, trading
compute for accuracy. For cluster lenses we over-sample around the centre of every cluster member so each
galaxy's Sersic profile is rendered accurately.

The source galaxies use a cored `SersicCore` profile so that lensed arcs can be evaluated without
explicit source-plane over-sampling.
"""
imaging_over_sample = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=imaging_grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres,
)

imaging_grid = imaging_grid.apply_over_sampling(over_sample_size=imaging_over_sample)

"""
__Main Lens Galaxies__

The 2 cluster member galaxies. Each is given a `SersicSph` light profile (used only for visualization —
the imaging data is not used in point-source modeling) and a `dPIEMassSph` mass profile with hand-tuned
parameters representative of cluster members: a larger central BCG and one smaller satellite galaxy.
"""
main_lens_dpie_params = [
    # (ra,  rs,   b0)  per galaxy — arcsec
    (8.0, 20.0, 3.0),  # BCG — strongest
    (5.0, 12.0, 1.2),  # satellite
]

main_lens_sersic_params = [
    # (intensity, effective_radius, sersic_index)
    (1.5, 3.0, 4.0),  # BCG — bright and extended
    (0.8, 1.5, 3.5),  # satellite
]

main_lens_galaxies = []
for centre, (ra, rs, b0), (intensity, effective_radius, sersic_index) in zip(
    main_lens_centres, main_lens_dpie_params, main_lens_sersic_params
):
    bulge = al.lp.SersicSph(
        centre=centre,
        intensity=intensity,
        effective_radius=effective_radius,
        sersic_index=sersic_index,
    )
    mass = al.mp.dPIEMassSph(centre=centre, ra=ra, rs=rs, b0=b0)
    main_lens_galaxies.append(al.Galaxy(redshift=redshift_lens, bulge=bulge, mass=mass))

"""
__Host Dark Matter Halo__

A standalone galaxy holding the cluster's NFW dark matter halo. It has no light profile — it sits in the
tracer solely to contribute mass. `NFWMCRLudlowSph` is parameterised by the physical halo mass within
r_200 and the redshifts; the concentration is set by the Ludlow et al. (2016) concentration-mass relation.
The ``redshift_source`` argument is anchored to the *furthest* source (``z = 2.0``) so the concentration
is computed against the deepest light cone in the multi-plane system.
"""
host_halo = al.mp.NFWMCRLudlowSph(
    centre=host_halo_centre,
    mass_at_200=10**15.3,
    redshift_object=redshift_lens,
    redshift_source=max(source_redshifts),
)

host_halo_galaxy = al.Galaxy(redshift=redshift_lens, dark=host_halo)

"""
__Source Galaxies__

The 2 background sources at *different* redshifts. Each carries a `SersicCore` light profile (used only
for visual confirmation of the lensed arcs — the cored profile changes gradually in the centre so explicit
source-plane over-sampling is unnecessary) and a `Point` model component whose multiple-image positions
we solve for and use as the modeling data.

Each source's redshift is taken from ``source_redshifts``, so source 0 sits at ``z = 1.0`` and source 1
at ``z = 2.0``. The `Tracer` ray-traces multi-plane through both planes automatically.
"""
source_galaxies = []
for i, (centre, src_z) in enumerate(zip(source_centres, source_redshifts)):
    bulge = al.lp.SersicCore(
        centre=centre,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0 + 30.0 * i),
        intensity=2.0,
        effective_radius=0.3,
        sersic_index=1.0,
    )
    point = al.ps.Point(centre=centre)
    source_galaxies.append(
        al.Galaxy(redshift=src_z, bulge=bulge, **{f"point_{i}": point})
    )

"""
__Ray Tracing__

Combine main lens galaxies, the host halo galaxy, and the source galaxies into a single tracer that
produces the simulated image. With sources at distinct redshifts, the tracer automatically handles
multi-plane ray tracing.
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + [host_halo_galaxy] + source_galaxies)

"""
__JAX JIT__

Register every concrete class reachable from the `Tracer` as a JAX pytree node so the tracer can be
passed across a ``jax.jit`` boundary. We use both registration paths:

 - ``register_model(...)`` (PyAutoFit) walks an ``af.Model`` mirror of the tracer and registers each
   ``Galaxy`` / profile class, populating the constructor-arg classifier needed for round-tripping.
 - ``register_instance_pytree(Tracer, no_flatten=("cosmology",))`` (PyAutoArray) registers the `Tracer`
   container itself via ``__dict__`` flattening, holding `cosmology` static across the JIT boundary.

The mirror model uses the same concrete values as the tracer above. It is built solely so that
``register_model`` has something to walk — the simulator does not need an `Analysis` or non-linear search.
"""
_lens_models = [
    af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        bulge=af.Model(
            al.lp.SersicSph,
            centre=g.bulge.centre,
            intensity=g.bulge.intensity,
            effective_radius=g.bulge.effective_radius,
            sersic_index=g.bulge.sersic_index,
        ),
        mass=af.Model(
            al.mp.dPIEMassSph,
            centre=g.mass.centre,
            ra=g.mass.ra,
            rs=g.mass.rs,
            b0=g.mass.b0,
        ),
    )
    for g in main_lens_galaxies
]

_halo_model = af.Model(
    al.Galaxy,
    redshift=redshift_lens,
    dark=af.Model(
        al.mp.NFWMCRLudlowSph,
        centre=host_halo_centre,
        mass_at_200=host_halo.mass_at_200,
        redshift_object=redshift_lens,
        redshift_source=max(source_redshifts),
    ),
)

_source_models = [
    af.Model(
        al.Galaxy,
        redshift=src_z,
        bulge=af.Model(
            al.lp.SersicCore,
            centre=src_centre,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0 + 30.0 * i),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        ),
        **{
            f"point_{i}": af.Model(al.ps.Point, centre=src_centre),
        },
    )
    for i, (src_centre, src_z) in enumerate(zip(source_centres, source_redshifts))
]

_registration_model = af.Collection(
    galaxies=af.Collection(*(_lens_models + [_halo_model] + _source_models))
)

_register_model_pytrees(_registration_model)
register_instance_pytree(Tracer, no_flatten=("cosmology",))

"""
__Point Solver__

For each source's `Point` component we solve for the (y, x) coordinates in the image plane that map to
the source-plane centre — these are the multiple images. The `PointSolver` ray-traces triangles from the
image plane back to the source plane, iteratively refining until the requested precision is reached.

The solver uses its own higher-resolution starting grid because the imaging grid above is tuned for image
rendering, not triangle root-finding. ``magnification_threshold=0.1`` discards heavily demagnified central
images, which are usually undetectable in real data.

The solve call is wrapped in ``jax.jit``: each invocation passes ``xp=jnp`` and
``remove_infinities=False`` so the solver returns a static-shape array (padded with ``inf`` sentinels
where no image was found), which JAX can JIT cleanly. The infinities are stripped outside the JIT after
the call returns.
"""
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(800, 800), pixel_scales=0.1),
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)


@jax.jit
def jitted_solve(tracer, source_plane_coordinate):
    return solver.solve(
        tracer=tracer,
        source_plane_coordinate=source_plane_coordinate,
        xp=jnp,
        remove_infinities=False,
    )


positions_list = []
for i, src_centre in enumerate(source_centres):
    coord = jnp.asarray(src_centre)
    raw = np.asarray(jitted_solve(tracer, coord).array)
    finite = ~(np.isinf(raw).any(axis=1) | np.isnan(raw).any(axis=1))
    positions_list.append(al.Grid2DIrregular(raw[finite]))

"""
__Point Datasets__

One `PointDataset` per source. The `name` (e.g. `point_0`, `point_1`) pairs each dataset with the
matching `Point` component in the lens model during modeling.

`redshift` is populated on each `PointDataset` so that the per-source redshifts round-trip through the
combined CSV below. This is the piece that makes the CSV self-describing for cluster modeling — position,
noise, and redshift live in a single spreadsheet.
"""
dataset_list = []
for i, positions in enumerate(positions_list):
    dataset = al.PointDataset(
        name=f"point_{i}",
        positions=positions,
        positions_noise_map=imaging_grid.pixel_scale,
        redshift=source_redshifts[i],
    )
    dataset_list.append(dataset)

"""
Output one .json file per dataset (exact round-trip; this is the canonical modeling input).
"""
for i, dataset in enumerate(dataset_list):
    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / f"point_dataset_{i}.json",
    )

"""
__Combined CSV__

For cluster-scale workflows with tens or hundreds of sources, a single CSV with one row per observed
image — grouped by ``name`` — is far easier to edit in a spreadsheet than many per-source JSON files.
``al.output_to_csv`` writes every dataset into one file. The `redshift` column is emitted automatically
because each dataset has its redshift set above.
"""
al.output_to_csv(
    datasets=dataset_list,
    file_path=dataset_path / "point_datasets.csv",
)

"""
__Manual CSV Editing__

The combined CSV is the preferred cluster input: it is human-readable, editable in Excel / LibreOffice /
any text editor, and round-trips cleanly back into `list_from_csv`. The expected format is one row per
observed multiple image with the following columns:

 - `name`    — the source identifier (e.g. `point_0`). All rows sharing a `name` belong to the same source.
 - `y`, `x`  — the image-plane position of the multiple image, in arc-seconds.
 - `positions_noise` — the positional uncertainty (typically the pixel scale of the imaging used to
   measure it).
 - `redshift` — the source redshift. Every row for a given `name` must share the *same* redshift
   (validated on load; `list_from_csv` raises if a group's rows disagree). Leave the cell blank if the
   redshift is unknown; blank is tolerated as long as *all* rows in a group are blank.

Optional columns `flux`, `flux_noise`, `time_delay`, `time_delay_noise` are also round-tripped — populate
them when the observation provides those measurements, leave them blank otherwise.

To build a cluster dataset by hand, simulate or manually collect one set of images per source, then edit
the CSV directly: add or remove rows, adjust positions or noises, and save. Reload the dataset in a
modeling script with ``al.list_from_csv(file_path=dataset_path / "point_datasets.csv")``.
"""

"""
__Tracer JSON__

Save the `Tracer` so the true light profiles, mass profiles and galaxies can be inspected after the fact.
This can be loaded via `tracer = al.from_json(file_path)`.
"""
al.output_to_json(
    obj=tracer,
    file_path=dataset_path / "tracer.json",
)

"""
__Centre JSON Files__

Save the main lens galaxy centres, the host halo centre, and the source centres as JSON so the modeling
scripts can load them directly (e.g. to fix positions, define priors, or set up scaling relations).
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=dataset_path / "main_lens_centres.json",
)
al.output_to_json(
    obj=al.Grid2DIrregular([host_halo_centre]),
    file_path=dataset_path / "host_halo_centre.json",
)
al.output_to_json(
    obj=al.Grid2DIrregular(source_centres),
    file_path=dataset_path / "source_centres.json",
)

"""
__Imaging__

Strong lens clusters typically come with imaging data — used to *measure* the point positions and to
visually confirm the lens configuration. Although modeling here is point-source only, we output CCD
imaging so the dataset looks like a realistic cluster observation.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=imaging_grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=imaging_grid)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output .png plots of the simulated dataset, the tracer, and the per-source point datasets.

These use the default galaxy-scale plotters and are known to be suboptimal for cluster-scale systems —
arcs span a much larger field, per-source images benefit from distinct colours, and multi-source overlays
are useful. A follow-up prompt (`PyAutoPrompt/cluster/1_visualization.md`) addresses these
visualization requirements.
"""
for i, pd in enumerate(dataset_list):
    aplt.subplot_point_dataset(
        dataset=pd, output_path=dataset_path, output_format="png"
    )

aplt.subplot_imaging_dataset(dataset=dataset)
aplt.subplot_tracer(
    tracer=tracer, grid=viz_grid, output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies_images(
    tracer=tracer, grid=viz_grid, output_path=dataset_path, output_format="png"
)

"""
Finished.
"""

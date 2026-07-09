"""
Lenstool Users: Cluster Modeling (SMACS J0723)
==============================================

**If you model galaxy clusters with Lenstool, this script is for you.** It repeats a published
Lenstool analysis — Mahler et al. 2023's model of SMACS J0723, the first JWST cluster — in
**PyAutoLens**, in three steps:

 1. **Reconstruct** the published best-fit model, reading the 149 dPIE potentials of ``best.par``
    directly into PyAutoLens profiles (no re-fitting, no conversion by hand).
 2. **Verify** that PyAutoLens reproduces the published lensing: the observed multiple images
    trace back to compact source-plane groups, and (optionally) forward-solving the lens equation
    recovers the observed image positions at the published RMS (0.32").
 3. **Refit** the cluster from scratch, composing the same model Lenstool optimized — same free
    parameters, same priors as ``input.par``, same positional likelihood — so the posterior can be
    compared with Table 3 of the paper number for number.

Run ``data.py`` first; it downloads the public model files and writes the CSVs read here.

__The .par → PyAutoLens dictionary__

    Lenstool                            PyAutoLens
    ------------------------------------------------------------------------------
    ``potentiel`` / profil 81           ``al.mp.dPIEMass`` (elliptical dPIE)
    x_centre, y_centre [arcsec]         ``centre=(y, x)`` — same relative frame (data.py)
    ellipticite = (a²-b²)/(a²+b²)       ``ellipticity`` (converted internally to (1-q)/(1+q))
    angle_pos [deg]                     ``angle_pos``
    core_radius / cut_radius [arcsec]   ``r_core`` / ``r_cut``
    v_disp [km/s]                       ``sigma`` — Lenstool's *fiducial* sigma_LT, see below
    z_lens                              ``redshift_object``
    ``potfile`` (member scaling)        shared priors + derived per-member parameters (below)
    ``arcs.dat``                        ``point_datasets.csv`` → ``al.PointDataset`` list
    sigposArcsec                        ``positions_noise`` column (same chi-squared)
    source-plane optimization           ``al.FitPositionsSource`` (this script)
    image-plane optimization            ``al.FitPositionsImagePair*`` (heavier; see guides)
    ``best.par``                        the max-likelihood instance of the PyAutoLens fit

__Three traps to know about__

 - **sigma is the fiducial velocity dispersion sigma_LT**, not the physical central velocity
   dispersion: sigma_0 = sqrt(3/2) * sigma_LT (Eliasdottir et al. 2007, App. A). PyAutoLens's
   ``from_lenstool`` / ``dPIEMassLenstool`` take sigma_LT — quote ``v_disp`` values unchanged.
   Feeding a *measured* stellar velocity dispersion here overestimates the mass by 50%.
 - **The x axis points West** in Lenstool's relative frame (data.py verifies this against the
   data). All coordinates in this script live in that frame, so numbers compare directly to the
   ``.par`` file; flip x when overlaying on a WCS-aligned image.
 - **Radii in .par files come in arcsec and kpc variants** (``core_radius`` vs
   ``core_radius_kpc``). PyAutoLens profiles work in arcsec; the kpc → arcsec conversion uses the
   Lenstool run's own cosmology (below), 1" = 5.313 kpc at z = 0.39.

__Cosmology__

Mahler et al. use flat LCDM with H0 = 70, Omega_m = 0.3 — the Lenstool-typical choice, *not*
PyAutoLens's Planck15 default. It is passed explicitly everywhere below; forgetting it shifts
D_LS/D_S and hence every mass normalization at the percent level.

__Contents__

- **Load Data:** the CSVs written by ``data.py``.
- **The Published Model, Reconstructed:** 149 ``from_lenstool`` profiles + 21 point sources.
- **Verification I — source-plane compactness:** observed images trace to tight source groups.
- **Verification II — image-plane RMS (optional):** forward-solve vs the published 0.32".
- **Critical Curves (optional):** the per-source-plane critical curves over the HST image.
- **The Refit:** the same free parameters and priors as ``input.par``, fit with Nautilus.

__Runtime__

Reconstruction + Verification I run in a couple of minutes. Verification II forward-solves the
lens equation for 21 sources through a 149-profile multi-plane tracer (one-off JAX compile of a
few minutes; enable with ``RUN_IMAGE_PLANE = True``), and the critical-curve figure costs ~10+
minutes per plane in numpy (``MAKE_FIGURES = True``). The full
refit is a production job (tens of hours on a workstation; Mahler et al. report k = 46 free
parameters — ours is 30 mass + 42 source-position parameters with the 16 photometric source
redshifts held at their published values). ``PYAUTO_TEST_MODE=2`` exercises the full composition
without running the sampler.
"""

from pathlib import Path

import numpy as np

import autofit as af
import autolens as al
import autolens.plot as aplt

RUN_IMAGE_PLANE = False

"""
__Cosmology__
"""
cosmology = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

Z_LENS = 0.39

kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=Z_LENS)
print(f'1" = {kpc_per_arcsec:.3f} kpc at z = {Z_LENS} (Lenstool-run cosmology)')

"""
__Load Data__

``data.py`` wrote one CSV per Lenstool input (see its docstring for the full conventions):

 - ``point_datasets.csv`` — the 60 multiple images of 21 sources (``arcs.dat``), with Lenstool's
   ``sigposArcsec`` as the position noise and per-system redshifts (spectroscopic where they
   exist, the model-optimized values of ``best.par`` otherwise).
 - ``mass.csv`` — the complete optimized mass model in the **canonical named-galaxy CSV**
   (the same ``al.galaxy_models_from_csv`` format every cluster script uses): 149 rows of
   ``profile_class = dPIEMassLenstool``, one per ``potential`` section of ``best.par``, whose
   columns are the ``.par`` keywords verbatim. The five individually-optimized halos are named
   O1 (cluster-scale), O2 (BCG), O3 ("dNW"), O4 ("ICL"), O5 ("eCM"); the 144 scaling members
   are ``member_<n>``.
 - ``members.csv`` — the member *catalogue* (``galcat.cat``) in the ``al.galaxy_table_from_csv``
   schema: centres + luminosities plus ``ellipticity`` / ``angle_pos`` / ``mag`` property
   columns (the refit derives member masses from these + the shared scaling parameters, as
   ``potfile`` does).
"""
dataset_path = Path("dataset") / "cluster" / "smacs0723"

dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
print(f"{len(dataset_list)} point-source systems loaded.")


mass_table = al.galaxy_models_from_csv(dataset_path / "mass.csv", family="mass")
members_table = al.galaxy_table_from_csv(file_path=dataset_path / "members.csv")

print(
    f"mass.csv: {len(mass_table.rows)} dPIEMassLenstool rows | "
    f"members.csv: {len(members_table.luminosities)} catalogue members"
)

"""
__The Published Model, Reconstructed__

Every ``potential`` section of ``best.par`` becomes one ``al.mp.dPIEMass`` via ``from_lenstool``
— the arguments are the ``.par`` keywords, verbatim. This is the whole point of the Lenstool-native
API: nothing is transcribed by hand, and the sqrt(3/2) sigma convention, the ellipticity
conversion and the D_LS/D_S normalization are handled (and unit-tested) inside PyAutoLens.

One subtlety that *will* bite anyone porting a model by hand: Lenstool normalizes each halo's
deflection per source plane at ray-trace time, whereas a PyAutoLens profile is normalized once, at
construction, against a chosen source redshift. In a multi-plane ``Tracer`` the cosmological
scaling factors beta are computed **relative to the tracer's final plane** — so every profile must
be normalized to the *highest* source redshift in the system (here the z = 11.76 candidate).
Normalize to any other plane and every deflection is off by a constant D-ratio (for z = 2 it is
~15%, which smears each source group by ~2" — we found out the honest way). Both codes then reduce
to the same beta = D_LS/D_S ratios in the same cosmology.
"""
Z_REF_SOURCE = max(float(dataset.redshift) for dataset in dataset_list)

# One call: every mass.csv row instantiates its dPIEMassLenstool with the .par values —
# the redshift_source (final-plane) normalization and the run's H0/Om0 travel inside the
# CSV columns, so nothing here needs to remember them.
lens_galaxies = list(al.galaxies_from_csv_tables(mass_table).values())

print(f"Reconstructed {len(lens_galaxies)} dPIE mass components from mass.csv.")

source_galaxies = [
    al.Galaxy(
        redshift=dataset.redshift, **{dataset.name: al.ps.Point(centre=(0.0, 0.0))}
    )
    for dataset in dataset_list
]

tracer = al.Tracer(galaxies=lens_galaxies + source_galaxies, cosmology=cosmology)
print(
    f"Tracer has {len(tracer.planes)} planes (1 lens + {len(tracer.planes)-1} source)."
)

"""
__Verification I — Source-Plane Compactness__

Lenstool's model was optimized so the observed images of each system meet at a single source
position. Ray-tracing the observed image positions through the reconstructed tracer must therefore
produce *compact* per-system source-plane groups — this is the fast, solver-free check that the
model transferred correctly (a wrong sigma convention, ellipticity definition or cosmology blows
these numbers up immediately).

For each system, every observed image is traced back to that system's source plane and the RMS
scatter about the group's centroid is reported. With the published image-plane RMS of 0.32" and typical
magnifications of a few, per-system values of ~0.1" (we measure a median of 0.07", every system
below 0.29") confirm the model transferred exactly.
"""
scatters = []
for dataset in dataset_list:
    positions = al.Grid2DIrregular(np.atleast_2d(np.asarray(dataset.positions)))
    plane_index = tracer.plane_index_via_redshift_from(redshift=dataset.redshift)
    traced = tracer.traced_grid_2d_list_from(grid=positions)[plane_index]
    traced = np.asarray(traced)
    centroid = traced.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((traced - centroid) ** 2, axis=1))))
    scatters.append(rms)
    print(
        f"  {dataset.name:>9} (z={float(dataset.redshift):6.3f}, "
        f'{len(traced)} images): source-plane rms = {rms:6.4f}"'
    )

print(f'Median source-plane rms: {np.median(scatters):.4f}"')

"""
__Verification II — Image-Plane RMS (optional)__

The definitive check forward-solves the lens equation: from each system's model source position
(the traced centroid above), find every image position the reconstructed mass model predicts, pair
predictions with observations, and measure the image-plane RMS. Mahler et al. report 0.32". This
uses the same JAX-compiled ``PointSolver`` as the cluster simulator; the one-off compile takes
several minutes for a 149-profile tracer, so it is gated behind ``RUN_IMAGE_PLANE``.
"""
if RUN_IMAGE_PLANE:
    import jax

    al.jax.register_tracer_classes(tracer)

    grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.7)
    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.01, use_jax=True
    )

    offsets = []
    for dataset in dataset_list:
        positions = al.Grid2DIrregular(np.atleast_2d(np.asarray(dataset.positions)))
        plane_index = tracer.plane_index_via_redshift_from(redshift=dataset.redshift)
        traced = np.asarray(
            tracer.traced_grid_2d_list_from(grid=positions)[plane_index]
        )
        source_centre = tuple(traced.mean(axis=0))

        predicted = solver.solve(
            tracer=tracer,
            source_plane_coordinate=source_centre,
            plane_redshift=float(dataset.redshift),
        )
        predicted = np.atleast_2d(np.asarray(predicted))

        for obs in np.atleast_2d(np.asarray(dataset.positions)):
            offsets.append(np.min(np.linalg.norm(predicted - obs, axis=1)))
        print(f"  {dataset.name}: {len(predicted)} predicted images")

    rms_image_plane = float(np.sqrt(np.mean(np.array(offsets) ** 2)))
    print(f'Image-plane rms = {rms_image_plane:.3f}" (published: 0.32")')

"""
__Critical Curves (optional)__

The cluster plotters draw each source plane's critical curves over the HST cutout — with 21
source planes we plot a representative subset (the lowest-redshift plane and the famous z = 11.8
candidate's plane). At cluster scale each plane's curves differ visibly; this is the multi-plane
structure a single-plane "the critical curve" plot hides.

Fair warning on cost: a critical-curve evaluation walks the whole grid through all 149 dPIE
profiles several times — ~10+ minutes per figure on a laptop — so it is gated behind
``MAKE_FIGURES`` like the forward-solve. Enable both when producing final figures on real
hardware.
"""
MAKE_FIGURES = False

if MAKE_FIGURES:
    image = al.Array2D.from_fits(
        file_path=dataset_path / "data.fits", pixel_scales=0.06
    )

    # 150x150 @ 1" resolves the cluster-scale curves; refine when producing final figures.
    viz_grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=1.0)

    plane_redshifts = [float(p.redshift) for p in tracer.planes]
    plane_indices = [
        plane_redshifts.index(min(r for r in plane_redshifts if r > Z_LENS)),
        len(plane_redshifts) - 1,
    ]

    aplt.plot_critical_curves(
        tracer,
        grid=viz_grid,
        image=image,
        plane_indices=plane_indices,
        output_path=str(dataset_path),
        output_filename="critical_curves_reconstruction",
        output_format="png",
    )
    print("Critical-curve figure written next to the dataset.")

"""
__The Refit__

Everything above used the published answer. A real analysis *fits*: the model below reproduces the
composition Lenstool optimized (``input.par``), using the Lenstool-parameterized profile
``al.mp.dPIEMassLenstool`` so every free parameter, prior bound and posterior number is in
Lenstool units:

 - **O1, cluster halo**: centre U(-5,5)" (both axes), ellipticity U(0,0.8), angle U(-90,90),
   r_core U(10,150) kpc, sigma U(300,1200) km/s; r_cut fixed at 1500 kpc. [6 free]
 - **O2, BCG**: geometry fixed to the light; r_core U(0.1,10) kpc, r_cut U(10,500) kpc,
   sigma U(100,500). [3 free]
 - **O3 "dNW"**: centre U(best ± 2.6)", ellipticity U(0,0.8), angle U(-90,90), r_core U(0,10) kpc,
   r_cut U(10,300) kpc, sigma U(0,200). [7 free]
 - **O4 "ICL"**: centre U(best ± 10)", ellipticity U(0,0.8), angle U(-90,90), r_core U(0,50) kpc,
   r_cut U(50,1000) kpc, sigma U(0,700). [7 free]
 - **O5 "eCM"**: centre fixed; ellipticity U(0,0.6), angle U(-90,90), r_core U(0,10) kpc,
   r_cut U(10,200) kpc, sigma U(0.1,300). [5 free]
 - **potfile members**: every catalogue member gets a ``dPIEMassLenstool`` with centre, shape and
   angle *fixed to the light* (``galcat.cat``) and its sigma / r_cut derived from two shared free
   parameters exactly as ``potfile`` defines —

       sigma_i = sigma_star * (L_i/L0)^0.25        sigma_star ~ U(50, 500) km/s
       r_cut_i = r_cut_star * (L_i/L0)^0.5         r_cut_star ~ U(1, 100) kpc
       r_core_i = 0.15 kpc (fixed)

   (``vdslope 4`` and ``slope 4`` in ``input.par`` are these fixed exponents; the same
   reference-anchored convention is the default throughout the PyAutoLens cluster workflow.)
   [2 free for all 146 catalogue members]

 - **Sources**: one ``al.ps.Point`` per system with a free centre initialised from the traced
   centroid of its observed images; redshifts fixed (spectroscopic or published model values).
   [42 free]

Total: 30 mass + 42 source-position parameters. Mahler et al.'s k = 46 counts 30 mass + 16 free
photometric redshifts — Lenstool eliminates source positions analytically in its optimization,
PyAutoLens samples them; the guides discuss this difference and the image-plane likelihood that
removes it.
"""
sigma_star = af.UniformPrior(lower_limit=50.0, upper_limit=500.0)
r_cut_star_kpc = af.UniformPrior(lower_limit=1.0, upper_limit=100.0)

R_CORE_MEMBER = 0.15 / kpc_per_arcsec  # potfile corekpc, converted once

member_models = []
for centre, luminosity, ellipticity, angle_pos in zip(
    members_table.centres,
    members_table.luminosities,
    members_table.properties["ellipticity"],
    members_table.properties["angle_pos"],
):
    mass = af.Model(al.mp.dPIEMassLenstool)
    mass.centre = tuple(centre)
    mass.ellipticity = ellipticity
    mass.angle_pos = angle_pos
    mass.r_core = R_CORE_MEMBER
    mass.r_cut = (r_cut_star_kpc / float(kpc_per_arcsec)) * luminosity**0.5
    mass.sigma = sigma_star * luminosity**0.25
    mass.redshift_object = Z_LENS
    mass.redshift_source = Z_REF_SOURCE
    mass.H0 = 70.0
    mass.Om0 = 0.3
    member_models.append(af.Model(al.Galaxy, redshift=Z_LENS, mass=mass))


# The named halos start as af.Models straight from mass.csv — the canonical
# al.galaxy_af_models_from_csv_tables call gives every row's values as fixed
# defaults, and we promote exactly the parameters input.par optimized to priors
# (in Lenstool units). Redshifts and H0/Om0 ride in from the CSV already fixed.
halo_af_models = al.galaxy_af_models_from_csv_tables(mass_table)


def halo_model_from(label, limits):
    galaxy_model = halo_af_models[label]
    mass = galaxy_model.mass
    row = next(r for r in mass_table.rows if r.galaxy == label)
    best_y, best_x = row.params["centre"]

    if "centre" in limits:
        half = limits["centre"]
        mass.centre_0 = af.UniformPrior(
            lower_limit=best_y - half, upper_limit=best_y + half
        )
        mass.centre_1 = af.UniformPrior(
            lower_limit=best_x - half, upper_limit=best_x + half
        )

    if "ellipticity" in limits:
        mass.ellipticity = af.UniformPrior(0.0, limits["ellipticity"])
        mass.angle_pos = af.UniformPrior(-90.0, 90.0)

    lo, hi = limits["r_core_kpc"]
    mass.r_core = af.UniformPrior(
        lo / float(kpc_per_arcsec), hi / float(kpc_per_arcsec)
    )

    if "r_cut_kpc" in limits:
        lo, hi = limits["r_cut_kpc"]
        mass.r_cut = af.UniformPrior(
            lo / float(kpc_per_arcsec), hi / float(kpc_per_arcsec)
        )
    # else: r_cut stays at its CSV value (O1: fixed 1500 kpc, already in arcsec).

    lo, hi = limits["sigma"]
    mass.sigma = af.UniformPrior(lo, hi)
    return galaxy_model


halo_models = [
    halo_model_from(
        "O1", dict(centre=5.0, ellipticity=0.8, r_core_kpc=(10, 150), sigma=(300, 1200))
    ),
    halo_model_from(
        "O2", dict(r_core_kpc=(0.1, 10), r_cut_kpc=(10, 500), sigma=(100, 500))
    ),
    halo_model_from(
        "O3",
        dict(
            centre=2.6,
            ellipticity=0.8,
            r_core_kpc=(0, 10),
            r_cut_kpc=(10, 300),
            sigma=(0, 200),
        ),
    ),
    halo_model_from(
        "O4",
        dict(
            centre=10.0,
            ellipticity=0.8,
            r_core_kpc=(0, 50),
            r_cut_kpc=(50, 1000),
            sigma=(0, 700),
        ),
    ),
    halo_model_from(
        "O5",
        dict(
            ellipticity=0.6, r_core_kpc=(0, 10), r_cut_kpc=(10, 200), sigma=(0.1, 300)
        ),
    ),
]

source_models = []
for dataset in dataset_list:
    positions = np.atleast_2d(np.asarray(dataset.positions))
    plane_index = tracer.plane_index_via_redshift_from(redshift=dataset.redshift)
    traced = np.asarray(
        tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(positions))[plane_index]
    )
    point = af.Model(al.ps.Point)
    point.centre_0 = af.GaussianPrior(mean=float(traced[:, 0].mean()), sigma=2.0)
    point.centre_1 = af.GaussianPrior(mean=float(traced[:, 1].mean()), sigma=2.0)
    source_models.append(
        af.Model(al.Galaxy, redshift=float(dataset.redshift), **{dataset.name: point})
    )

model = af.Collection(
    halos=af.Collection(halo_models),
    members=af.Collection(member_models),
    sources=af.Collection(source_models),
)

print(
    f"Refit model composed: {model.prior_count} free parameters "
    f"(30 mass + {2 * len(dataset_list)} source positions)."
)
print(model.info)

"""
__Search__

Source-plane chi-squared (Lenstool's default likelihood) via ``AnalysisPoint``; the factor-graph
pattern is identical to ``scripts/cluster/modeling.py``. This is a production-scale job — run it
on real hardware, not in a tutorial session. The result's max-likelihood instance is the PyAutoLens
equivalent of ``best.par``; compare it to Table 3 of Mahler et al. 2023 and to the reconstruction
above.
"""
search = af.Nautilus(
    path_prefix=Path("cluster") / "smacs0723",
    name="lenstool_refit",
    unique_tag="mahler2023",
    n_live=400,
    number_of_cores=4,
)

solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=1.4),
    pixel_scale_precision=0.025,
)

analysis_list = [
    al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        fit_positions_cls=al.FitPositionsSource,
        cosmology=cosmology,
    )
    for dataset in dataset_list
]

import os

if os.environ.get("LENSTOOL_EXAMPLE_RUN_FIT"):
    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model, analysis=analysis)
        for analysis in analysis_list
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list)
    result_list = search.fit(
        model=factor_graph.global_prior_model, analysis=factor_graph
    )
    print("Refit complete — compare result_list max-likelihood values with Table 3.")
else:
    print(
        "Refit composition validated (the model.info above is the structural pass); set "
        "LENSTOOL_EXAMPLE_RUN_FIT=1 to execute the production-scale search — a 72-parameter "
        "factor-graph fit is never smoke-mode material."
    )

"""
Finished. The README in this folder is the narrative companion: the full .par ↔ PyAutoLens
dictionary, the conventions verified here, and where to go next (image-plane likelihood, JAX,
extended-source modeling of the arcs — everything Lenstool cannot do is on the other side of
this door).
"""

"""
Modeling Features (Group): The Group Halo — An Explicit Choice
==============================================================

**Whether a group-scale lens model includes a separate dark-matter halo for the group is a modelling
choice, not an assumption.** Some groups genuinely require one; others are adequately described by their
member galaxies alone. This tutorial — the signature example of the group regime — fits the SAME dataset
with both compositions and walks through how to decide between them:

 - **Model A (members-only):** the BGG and member galaxies carry all the mass — tidally truncated dPIE
   subhalos, nothing else.
 - **Model B (members + group halo):** the same members, plus a group-scale dPIE dark-matter halo not
   tied to any galaxy's light.

__When is a group halo scientifically motivated?__

The literature offers three practical tests, which this tutorial exercises:

 1. **The radii the arcs probe.** A halo and its members have different mass profiles at large radius, but
    inside the innermost ~20 kpc the BGG dominates either way (Wang et al. 2022's CSWA 31 analysis finds
    exactly this radius-dependence). If your arcs sit close to the BGG, members-only may fit perfectly; if
    image configurations span large radii — wide-separation arcs the members cannot bend enough — the halo
    earns its place.
 2. **Bayesian model comparison.** Fit both compositions and compare log evidences: the halo is kept when
    the evidence justifies its extra parameters. This tutorial's dataset was simulated WITH a halo, so
    Model B should win; re-simulate with ``include_group_halo = False`` in ``simulator.py`` and A wins.
 3. **External information.** X-ray emission, member dynamics (velocity dispersion of the group), or weak
    lensing can independently establish a massive common halo (the SL2S group analyses combine exactly
    these), settling the question before any strong-lens fitting.

Environment statistics say this choice arises constantly: roughly half of real lens deflectors live in
group or cluster environments, while fewer than 10% have group-scale Einstein radii (AGEL survey) — i.e.
the environment usually contributes convergence and shear without dominating, and a full halo component
is only sometimes warranted.

__The truncation thread__

Note what does NOT change between the two models: the members are tidally truncated dPIE profiles
(vanishing cores, finite ``r_cut``) in both. Truncation is the signature of the group and cluster
regimes — it encodes stripping of the members' outer halos by the shared potential — and it is present
even when the halo itself is left out of the model, because the *physical* group environment exists
either way; the question is only whether the data demands the halo as a separate mass component. (At
galaxy and multi-galaxy scale, with no host environment, profiles are untruncated — see `multi_galaxy/`.)

Lens light is omitted from this dataset to isolate the mass question; the standard group light handling
(MGE per galaxy) is unchanged from `group/modeling.py`.

__Contents__

- **Dataset:** Load (auto-simulating if absent) the group-halo dataset.
- **Model A:** Members-only composition.
- **Model B:** Members + group halo composition.
- **Fits:** Fit both models to the same data.
- **Model Comparison:** Compare the evidences and interpret.
- **Wrap Up:** The decision checklist.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the `group_halo` dataset, auto-simulating it if absent. The truth includes a group halo (see
`simulator.py` — flip `include_group_halo` there to build intuition for the opposite verdict).
"""
dataset_name = "group_halo"
dataset_path = Path("dataset", "group", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/features/group_halo/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

member_centres = al.from_json(file_path=dataset_path / "member_centres.json")

mask_radius = 6.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Shared Setup__

Both models share the member tier: the BGG with free `sigma` and `r_cut` (vanishing core fixed at 0), and
the two fainter members with their truncated dPIE parameters tied to the BGG via the modern scaling
convention (see `cluster/mass_parameterizations.py` for the full conventions guide). Sharing the member
tier between the models means the comparison isolates exactly one question: does the data want a halo?
"""
redshift_lens = 0.5
redshift_source = 1.0

H0 = 67.66
Om0 = 0.30966

# Member luminosity ratios relative to the BGG — matching the simulator truth here
# (in a real analysis these come from photometry).
member_luminosity_ratios = [1.0, 0.25, 0.15]

alpha = 0.25  # Faber-Jackson dispersion exponent
gamma = 0.2  # mass-to-light tilt, universally fixed
beta_cut = 1.0 + gamma - 2.0 * alpha  # 0.7 — the tied truncation exponent


def member_tier_models():
    """The BGG (free sigma + r_cut) and two members tied to it by the scaling relation."""
    bgg_mass = af.Model(al.mp.dPIEMassSph)
    bgg_mass.centre = tuple(member_centres[0])
    bgg_mass.sigma = af.UniformPrior(lower_limit=100.0, upper_limit=500.0)
    bgg_mass.r_core = 0.0  # vanishing core — fixed, never scaled
    bgg_mass.r_cut = af.UniformPrior(lower_limit=2.0, upper_limit=30.0)
    bgg_mass.redshift_object = redshift_lens
    bgg_mass.redshift_source = redshift_source
    bgg_mass.H0 = H0
    bgg_mass.Om0 = Om0

    galaxies = [af.Model(al.Galaxy, redshift=redshift_lens, mass=bgg_mass)]

    for centre, luminosity_ratio in zip(member_centres[1:], member_luminosity_ratios[1:]):
        mass = af.Model(al.mp.dPIEMassSph)
        mass.centre = tuple(centre)
        mass.sigma = bgg_mass.sigma * luminosity_ratio**alpha  # tied to the BGG
        mass.r_core = 0.0
        mass.r_cut = bgg_mass.r_cut * luminosity_ratio**beta_cut  # tied truncation
        mass.redshift_object = redshift_lens
        mass.redshift_source = redshift_source
        mass.H0 = H0
        mass.Om0 = Om0
        galaxies.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

    return galaxies


def source_model():
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )
    return af.Model(al.Galaxy, redshift=redshift_source, bulge=bulge)


"""
__Model A — Members Only__

The BGG + members carry all the mass. Free parameters: the BGG's `sigma` and `r_cut` (the members are
tied to both) plus the source.
"""
members_a = member_tier_models()

model_a = af.Collection(
    galaxies=af.Collection(
        **{f"lens_{i}": g for i, g in enumerate(members_a)},
        source=source_model(),
    )
)

print("Model A (members-only) free parameters:", model_a.prior_count)

"""
__Model B — Members + Group Halo__

The identical member tier, plus an elliptical dPIE group halo: free `ellipticity`, `angle_pos`, `sigma`
and `r_core`, centre near the BGG and truncation fixed large — the Lenstool host-halo convention (swap in
`al.mp.NFW` / a gNFW for the physically preferred alternative; the comparison logic is unchanged).
"""
halo_mass = af.Model(al.mp.dPIEMass)
halo_mass.centre = (0.3, -0.2)  # fixed near the BGG; free it for real data
halo_mass.ellipticity = af.UniformPrior(lower_limit=0.0, upper_limit=0.7)
halo_mass.angle_pos = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)
halo_mass.sigma = af.UniformPrior(lower_limit=300.0, upper_limit=900.0)
halo_mass.r_core = af.UniformPrior(lower_limit=2.0, upper_limit=30.0)
halo_mass.r_cut = 200.0  # unconstrained within the field; fixed large
halo_mass.redshift_object = redshift_lens
halo_mass.redshift_source = redshift_source
halo_mass.H0 = H0
halo_mass.Om0 = Om0

members_b = member_tier_models()

model_b = af.Collection(
    galaxies=af.Collection(
        **{f"lens_{i}": g for i, g in enumerate(members_b)},
        halo=af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass),
        source=source_model(),
    )
)

print("Model B (members + halo) free parameters:", model_b.prior_count)

"""
__Fits__

Fit both models to the same data with the same search settings, so the evidences are comparable.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

search_a = af.Nautilus(
    path_prefix=Path("group") / "features" / "group_halo",
    name="members_only",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_full_update=100000,
)

result_a = search_a.fit(model=model_a, analysis=analysis)

search_b = af.Nautilus(
    path_prefix=Path("group") / "features" / "group_halo",
    name="members_plus_halo",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_full_update=100000,
)

result_b = search_b.fit(model=model_b, analysis=analysis)

"""
__Model Comparison__

Compare the Bayesian log evidences. A commonly used rule of thumb: a difference in log evidence above ~3
("strong" on the Jeffreys scale) justifies the more complex model; below that, prefer the simpler
members-only composition. (Under PYAUTO_TEST_MODE the searches are bypassed and these numbers are
meaningless — run for real to see the comparison work.)
"""
log_evidence_a = result_a.samples.log_evidence
log_evidence_b = result_b.samples.log_evidence

print(f"Model A (members-only)  log evidence: {log_evidence_a}")
print(f"Model B (members+halo)  log evidence: {log_evidence_b}")

if log_evidence_a is not None and log_evidence_b is not None:
    delta = log_evidence_b - log_evidence_a
    print(f"Delta log evidence (B - A): {delta}")
    print(
        "Halo justified (strong evidence)."
        if delta > 3.0
        else "Prefer members-only — the halo's extra parameters are not earned."
    )

aplt.subplot_fit_imaging(fit=result_a.max_log_likelihood_fit)
aplt.subplot_fit_imaging(fit=result_b.max_log_likelihood_fit)

"""
__Wrap Up__

The decision checklist for your own group:

 1. Plot the residuals of a members-only fit first: coherent large-scale residuals around wide arcs are
    the classic halo signature.
 2. Fit both compositions; compare log evidences (this tutorial).
 3. Bring in external information where it exists — X-ray, group-member dynamics, weak lensing.
 4. Whichever wins, keep the members truncated: the group environment tidally strips its members whether
    or not the halo is explicit in the model.

The same machinery (host halo + truncated members + scaling relations) IS the cluster mass model — at
cluster scale the halo stops being optional and the analysis switches to point-source constraints; see
`cluster/start_here.py`.
"""

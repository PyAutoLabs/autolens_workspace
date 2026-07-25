"""
Cluster & Group Mass Parameterizations
======================================

**A plain-language tour of the ways to parameterize and couple the masses in a Lenstool-style cluster
model — the cluster-scale halo, the BCG and other individually-freed galaxies, and the member
("scaling") population.**

There are several parameterizations floating around — velocity dispersion vs the internal ``b0``;
magnitudes vs luminosities; members tied to a scaling relation vs left free; the members normalized to
their own free parameter vs to the BCG's mass. They describe the *same physics*, so it is easy to lose
track of what actually differs. This script builds each option **with the model-composition API and
prints ``model.info``**, so for every model you can see at a glance *exactly which parameters are free
and which are fixed* — the thing that really distinguishes the approaches.

Nothing here is fitted. Each section composes an ``af.Collection`` model and prints it (plus a few
assertions); the point is to *read* the models, not run them.

__The axes of choice__

Every model below is a point in the same space of decisions:

 - **What number carries the mass?**  ``sigma`` (velocity dispersion, km/s — Lenstool's native
   parameter) or the internal ``b0`` (arcsec, the dPIE lens strength). Inter-convertible; which you
   make *free* is the choice.
 - **How are members coupled?**  A Faber-Jackson scaling relation ties the whole member population to a
   shared normalization; individual galaxies are left free (only feasible for a few).
 - **What anchors the relation?**  A reference *magnitude* ``mag0`` (Lenstool's ``potfile`` keyword).
 - **What is the member normalization?**  Its own free parameter (``sigma_ref``), or the BCG's own mass
   (mass anchoring — the member tier then costs zero parameters).
 - **How many relation parameters?**  Just the dispersion normalization, or the truncation
   normalization ``r_cut_ref`` as well.

__Contents__

Each model is a **complete, self-contained** Lenstool-style cluster model — a cluster-scale halo, the
individually-modelled galaxies, and the member relation — so you can copy any one block wholesale
(there is deliberate redundancy between them). All use dPIE / Lenstool parameterizations only; mapping
these onto PyAutoLens-native profiles (isothermals, etc.) is a separate guide.

 - **Model 1 — The Standard Lenstool Model** — members on a velocity-dispersion Faber-Jackson relation
   anchored to ``mag0``; BCG + a freed member modelled individually; a free elliptical halo. No ``b0``.
 - **Model 2 — The Angular (b0) Parameterization** — the same model with ``sigma -> b0`` (redshift-free).
 - **Model 3 — Mass Anchoring** — the members normalized to the BCG's own mass (member tier -> 0 free).
 - **Model 4 — The Two-Parameter Relation** — free the truncation normalization ``r_cut_ref`` too.

Freeing the Faber-Jackson *exponent* is a one-line change (make ``sigma_exponent`` a prior instead of
``0.25``; the truncation exponent then follows automatically from the ``2*alpha + beta = 1 + gamma``
tie — see Model 1) and is noted inline rather than given its own model. How Lenstool handles *unknown
source redshifts* (``z_m_limit``) is about the source, not the mass, and is out of scope here.

__Reading model.info__

For every model we print ``model.info``. A line showing a ``UniformPrior`` (or any prior) is a **free**
parameter the search would sample; a bare number is **fixed**. A member's ``sigma`` shown as a
``MultiplePrior`` is *tied* to a shared free parameter, not free in its own right.
"""

import numpy as np

import autofit as af
import autolens as al
import autogalaxy as ag

"""
__Shared Setup__

Redshifts and cosmology are the same as the ``cluster/simple`` example. The cosmology is the flat
``H0 = 67.66``, ``Om0 = 0.30966`` that the dPIE profile uses internally, so nothing is approximate.
For a real cluster, set these to the measured lens redshift and the run's cosmology.

The photometry below is illustrative. In a real analysis it comes from a member catalogue (one
magnitude per galaxy in the band the scaling relation is calibrated in — e.g. F160W) plus each
galaxy's sky position. ``scaling_*`` are the fainter cluster members that go on the scaling relation;
``individual_*`` are the galaxies modelled on their own — the BCG (first) plus any bright member worth
freeing (e.g. a galaxy near a multiple image). The BCG's magnitude also anchors the relation (mag0).
"""
redshift_lens = 0.5
redshift_source = 2.0

H0 = 67.66
Om0 = 0.30966

# Scaling-relation members (the "potfile" population): (y, x) arc-second position + apparent magnitude.
scaling_centres = [(5.5, -6.5), (-7.5, 3.0), (3.0, 13.0), (12.0, -5.0), (-6.5, 11.0)]
scaling_magnitudes = [19.2, 19.8, 20.4, 21.0, 21.6]

# Individually-modelled galaxies: the BCG (first) plus any bright member we want to free — e.g. a
# galaxy sitting near a multiple image, which the scaling relation cannot describe accurately enough
# locally. In Lenstool these are just separate "potentiel" sections; the BCG is not structurally
# special, only the first of them. The BCG's magnitude also anchors the scaling relation (mag0).
individual_centres = [(0.0, 0.0), (8.5, 5.5)]
individual_magnitudes = [17.8, 18.9]


"""
__Model 1 — The Standard Lenstool Model__

This is the model a Lenstool ``.par`` file describes. It has the three components of a real cluster
model, and this script builds all three:

 - a **cluster-scale dark-matter halo** — an elliptical dPIE, fully free, not tied to any light: the
   dominant, smooth large-scale mass;
 - **individually-modelled galaxies** — the BCG and any bright member worth freeing, each its own free
   dPIE (Lenstool's individual "potentiel" sections; the BCG is simply the first);
 - the **member population** ("potfile") — masses set by a velocity-dispersion Faber-Jackson relation
   anchored to a reference magnitude.

dPIE profiles throughout, composed directly in ``sigma`` (km/s), ``r_core`` and ``r_cut`` (arcsec) —
exactly the numbers a Lenstool results table quotes. No ``b0`` appears.

**A note on the halo profile.** A dPIE for the *cluster-scale* halo is the Lenstool-literature
convention and is what this guide follows — but physically it is an odd choice for a dark-matter halo,
and cosmological simulations motivate an NFW or generalized NFW (gNFW) instead. Swapping the halo for
``al.mp.NFW`` / a gNFW while keeping the dPIE members is a supported, arguably preferable model; the
mapping guide (``mass_parameterizations_pyautolens.py``) shows how Lenstool-style components translate
to PyAutoLens-native profiles.

**The reference magnitude ``mag0``.** The relation is anchored to a reference magnitude (the ``potfile``
keyword ``mag0``); ``sigma_ref`` is the velocity dispersion of a galaxy *at that magnitude*. A member of
magnitude ``m`` enters through its brightness *relative* to the reference,
``L/L_ref = 10 ** (0.4 * (mag0 - m))``, giving clean magnitude forms:

    sigma_i  = sigma_ref  * 10 ** (0.4 * alpha    * (mag0 - m_i))     # alpha    = 0.25 -> 0.1
    r_cut_i  = r_cut_ref  * 10 ** (0.4 * beta_cut * (mag0 - m_i))     # beta_cut = 0.70 -> 0.28
    r_core_i = 0                                                      # vanishing core, NOT scaled

**The modern scaling-relation convention (Bergamini et al. 2019).** The dispersion exponent ``alpha``
(Faber-Jackson, ``sigma ~ L^alpha``) and the truncation exponent ``beta_cut`` (``r_cut ~ L^beta_cut``)
are not independent: the member's total mass scales as ``M ~ sigma^2 r_cut``, so demanding a
mass-to-light tilt ``M/L ~ L^gamma`` enforces

    2 * alpha + beta_cut = 1 + gamma        (valid for r_core << r_cut, which the vanishing core ensures)

with ``gamma = 0.2`` universally fixed in the modern literature. With ``alpha = 0.25`` this gives
``beta_cut = 0.7`` — NOT the ``0.5`` ("constant M/L", ``gamma = 0``) slope of the original Lenstool-era
papers, which is dated. In detailed modelling ``alpha`` itself can be freed (Bergamini et al. 2019
measure ~0.27 from MUSE member kinematics); ``beta_cut`` then follows from the tie automatically.

**Member cores vanish and are never scaled.** The standard convention assigns members (and the BCG) a
vanishing core: ``r_core`` is fixed to zero (or a negligibly small value) and is NOT scaled with
luminosity. PyAutoLens's dPIE handles ``r_core = 0`` analytically — the deflection prefactor is
``b0 * r_cut / (r_cut - r_core)``, with no division by ``r_core`` anywhere — so, unlike some Lenstool /
lenstronomy code paths where a zero core divides by zero, ``r_core = 0`` is numerically safe here.

**The reference truncation is small.** ``r_cut_ref = 5.0`` arcsec — lensing-constrained member
truncations are typically ~5", not the tens of arcseconds sometimes seen in older examples. Freeing and
constraining ``r_cut_ref`` (Model 4) is scientifically interesting in its own right: it is directly
related to the galaxy-galaxy strong-lensing excess in clusters reported by Meneghetti et al. (2020,
Science).

We set ``mag0`` to the BCG's magnitude — a natural anchor. Note this uses only the BCG's *brightness*;
the BCG's *mass* is one of the individually-modelled galaxies and is never coupled to the members.

**Free vs fixed.**

 - Halo: free ``ellipticity``, ``angle_pos``, ``sigma``, ``r_core`` (4). Fixed centre (often freed near
   the BCG) and ``r_cut`` (unconstrained within the field).
 - Individually-modelled galaxies (BCG + freed member): each free ``sigma`` + ``r_cut`` (2 per galaxy);
   fixed centre and ``r_core``.
 - Members: one free parameter for the whole population, ``sigma_ref``. Adding members adds nothing.
 - The exponents, reference radii, ``mag0``, redshifts and cosmology are fixed.

Total: 9 free parameters (halo 4 + 2 individual galaxies x 2 + members 1).
"""

# --- Cluster-scale dark-matter halo: elliptical dPIE, fully free, not tied to light ---
halo_mass = af.Model(al.mp.dPIEMass)
halo_mass.centre = (
    0.0,
    0.0,
)  # [FIXED] often freed near the BCG (set centre_0 / centre_1 to priors)
halo_mass.ellipticity = af.UniformPrior(lower_limit=0.0, upper_limit=0.7)  # [FREE]
halo_mass.angle_pos = af.UniformPrior(
    lower_limit=0.0, upper_limit=180.0
)  # [FREE] degrees
halo_mass.sigma = af.UniformPrior(
    lower_limit=500.0, upper_limit=1500.0
)  # [FREE] cluster-scale dispersion (km/s)
halo_mass.r_core = af.UniformPrior(
    lower_limit=20.0, upper_limit=150.0
)  # [FREE] large halo core (arcsec)
halo_mass.r_cut = (
    1000.0  # [FIXED] truncation unconstrained within the field; fixed large
)
halo_mass.redshift_object = redshift_lens
halo_mass.redshift_source = redshift_source
halo_mass.H0 = H0
halo_mass.Om0 = Om0
cluster_halo = af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass)

# --- Individually-modelled galaxies: the BCG (first) + any freed member, each its own free dPIE ---
individual_sigma_priors = [
    (200.0, 600.0),
    (150.0, 450.0),
]  # per-galaxy free sigma range (km/s)
individual_rcut_priors = [
    (2.0, 200.0),
    (2.0, 150.0),
]  # per-galaxy free r_cut range (arcsec; lower bound admits the ~5" typical of members)

individual_galaxies = []
for centre, (sigma_lo, sigma_hi), (rcut_lo, rcut_hi) in zip(
    individual_centres, individual_sigma_priors, individual_rcut_priors
):
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre  # [FIXED] observed position
    mass.sigma = af.UniformPrior(
        lower_limit=sigma_lo, upper_limit=sigma_hi
    )  # [FREE] this galaxy's own dispersion
    mass.r_core = 0.0  # [FIXED] vanishing core (handled analytically — see intro)
    mass.r_cut = af.UniformPrior(
        lower_limit=rcut_lo, upper_limit=rcut_hi
    )  # [FREE] this galaxy's own truncation
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    individual_galaxies.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

# --- Members: one free sigma_ref; Bergamini+19 tied exponents (2*alpha + beta_cut = 1 + gamma) ---
sigma_ref = af.UniformPrior(
    lower_limit=100.0, upper_limit=400.0
)  # [FREE] km/s (dispersion at mag0)
mag0 = individual_magnitudes[
    0
]  # [FIXED] reference magnitude (Lenstool's mag0) = the BCG's brightness
sigma_exponent = 0.25  # [FIXED] Faber-Jackson alpha (free it via af.UniformPrior — one line; beta_cut then follows)
gamma = 0.2  # [FIXED] mass-to-light tilt M/L ~ L^gamma — universally fixed at 0.2
rcut_exponent = (
    1.0 + gamma - 2.0 * sigma_exponent
)  # [TIED] beta_cut = 0.7 — enforces M ~ sigma^2 r_cut ~ L^(1+gamma) (Bergamini+19)
r_core_ref = 0.0  # [FIXED] vanishing core — never scaled with luminosity
r_cut_ref = 5.0  # [FIXED] arcsec — typical lensing-constrained member truncation

scaling_galaxies = []
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    luminosity_ratio = 10.0 ** (0.4 * (mag0 - magnitude))  # L / L_ref from magnitudes

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre  # [FIXED]
    mass.sigma = (
        sigma_ref * luminosity_ratio**sigma_exponent
    )  # tied to the one free sigma_ref
    mass.r_core = r_core_ref  # [FIXED] not scaled with luminosity
    mass.r_cut = r_cut_ref * luminosity_ratio**rcut_exponent
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    scaling_galaxies.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

model_1 = af.Collection(
    cluster_halo=cluster_halo,
    individual_galaxies=af.Collection(individual_galaxies),
    scaling_galaxies=af.Collection(scaling_galaxies),
)

print("=" * 80)
print("Model 1 — The Standard Lenstool Model")
print("=" * 80)
print(
    f"Total free parameters: {model_1.prior_count}  (halo 4 + 2 individual x 2 + members 1 = 9)"
)
assert model_1.prior_count == 9
print(model_1.info)


"""
__Model 2 — The Angular Parameterization: dPIEMassB0__

**How.** Swap every dPIE for its ``B0`` twin and compose in angular parameters: ``ra`` (core),
``rs`` (truncation) and ``b0`` (lens strength) — all arcsec, with **no redshift and no cosmology**. The
radii map one-to-one (``r_core -> ra``, ``r_cut -> rs``); the mass normalization moves ``sigma``
(km/s) ``-> b0`` (arcsec). The scaling relation moves with it, and the exponent **doubles**:

    Model 1 (sigma):   sigma_i = sigma_ref * (L/L_ref)^0.25
    Model 2 (b0):      b0_i    = b0_ref    * (L/L_ref)^0.50      # because b0 ~ sigma^2

It is the *same* Faber-Jackson relation, written in the quantity the lensing actually constrains. The
halo becomes an elliptical ``dPIEMassB0`` (ellipticity via ``ell_comps``); individual galaxies and
members become ``dPIEMassB0Sph``.

**Why.** The ``sigma`` parameterization needs the redshifts (``b0 = 6*648000*(sigma/c)^2*(D_LS/D_S)``),
so ``sigma`` only means something once ``z_lens`` / ``z_source`` are known. When they are *not*, the
images constrain only the angular ``b0``, and ``sigma`` is degenerate with ``D_LS/D_S``. Fitting ``b0``
directly fits the quantity the data determines, in arcsec you can reason about. Convert back once the
redshifts are measured: ``sigma = c * sqrt(b0 / K)``, ``K = 6*648000*(D_LS/D_S)``. The ``B0`` mass
carries no redshift; the galaxy ``redshift`` is only a ray-tracing plane label.

**A convention warning: sigma is sigma_LT, not the central dispersion.** The ``6`` in the prefactor is
Lenstool's *fiducial* velocity-dispersion convention: ``sigma`` here is Lenstool's ``v_disp``
(``sigma_LT``), related to the physical central dispersion by ``sigma_0 = sqrt(3/2) * sigma_LT`` — the
same ``b0`` written in ``sigma_0`` is ``b0 = 4*648000*(sigma_0/c)^2*(D_LS/D_S)``. The ``sqrt(3/2)`` is
not a physical definition from Elíasdóttir et al. (2007): it is bookkeeping that arises because Lenstool
pairs an E07-style ``b0`` coefficient with the Kassiola & Kovner (1993) / Limousin et al. (2005)
deflection amplitude (a historical mismatch between the two parameterizations; see the contributed
derivation note "On the definitions of b0 and velocity dispersion in Lenstool / dPIE", H. Ding 2026).
PyAutoLens keeps ``sigma_LT`` as the ``dPIEMass`` parameter so fitted posteriors read like Lenstool
results tables — but for physical interpretation (comparison with measured stellar kinematics) convert
to ``sigma_0 = sqrt(3/2) * sigma``; quoting a measured dispersion directly as ``sigma`` overestimates
the mass by 50%.

**Free vs fixed.** Identical structure to Model 1 (9 free): halo free ``ell_comps`` (2) + ``ra`` + ``b0``;
individual galaxies free ``b0`` + ``rs``; members one free ``b0_ref``.
"""

# --- Halo: elliptical dPIEMassB0 (ellipticity via ell_comps) ---
halo_mass = af.Model(al.mp.dPIEMassB0)
halo_mass.centre = (0.0, 0.0)  # [FIXED]
halo_mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=-0.5, upper_limit=0.5
)  # [FREE]
halo_mass.ell_comps.ell_comps_1 = af.UniformPrior(
    lower_limit=-0.5, upper_limit=0.5
)  # [FREE]
halo_mass.ra = af.UniformPrior(
    lower_limit=20.0, upper_limit=150.0
)  # [FREE] core (arcsec)
halo_mass.rs = 1000.0  # [FIXED] truncation, fixed large
halo_mass.b0 = af.UniformPrior(
    lower_limit=5.0, upper_limit=50.0
)  # [FREE] cluster-scale lens strength (arcsec)
cluster_halo = af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass)

# --- Individually-modelled galaxies: free b0 + free rs (angular, redshift-free) ---
individual_b0_priors = [
    (0.0, 5.0),
    (0.0, 3.0),
]  # arcsec, angular lens-strength range per galaxy
individual_rs_priors = [
    (2.0, 200.0),
    (2.0, 150.0),
]  # arcsec, truncation range per galaxy

individual_galaxies_b0 = []
for centre, (b0_lo, b0_hi), (rs_lo, rs_hi) in zip(
    individual_centres, individual_b0_priors, individual_rs_priors
):
    mass = af.Model(al.mp.dPIEMassB0Sph)
    mass.centre = centre  # [FIXED]
    mass.b0 = af.UniformPrior(
        lower_limit=b0_lo, upper_limit=b0_hi
    )  # [FREE] angular lens strength (arcsec)
    mass.ra = 0.0  # [FIXED] = Model 1's vanishing core
    mass.rs = af.UniformPrior(
        lower_limit=rs_lo, upper_limit=rs_hi
    )  # [FREE] truncation (arcsec)
    individual_galaxies_b0.append(
        af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
    )

# --- Members: one free b0_ref; exponents follow Model 1's tied relation ---
b0_ref = af.UniformPrior(
    lower_limit=0.0, upper_limit=1.0
)  # [FREE] arcsec (lens strength at mag0)
b0_exponent = 0.5  # [FIXED] b0 ~ L^(2*alpha) = L^0.5 (b0 ~ sigma^2 doubles Model 1's alpha = 0.25)
rs_exponent = 0.7  # [FIXED] = Model 1's beta_cut = 1 + gamma - 2*alpha (gamma = 0.2)
mag0 = individual_magnitudes[0]  # [FIXED]
ra_ref = 0.0  # [FIXED] = Model 1's vanishing core — never scaled
rs_ref = 5.0  # [FIXED] = Model 1's r_cut_ref

scaling_galaxies_b0 = []
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    luminosity_ratio = 10.0 ** (0.4 * (mag0 - magnitude))

    mass = af.Model(al.mp.dPIEMassB0Sph)
    mass.centre = centre  # [FIXED]
    mass.b0 = b0_ref * luminosity_ratio**b0_exponent  # tied to the one free b0_ref
    mass.ra = ra_ref  # [FIXED] not scaled with luminosity
    mass.rs = rs_ref * luminosity_ratio**rs_exponent
    scaling_galaxies_b0.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

model_2 = af.Collection(
    cluster_halo=cluster_halo,
    individual_galaxies=af.Collection(individual_galaxies_b0),
    scaling_galaxies=af.Collection(scaling_galaxies_b0),
)

print("\n" + "=" * 80)
print("Model 2 — The Angular (b0) Parameterization")
print("=" * 80)
print(f"Total free parameters: {model_2.prior_count}  (same structure as Model 1)")
assert model_2.prior_count == 9

"""
__Model 2 — Numerical Checks__

Model 2 is Model 1 re-expressed, so it must reproduce Model 1's physics once the redshifts are
supplied. We relate the two with the distance factor ``K`` (redshifts + cosmology used *only* for the
comparison, never inside the b0 model), and check three things:

 (a) the Faber-Jackson relation is consistent — a member's ``b0`` from the ``b0``-relation equals
     ``K (sigma/c)^2`` with ``sigma`` from the ``sigma``-relation (the exponent doubling is exact);
 (b) the deflections are identical, profile-for-profile;
 (c) a fitted ``b0_ref`` converts back to the Model 1 ``sigma_ref`` exactly.
"""
C_KM_S = 299792.458
cosmology = ag.cosmo.FlatLambdaCDM(
    H0=H0, Om0=Om0
)  # matches dPIEMass's internal cosmology
d_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_source)
d_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
    redshift_0=redshift_lens, redshift_1=redshift_source
)
K = 6.0 * 648000.0 * (d_ls / d_s)  # b0 = K * (sigma / c)^2

grid = al.Grid2DIrregular([[3.0, 4.0], [-8.0, 2.0], [10.0, -6.0]])

# A worked value: the b0_ref a redshift-free fit would return for sigma_ref = 250 km/s.
sigma_ref_value = 250.0
b0_ref_value = K * (sigma_ref_value / C_KM_S) ** 2

# (a) Faber-Jackson consistency.
worst_relation = 0.0
for magnitude in scaling_magnitudes:
    ratio = 10.0 ** (0.4 * (mag0 - magnitude))
    worst_relation = max(
        worst_relation,
        abs(
            b0_ref_value * ratio**b0_exponent
            - K * (sigma_ref_value * ratio**0.25 / C_KM_S) ** 2
        ),
    )
print(
    f"(a) member b0  [b0-relation vs sigma-relation]:  max diff = {worst_relation:.2e}"
)
assert worst_relation < 1e-12

# (b) Deflections identical: dPIEMassB0Sph(b0) == dPIEMassSph(sigma) for each member.
worst_defl = 0.0
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    ratio = 10.0 ** (0.4 * (mag0 - magnitude))
    mass_b0 = al.mp.dPIEMassB0Sph(
        centre=centre,
        ra=ra_ref,
        rs=rs_ref * ratio**rs_exponent,
        b0=b0_ref_value * ratio**b0_exponent,
    )
    mass_sigma = al.mp.dPIEMassSph(
        centre=centre,
        sigma=sigma_ref_value * ratio**0.25,
        r_core=ra_ref,
        r_cut=rs_ref * ratio**rs_exponent,
        redshift_object=redshift_lens,
        redshift_source=redshift_source,
        H0=H0,
        Om0=Om0,
    )
    worst_defl = max(
        worst_defl,
        np.max(
            np.abs(
                np.asarray(mass_b0.deflections_yx_2d_from(grid=grid))
                - np.asarray(mass_sigma.deflections_yx_2d_from(grid=grid))
            )
        ),
    )
print(f"(b) member deflections [b0 vs sigma param]:      max diff = {worst_defl:.2e}")
assert worst_defl < 1e-12

# (c) Round-trip.
sigma_ref_recovered = C_KM_S * (b0_ref_value / K) ** 0.5
print(
    f"(c) b0_ref = {b0_ref_value:.5f} arcsec  ->  sigma_ref = {sigma_ref_recovered:.2f} km/s  (input {sigma_ref_value})"
)
assert np.isclose(sigma_ref_recovered, sigma_ref_value, rtol=1e-9)
print("Model 2 checks passed.")


"""
__Model 3 — Mass Anchoring: the Members Normalized to the BCG__

**How.** Take Model 1 and *delete* the members' free normalization ``sigma_ref``. In its place the
members are normalized to the BCG's OWN velocity dispersion — the same free ``sigma`` that sets the
BCG's mass:

    Model 1:   sigma_i = sigma_ref  * (L/L_BCG)^0.25     # sigma_ref FREE, independent of the BCG
    Model 3:   sigma_i = sigma_BCG  * (L/L_BCG)^0.25     # sigma_BCG = the BCG's own free mass

Because ``sigma_BCG`` is already free (it *is* the BCG's mass), the member population now adds **zero**
free parameters. This is the referee's "anchor alpha to the BCG Einstein radius" — ``sigma_BCG`` and
the BCG Einstein radius are the same quantity up to the fixed conversion.

**Why.** The most economical model: one free normalization (the BCG's, tightly pinned by the central
images) describes everyone, with an interpretable prior. You save a parameter.

**The cost — and why this is NOT the default (Model 1).** It *assumes the BCG lies on the member
relation*; real BCGs frequently deviate (merger products at the bottom of the potential well), which is
why high-precision models free the BCG separately (Model 1). And it hard-couples every member's mass to
the BCG's — move the fitted BCG mass and all members move in fixed proportion. A real, used variant
(simpler models, the referee's convention), but a deliberate coupling choice, not the standard.

**Free vs fixed.** Halo as Model 1 (4). BCG: free ``sigma`` (its own mass *and* the member anchor) +
``r_cut``. Other freed galaxy: its own free ``sigma`` + ``r_cut``. Members: **zero** free. Total: 8.
"""

# --- Halo: identical to Model 1 (the halo is unaffected by how the members are anchored) ---
halo_mass = af.Model(al.mp.dPIEMass)
halo_mass.centre = (0.0, 0.0)  # [FIXED]
halo_mass.ellipticity = af.UniformPrior(lower_limit=0.0, upper_limit=0.7)  # [FREE]
halo_mass.angle_pos = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)  # [FREE]
halo_mass.sigma = af.UniformPrior(lower_limit=500.0, upper_limit=1500.0)  # [FREE]
halo_mass.r_core = af.UniformPrior(lower_limit=20.0, upper_limit=150.0)  # [FREE]
halo_mass.r_cut = 1000.0  # [FIXED]
halo_mass.redshift_object = redshift_lens
halo_mass.redshift_source = redshift_source
halo_mass.H0 = H0
halo_mass.Om0 = Om0
cluster_halo = af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass)

# --- The BCG: the ANCHOR. Its free sigma sets its own mass AND normalizes every member. ---
bcg_mass = af.Model(al.mp.dPIEMassSph)
bcg_mass.centre = individual_centres[0]  # [FIXED]
bcg_mass.sigma = af.UniformPrior(
    lower_limit=200.0, upper_limit=600.0
)  # [FREE] BCG mass + member anchor
bcg_mass.r_core = 0.0  # [FIXED] vanishing core
bcg_mass.r_cut = af.UniformPrior(
    lower_limit=2.0, upper_limit=200.0
)  # [FREE] BCG's own truncation
bcg_mass.redshift_object = redshift_lens
bcg_mass.redshift_source = redshift_source
bcg_mass.H0 = H0
bcg_mass.Om0 = Om0
bcg = af.Model(al.Galaxy, redshift=redshift_lens, mass=bcg_mass)

# --- Other individually-freed galaxies (still independent — e.g. a member on an arc) ---
individual_sigma_priors = [(200.0, 600.0), (150.0, 450.0)]
individual_rcut_priors = [(2.0, 200.0), (2.0, 150.0)]

other_individual_galaxies = []
for centre, (sigma_lo, sigma_hi), (rcut_lo, rcut_hi) in zip(
    individual_centres[1:], individual_sigma_priors[1:], individual_rcut_priors[1:]
):
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = af.UniformPrior(
        lower_limit=sigma_lo, upper_limit=sigma_hi
    )  # [FREE] independent of the BCG
    mass.r_core = 0.0  # [FIXED] vanishing core
    mass.r_cut = af.UniformPrior(lower_limit=rcut_lo, upper_limit=rcut_hi)  # [FREE]
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    other_individual_galaxies.append(
        af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
    )

# --- Members: normalized to the BCG's OWN sigma. ZERO new free parameters. ---
mag0 = individual_magnitudes[0]  # [FIXED] = BCG magnitude
r_core_ref = 0.0  # [FIXED] vanishing core — never scaled
r_cut_ref = 5.0  # [FIXED]
rcut_exponent = 0.7  # [TIED] = 1 + gamma - 2*alpha (alpha = 0.25, gamma = 0.2)

mass_anchored_members = []
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    luminosity_ratio = 10.0 ** (0.4 * (mag0 - magnitude))  # L / L_BCG

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = (
        bcg_mass.sigma * luminosity_ratio**0.25
    )  # tied to the BCG's free sigma -> NO new parameter
    mass.r_core = r_core_ref  # [FIXED] not scaled with luminosity
    mass.r_cut = r_cut_ref * luminosity_ratio**rcut_exponent
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    mass_anchored_members.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

model_3 = af.Collection(
    cluster_halo=cluster_halo,
    individual_galaxies=af.Collection([bcg] + other_individual_galaxies),
    scaling_galaxies=af.Collection(mass_anchored_members),
)

print("\n" + "=" * 80)
print("Model 3 — Mass Anchoring (members normalized to the BCG)")
print("=" * 80)
print(
    f"Total free parameters: {model_3.prior_count}   (Model 1 was 9; the member tier now adds 0)"
)
assert model_3.prior_count == 8

"""
__Model 3 — Numerical Checks__

 (a) the member collection adds ZERO free parameters (all 8 belong to the halo + individual galaxies);
 (b) at any instance, each member's sigma is exactly ``sigma_BCG * (L/L_BCG)^0.25`` — a fixed fraction
     of the BCG, so a change in the fitted BCG mass rescales them all together.
"""
# (a) The member tier adds zero free parameters.
without_members = af.Collection(
    cluster_halo=cluster_halo,
    individual_galaxies=af.Collection([bcg] + other_individual_galaxies),
)
added = model_3.prior_count - without_members.prior_count
print(
    f"(a) free params:  without members = {without_members.prior_count},  with members = {model_3.prior_count}  (members added {added})"
)
assert added == 0

# (b) The members are driven by the BCG's sigma.
instance = model_3.instance_from_prior_medians()
sigma_bcg_instance = instance.individual_galaxies[0].mass.sigma
print(
    f"(b) instance BCG sigma = {sigma_bcg_instance:.2f} km/s;  members = sigma_BCG * (L/L_BCG)^0.25:"
)
worst_coupling = 0.0
for i, magnitude in enumerate(scaling_magnitudes):
    ratio = 10.0 ** (0.4 * (mag0 - magnitude))
    expected = sigma_bcg_instance * ratio**0.25
    actual = instance.scaling_galaxies[i].mass.sigma
    worst_coupling = max(worst_coupling, abs(actual - expected))
    print(
        f"    member {i} (mag {magnitude}):  sigma = {actual:6.2f} km/s  = {ratio**0.25:.3f} x sigma_BCG"
    )
assert worst_coupling < 1e-9
print("Model 3 checks passed.")


"""
__Model 4 — The Two-Parameter Relation__

**How.** Model 1 fixes the truncation reference ``r_cut_ref`` and frees only ``sigma_ref``. Here we
free ``r_cut_ref`` as well, so the member population has **two** shared free parameters — a dispersion
normalization and a truncation normalization:

    sigma_i = sigma_ref * (L/L_ref)^0.25       # sigma_ref FREE  (as Model 1)
    r_cut_i = r_cut_ref * (L/L_ref)^0.7        # r_cut_ref now FREE too (beta_cut = 1 + gamma - 2*alpha)

Every member's truncation now scales from a *free* reference instead of a fixed one. It is a one-symbol
change (``r_cut_ref`` becomes a prior) — but it is a real modelling decision, so it earns a section.

**Why.** The reference truncation ``r_cut*`` is genuinely uncertain — it sets how much mass each member
carries at large radius, and the data can constrain it. Lenstool's ``potfile`` routinely optimizes both
``sigma*`` and ``r_cut*``. You free it when the members contribute enough lensing that their outer mass
matters; you fix it (Model 1) when they do not, to save a parameter and avoid a weakly-constrained
direction. Constraining ``r_cut_ref`` from lensing is also scientifically interesting in its own right:
the compactness of cluster members' truncated subhalos is directly related to the galaxy-galaxy
strong-lensing excess in clusters reported by Meneghetti et al. (2020, Science) — observed members are
substantially more efficient galaxy-galaxy lenses than their LCDM-simulation counterparts, and the
fitted ``r_cut_ref`` is the model-side handle on that tension. Lensing-constrained values are typically
~5".

**Free vs fixed.** As Model 1, but the member tier now has 2 free parameters (``sigma_ref`` +
``r_cut_ref``) instead of 1. Total: 10.
"""

# --- Halo: as Model 1 ---
halo_mass = af.Model(al.mp.dPIEMass)
halo_mass.centre = (0.0, 0.0)  # [FIXED]
halo_mass.ellipticity = af.UniformPrior(lower_limit=0.0, upper_limit=0.7)  # [FREE]
halo_mass.angle_pos = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)  # [FREE]
halo_mass.sigma = af.UniformPrior(lower_limit=500.0, upper_limit=1500.0)  # [FREE]
halo_mass.r_core = af.UniformPrior(lower_limit=20.0, upper_limit=150.0)  # [FREE]
halo_mass.r_cut = 1000.0  # [FIXED]
halo_mass.redshift_object = redshift_lens
halo_mass.redshift_source = redshift_source
halo_mass.H0 = H0
halo_mass.Om0 = Om0
cluster_halo = af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass)

# --- Individually-modelled galaxies: as Model 1 ---
individual_sigma_priors = [(200.0, 600.0), (150.0, 450.0)]
individual_rcut_priors = [(2.0, 200.0), (2.0, 150.0)]

individual_galaxies = []
for centre, (sigma_lo, sigma_hi), (rcut_lo, rcut_hi) in zip(
    individual_centres, individual_sigma_priors, individual_rcut_priors
):
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = af.UniformPrior(lower_limit=sigma_lo, upper_limit=sigma_hi)  # [FREE]
    mass.r_core = 0.0  # [FIXED] vanishing core
    mass.r_cut = af.UniformPrior(lower_limit=rcut_lo, upper_limit=rcut_hi)  # [FREE]
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    individual_galaxies.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

# --- Members: TWO free normalizations, sigma_ref and r_cut_ref ---
sigma_ref = af.UniformPrior(lower_limit=100.0, upper_limit=400.0)  # [FREE] km/s
r_cut_ref = af.UniformPrior(
    lower_limit=1.0, upper_limit=20.0
)  # [FREE] arcsec  <-- now a prior, not a constant; lensing-constrained values are typically ~5"
mag0 = individual_magnitudes[0]  # [FIXED]
sigma_exponent = 0.25  # [FIXED] alpha
gamma = 0.2  # [FIXED]
rcut_exponent = 1.0 + gamma - 2.0 * sigma_exponent  # [TIED] beta_cut = 0.7
r_core_ref = 0.0  # [FIXED] vanishing core — never scaled with luminosity

scaling_galaxies = []
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    luminosity_ratio = 10.0 ** (0.4 * (mag0 - magnitude))

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = sigma_ref * luminosity_ratio**sigma_exponent  # tied to free sigma_ref
    mass.r_core = r_core_ref  # [FIXED] not scaled with luminosity
    mass.r_cut = r_cut_ref * luminosity_ratio**rcut_exponent  # tied to free r_cut_ref
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    scaling_galaxies.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

model_4 = af.Collection(
    cluster_halo=cluster_halo,
    individual_galaxies=af.Collection(individual_galaxies),
    scaling_galaxies=af.Collection(scaling_galaxies),
)

print("\n" + "=" * 80)
print("Model 4 — The Two-Parameter Relation")
print("=" * 80)
print(
    f"Total free parameters: {model_4.prior_count}   (Model 1 was 9; the member tier is now 2, not 1)"
)
assert model_4.prior_count == 10

# The member tier now contributes 2 free parameters (sigma_ref + r_cut_ref) instead of 1.
without_members = af.Collection(
    cluster_halo=cluster_halo, individual_galaxies=af.Collection(individual_galaxies)
)
added = model_4.prior_count - without_members.prior_count
print(f"member tier free parameters: {added}  (sigma_ref + r_cut_ref)")
assert added == 2
print("Model 4 checks passed.")

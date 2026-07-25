"""
Lenstool Users: dPIE Parameterization Mapping (Scaling Galaxies)
===============================================================

**If you want to fit cluster member (scaling) galaxies without knowing the source or lens redshift,
this script shows you how — and proves it recovers the standard Lenstool model exactly once the
redshifts become known.**

PyAutoLens ships the dPIE mass profile in two equivalent parameterizations:

 - ``al.mp.dPIEMass`` / ``dPIEMassSph`` — **Lenstool's native parameters** (``sigma`` = fiducial
   ``v_disp``, ``r_core``, ``r_cut``, plus the redshifts and cosmology). This is what
   ``cluster/modeling.py`` and ``cluster/start_here.py`` fit, and what a Lenstool ``.par`` file
   quotes. The lens strength ``b0`` is computed *internally* from ``sigma`` and the redshifts.

 - ``al.mp.dPIEMassB0`` / ``dPIEMassB0Sph`` — the **internal, angular parameters**
   (``ra``, ``rs``, ``b0``, all in arcsec). No redshift, no cosmology. ``b0`` is the lens strength
   directly — the quantity the image positions actually constrain.

The two are the same profile; they differ only in *which numbers you hold as free parameters*. The
bridge is a single formula (Eliasdottir et al. 2007, App. A; verified against the Lenstool C source
in ``autolens_workspace_test/scripts/cluster/lenstool_parity.py``):

    b0 = 6 * 648000 * (sigma / c)^2 * (D_LS / D_S)   [arcsec]           (forward, dPIEMass)
    sigma = c * sqrt(b0 / K),   K = 6 * 648000 * (D_LS / D_S)          (inverse, dPIEMassB0 -> Lenstool)

``r_core`` = ``ra`` and ``r_cut`` = ``rs`` are angular identities — they never involve the redshift.

__Why this matters for scaling galaxies__

Cluster members are not fitted individually; a whole population shares one *reference normalization*
tied to luminosity by a Faber-Jackson scaling relation. When the source/lens redshifts are unknown
you cannot fit ``sigma`` directly — the lensing only constrains ``b0``, and ``sigma`` is degenerate
with the redshift through ``D_LS / D_S`` (see ``cluster/lenstool/README.md``). So you fit the tier in
the **angular** ``b0`` parameterization (redshift-free), and convert to physical ``sigma`` afterwards.

This script demonstrates, with deflection angles as the test:

 1. **Single-profile parity** — a ``dPIEMassB0`` and its ``dPIEMass`` twin deflect light identically
    (including ellipticity).
 2. **The scaling relation is preserved exactly** — because the distance factor ``K`` is *common to
    the whole tier*, a ``b0 ~ L^0.5`` relation maps to a ``sigma ~ L^0.25`` Faber-Jackson relation
    with the correct exponent falling out of the square root. One free ``b0_ref`` maps to one free
    ``sigma_ref``.
 3. **Fit -> known redshift -> Lenstool** — a scaling tier fitted as ``af.Model(dPIEMassB0Sph)`` on a
    single free ``b0_ref``, once the redshifts are known, converts to the *exact*
    ``af.Model(dPIEMassSph)`` tier that ``cluster/modeling.py`` fits directly — member for member,
    to machine precision.
 4. **The Faber-Jackson exponent chain** — ``b0 ~ sigma^2`` exactly, so the ``b0 ~ L^0.5`` normalization
    *is* the Faber-Jackson relation ``sigma ~ L^0.25``; the radii scale as ``L^0.5`` from constant
    mass-to-light. One self-consistent system, not three independent choices.
 5. **Robustness** — the mapping is exact for *any* ``b0_ref`` across the prior and for *any* reference
    luminosity (fiducial or BCG-anchored), member by member.
 6. **Anchoring to the BCG** — the two physically distinct meanings of "linking the relation to the
    BCG" (anchor its *luminosity* vs anchor its *mass*), and why the second maps cleanly yet is the
    coupling to be wary of for galaxy/group-scale lenses.

Only the scaling tier is covered here (main lenses are modelled directly in Lenstool parameters in
the other cluster scripts).
"""

import numpy as np

import autofit as af
import autolens as al
import autogalaxy as ag

"""
__Cosmology, Redshifts, and the Distance Factor K__

The only place a redshift enters the dPIE deflection is the ratio ``D_LS / D_S`` (the lensing
efficiency). We wrap it into the single constant ``K`` so the ``b0 <-> sigma`` conversion is one
line. These are the same redshifts as the ``cluster/simple`` example (``z_lens = 0.5``,
``z_source = 2.0``).

The cosmology **must be the one ``dPIEMass`` uses internally** to build ``b0`` — a flat
``H0 = 67.66``, ``Om0 = 0.30966`` (the profile's ``H0`` / ``Om0`` defaults) — so the round-trip is
exact. Using a different cosmology here (e.g. ``Planck15`` with massive neutrinos) shifts the
distance ratio by ~0.01% and the mapping is then only approximate.

``K`` depends *only* on the redshifts and cosmology — **not** on the galaxy. Every member of a
scaling tier sits at the same lens redshift and lenses the same source plane, so they all share this
one ``K``. That shared-ness is the whole reason the scaling relation survives the conversion intact.
"""
C_KM_S = 299792.458

redshift_lens = 0.5
redshift_source = 2.0

# Match dPIEMass's internal cosmology (flat H0=67.66, Om0=0.30966) so b0 <-> sigma is exact.
cosmology = ag.cosmo.FlatLambdaCDM(H0=67.66, Om0=0.30966)

d_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_source)
d_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
    redshift_0=redshift_lens, redshift_1=redshift_source
)

K = (
    6.0 * 648000.0 * (d_ls / d_s)
)  # b0 = K * (sigma / c)^2  <=>  sigma = c * sqrt(b0 / K)

print(f"Distance factor K = {K:.4f}  (b0 [arcsec] = K * (sigma/c)^2)")


def sigma_from_b0(b0):
    """Invert the dPIE lens strength to Lenstool's fiducial velocity dispersion (km/s)."""
    return C_KM_S * (b0 / K) ** 0.5


"""
__1. Single-Profile Parity (the mechanism)__

Before the scaling relation, confirm the two parameterizations are literally the same profile. We
build a ``dPIEMass`` from Lenstool parameters (including ellipticity), read off the ``ell_comps`` and
``b0`` it computed internally, construct the angular ``dPIEMassB0`` twin from those, and check the
deflection fields agree to machine precision.

The test grid is a handful of off-centre positions; deflection angles are what the point-source
likelihood actually compares, so parity here is parity where it counts.
"""
grid = al.Grid2DIrregular([[3.0, 4.0], [-2.0, 1.5], [8.0, -3.0]])

mass_lenstool = al.mp.dPIEMass(
    centre=(0.1, -0.2),
    ellipticity=0.3,
    angle_pos=40.0,
    sigma=250.0,
    r_core=1.5,
    r_cut=25.0,
    redshift_object=redshift_lens,
    redshift_source=redshift_source,
)

# The angular twin: same centre, the ell_comps/b0 the Lenstool profile derived, and r_core/r_cut
# carried across unchanged (they are angular identities).
mass_b0 = al.mp.dPIEMassB0(
    centre=mass_lenstool.centre,
    ell_comps=mass_lenstool.ell_comps,
    ra=mass_lenstool.r_core,
    rs=mass_lenstool.r_cut,
    b0=mass_lenstool.b0,
)

deflections_lenstool = np.asarray(mass_lenstool.deflections_yx_2d_from(grid=grid))
deflections_b0 = np.asarray(mass_b0.deflections_yx_2d_from(grid=grid))

max_diff = np.max(np.abs(deflections_lenstool - deflections_b0))
print(
    f"\n1. Single-profile parity (elliptical):  max deflection difference = {max_diff:.2e}"
)
assert max_diff < 1e-12

# And the inverse direction we will actually use: b0 -> sigma reconstructs the Lenstool profile.
sigma_recovered = sigma_from_b0(mass_b0.b0)
print(f"   sigma recovered from b0:  {sigma_recovered:.4f} km/s  (input was 250.0)")
assert np.isclose(sigma_recovered, 250.0, rtol=1e-6)

"""
__2. The Scaling Relation is Preserved Exactly__

Scaling galaxies are spherical (``dPIEMassSph`` / ``dPIEMassB0Sph`` — the ellipticity-zero members of
the families above), so from here on we drop the angle and ellipticity.

The reference-anchored relation used by Lenstool and every published cluster analysis ties each
member to a reference luminosity ``L_ref`` (Lenstool's ``mag0``):

    Lenstool (sigma-space):   sigma_i = sigma_ref * (L_i / L_ref)^0.25     (Faber-Jackson)
    Angular  (b0-space):      b0_i    = b0_ref    * (L_i / L_ref)^0.5

These are the *same relation*. Because ``sigma_i = c * sqrt(b0_i / K)`` and ``K`` is common to the
whole tier:

    sigma_i = c * sqrt( b0_ref * (L_i/L_ref)^0.5 / K )
            = [ c * sqrt(b0_ref / K) ] * (L_i/L_ref)^0.25
            = sigma_ref * (L_i/L_ref)^0.25,     with   sigma_ref = c * sqrt(b0_ref / K)

The square root **halves the exponent**: a ``b0 ~ L^0.5`` relation is a ``sigma ~ L^0.25``
Faber-Jackson relation, automatically. The single free normalization ``b0_ref`` maps to the single
free normalization ``sigma_ref`` — no extra parameters, no distortion of the *relative* member
masses (the ``(L_i/L_ref)`` factors are dimensionless and redshift-free). The core and truncation
radii scale as ``L^0.5`` in both, and carry across as ``r_core = ra``, ``r_cut = rs`` identities.

Note the radii values and scaling used on this page (``r_core_ref = 0.158``, ``r_cut_ref = 15.8``,
radii ``~ L^0.5``) are Lenstool's **classic** ``potfile`` convention (``slope 4``, scaled cores) —
documented here deliberately, because this page's job is mapping Lenstool's native parameterization.
``scripts/cluster/`` itself now defaults to the modern convention: the tied truncation exponent
``beta_cut = 1 + gamma - 2*alpha = 0.7`` (``gamma = 0.2``), vanishing unscaled cores and
``r_cut_ref ~ 5"`` — see ``cluster/mass_parameterizations.py``. The exponent-halving identity above
is unaffected (it applies to whatever radii scaling you choose).

We verify the exponent-halving numerically for a spread of member luminosities.
"""
reference_luminosity = (
    1.0  # Lenstool "mag0" — an explicit fixed anchor, not the sample max
)
member_luminosities = [0.40, 0.25, 0.16, 0.10, 0.06]

r_core_ref = 0.158  # arcsec, fixed
r_cut_ref = 15.8  # arcsec, fixed

# Pick the b0_ref that a redshift-free fit of the `cluster/simple` truth would return: the value
# that converts (with THESE redshifts) to sigma_ref = 85 km/s — exactly the normalization
# `cluster/modeling.py` fits directly. This makes the "we recover the standard example" claim literal.
sigma_ref_target = 85.0
b0_ref = K * (sigma_ref_target / C_KM_S) ** 2

sigma_ref = sigma_from_b0(b0_ref)
print(
    f"\n2. Scaling relation mapping:  b0_ref = {b0_ref:.5f} arcsec  ->  sigma_ref = {sigma_ref:.4f} km/s"
)
assert np.isclose(sigma_ref, sigma_ref_target, rtol=1e-10)

max_relation_diff = 0.0
for luminosity in member_luminosities:
    ratio = luminosity / reference_luminosity
    b0_i = b0_ref * ratio**0.5
    sigma_i_direct = sigma_from_b0(b0_i)  # convert the member's own b0
    sigma_i_relation = sigma_ref * ratio**0.25  # Faber-Jackson in sigma-space
    max_relation_diff = max(max_relation_diff, abs(sigma_i_direct - sigma_i_relation))
print(
    f"   sigma_i(from b0_i)  vs  sigma_ref * (L/L_ref)^0.25:  max diff = {max_relation_diff:.2e}"
)
assert max_relation_diff < 1e-9

"""
__3. Fit -> Known Redshift -> Recover the Lenstool Model__

Now the end-to-end demonstration. We build the scaling tier the way you would **fit** it when the
redshifts are unknown: an ``af.Model(dPIEMassB0Sph)`` per member, every ``b0`` tied to a *single*
free ``b0_ref`` (one prior for the whole tier), everything in arcsec. This is a well-posed,
redshift-free model.

We then simulate a "fit result" by drawing the instance at our known ``b0_ref``, and — the redshifts
now being known — convert ``b0_ref -> sigma_ref`` and build the equivalent ``af.Model(dPIEMassSph)``
tier: the *exact* parameterization ``cluster/modeling.py`` fits. Drawing the corresponding
``sigma_ref`` instance, we confirm every member deflects light identically.

Member centres are illustrative (as in ``cluster/simulator.py``).
"""
member_centres = [(5.5, -6.5), (-7.5, 3.0), (3.0, 13.0), (12.0, -5.0), (-6.5, 11.0)]

# --- Fit-time model: one free b0_ref, angular, redshift-free ---
b0_ref_prior = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

b0_member_models = []
for centre, luminosity in zip(member_centres, member_luminosities):
    ratio = luminosity / reference_luminosity
    mass = af.Model(al.mp.dPIEMassB0Sph)
    mass.centre = centre
    mass.b0 = b0_ref_prior * ratio**0.5
    mass.ra = r_core_ref * ratio**0.5
    mass.rs = r_cut_ref * ratio**0.5
    b0_member_models.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

tier_b0 = af.Collection(b0_member_models)
print(
    f"\n3. Fit-time tier (b0-space) free parameters:  {tier_b0.prior_count}  (expect 1: b0_ref)"
)
assert tier_b0.prior_count == 1

# Simulate the fit returning b0_ref (UniformPrior(0, 1) -> unit value == b0_ref).
b0_ref_fit = b0_ref
instance_b0 = tier_b0.instance_from_unit_vector([b0_ref_fit])

# --- Redshifts now known: convert, then build the Lenstool-parameterized tier ---
sigma_ref_fit = sigma_from_b0(b0_ref_fit)

sigma_ref_prior = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)

lenstool_member_models = []
for centre, luminosity in zip(member_centres, member_luminosities):
    ratio = luminosity / reference_luminosity
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = sigma_ref_prior * ratio**0.25  # Faber-Jackson exponent
    mass.r_core = r_core_ref * ratio**0.5
    mass.r_cut = r_cut_ref * ratio**0.5
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = 67.66
    mass.Om0 = 0.30966
    lenstool_member_models.append(
        af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
    )

tier_lenstool = af.Collection(lenstool_member_models)
print(
    f"   Lenstool tier (sigma-space) free parameters:  {tier_lenstool.prior_count}  (expect 1: sigma_ref)"
)
assert tier_lenstool.prior_count == 1

instance_lenstool = tier_lenstool.instance_from_unit_vector([sigma_ref_fit / 300.0])

# --- Deflection test: the two instances must be the same tier, member for member ---
print(f"   b0_ref = {b0_ref_fit:.5f} arcsec  ->  sigma_ref = {sigma_ref_fit:.4f} km/s")

max_member_diff = 0.0
for galaxy_b0, galaxy_lenstool in zip(instance_b0, instance_lenstool):
    d_b0 = np.asarray(galaxy_b0.mass.deflections_yx_2d_from(grid=grid))
    d_lenstool = np.asarray(galaxy_lenstool.mass.deflections_yx_2d_from(grid=grid))
    max_member_diff = max(max_member_diff, np.max(np.abs(d_b0 - d_lenstool)))
print(
    f"   per-member deflection agreement (b0-tier vs Lenstool-tier):  max diff = {max_member_diff:.2e}"
)
assert max_member_diff < 1e-12

"""
__4. The Faber-Jackson Exponent Chain__

The three exponents (``sigma ~ L^0.25``, ``b0 ~ L^0.5``, ``r_core, r_cut ~ L^0.5``) are not three
independent choices — they are one self-consistent system resting on a single physical fact: for the
dPIE the lens strength is **exactly** proportional to the velocity dispersion squared,

    b0 = K * (sigma / c)^2     =>     b0  is proportional to  sigma^2 .

From there:

 - Faber-Jackson (Faber & Jackson 1976), ``L ~ sigma^4``, gives ``sigma ~ L^0.25``. Squaring,
   ``b0 ~ sigma^2 ~ L^0.5`` — the ``b0``-space normalization *is* the Faber-Jackson relation, not an
   approximation of it. This is why the exponent halves when you move from ``b0`` to ``sigma``.
 - Constant mass-to-light (``M ~ sigma^2 * r_cut ~ L``) with ``sigma ~ L^0.25`` forces
   ``r_cut ~ L^0.5``; the core radius follows the same scaling (Lenstool's ``potfile`` applies it to
   both). These are angular, so they are identical in the two parameterizations.

We confirm ``b0 ~ sigma^2`` directly (``b0 / sigma^2`` is constant), then that a member's ``b0`` built
from its Faber-Jackson ``sigma`` equals its ``b0`` from the direct ``L^0.5`` relation.
"""
b0_over_sigma_sq = []
for sigma_test in [100.0, 200.0, 300.0]:
    mass_test = al.mp.dPIEMassSph(
        sigma=sigma_test,
        r_core=0.5,
        r_cut=20.0,
        redshift_object=redshift_lens,
        redshift_source=redshift_source,
    )
    b0_over_sigma_sq.append(mass_test.b0 / sigma_test**2)
print(
    f"\n4. Faber-Jackson chain:  b0 / sigma^2 = {b0_over_sigma_sq[0]:.6e}  (constant => b0 ~ sigma^2)"
)
assert np.allclose(b0_over_sigma_sq, b0_over_sigma_sq[0], rtol=1e-12)

ratio_check = 0.16
b0_via_faber_jackson = (
    K * (sigma_ref * ratio_check**0.25 / C_KM_S) ** 2
)  # b0 built from the FJ sigma
b0_via_b0_relation = b0_ref * ratio_check**0.5  # b0-space relation directly
print(
    f"   member (L/L_ref={ratio_check}):  b0 via Faber-Jackson sigma = {b0_via_faber_jackson:.6f}"
    f"  vs  b0 via L^0.5 = {b0_via_b0_relation:.6f}"
)
assert np.isclose(b0_via_faber_jackson, b0_via_b0_relation, rtol=1e-9)

"""
__5. Robustness: All Members, Full Prior Range, Either Reference__

Section 3 checked a single ``b0_ref``. The mapping is exact for *any* ``b0_ref`` (the conversion
``sigma = c * sqrt(b0 / K)`` is exact for all ``b0``) and for *any* reference luminosity — the
``(L / L_ref)`` ratios are dimensionless and cancel identically in both parameterizations, so the
choice of anchor cannot change the *relative* member masses.

We sweep several ``b0_ref`` across the prior under both the fiducial (``L_ref = 1``) and the
BCG-anchored (``L_ref =`` brightest member) conventions, checking every member stays in parity. This
is the "over all galaxies, comparable everywhere" guarantee, not just parity at one point.
"""
worst_robustness = 0.0
reference_cases = [
    (1.0, "fiducial L_ref=1"),
    (max(member_luminosities), "L_ref=brightest member"),
]
for reference_case, reference_tag in reference_cases:
    for b0_ref_case in [0.05, 0.20, 0.50, 0.90]:
        sigma_ref_case = sigma_from_b0(b0_ref_case)
        for centre, luminosity in zip(member_centres, member_luminosities):
            ratio = luminosity / reference_case
            mass_b0_case = al.mp.dPIEMassB0Sph(
                centre=centre,
                ra=r_core_ref * ratio**0.5,
                rs=r_cut_ref * ratio**0.5,
                b0=b0_ref_case * ratio**0.5,
            )
            mass_ls_case = al.mp.dPIEMassSph(
                centre=centre,
                sigma=sigma_ref_case * ratio**0.25,
                r_core=r_core_ref * ratio**0.5,
                r_cut=r_cut_ref * ratio**0.5,
                redshift_object=redshift_lens,
                redshift_source=redshift_source,
            )
            diff = np.max(
                np.abs(
                    np.asarray(mass_b0_case.deflections_yx_2d_from(grid=grid))
                    - np.asarray(mass_ls_case.deflections_yx_2d_from(grid=grid))
                )
            )
            worst_robustness = max(worst_robustness, diff)
print(
    f"\n5. Robustness (2 references x 4 b0_ref x all members):  worst deflection diff = {worst_robustness:.2e}"
)
assert worst_robustness < 1e-12

"""
__6. Anchoring the Reference to the BCG (main lens galaxy)__

"Linking the scaling relation to the BCG" can mean two physically different things, and the
distinction matters more than the mathematics:

 (a) **Anchor the reference LUMINOSITY to the BCG** (``L_ref = L_BCG``). The normalization
     (``sigma_ref`` / ``b0_ref``) stays a *free* parameter; the BCG's own fitted mass is independent
     of the tier. The mapping is unaffected (Section 5 tested exactly this via ``L_ref = brightest
     member``). **This is the workspace convention** — ``group/slam.py`` anchors ``L_ref`` to the
     brightest main galaxy while keeping its mass free and uncoupled from the tier.

 (b) **Anchor the reference NORMALIZATION to the BCG's MASS** — the strict Lenstool / referee
     convention ``alpha = theta_E,BCG / L_BCG^0.5``, ``beta = 0.5``, with *zero* free parameters: the
     fitted BCG mass drives every satellite. The ``b0 <-> sigma`` mapping is *still exact* here,
     because the BCG shares the lens plane and the source plane (same ``K``). But it couples every
     satellite's mass to the BCG's — a strong assumption, often unphysical for galaxy- and
     group-scale lenses where the BCG is modelled as its own free profile. The mapping being exact
     does not make that modelling choice safe; they are separate questions.

We demonstrate that case (b) maps exactly: a BCG modelled as its own dPIE (``sigma_BCG``), with the
tier anchored to it, agrees member-for-member in both parameterizations.
"""
sigma_bcg = 320.0
b0_bcg = (
    K * (sigma_bcg / C_KM_S) ** 2
)  # the BCG's own lens strength; L_BCG = 1.0 is the reference
print(
    f"\n6. BCG-anchored:  sigma_BCG = {sigma_bcg} km/s  <->  b0_BCG = {b0_bcg:.5f} arcsec"
)

worst_bcg = 0.0
for centre, luminosity in zip(member_centres, member_luminosities):
    ratio = luminosity / 1.0  # the BCG (L_BCG = 1.0) is the reference
    mass_b0_bcg = al.mp.dPIEMassB0Sph(
        centre=centre,
        ra=r_core_ref * ratio**0.5,
        rs=r_cut_ref * ratio**0.5,
        b0=b0_bcg * ratio**0.5,
    )
    mass_ls_bcg = al.mp.dPIEMassSph(
        centre=centre,
        sigma=sigma_bcg * ratio**0.25,
        r_core=r_core_ref * ratio**0.5,
        r_cut=r_cut_ref * ratio**0.5,
        redshift_object=redshift_lens,
        redshift_source=redshift_source,
    )
    worst_bcg = max(
        worst_bcg,
        np.max(
            np.abs(
                np.asarray(mass_b0_bcg.deflections_yx_2d_from(grid=grid))
                - np.asarray(mass_ls_bcg.deflections_yx_2d_from(grid=grid))
            )
        ),
    )
print(
    f"   tier anchored to BCG mass:  all-member b0-vs-sigma max defl diff = {worst_bcg:.2e}"
)
assert worst_bcg < 1e-12

"""
__Caveat: dPIE b0 is the Einstein radius only in the SIS limit__

The referee's convention anchors to the BCG *Einstein radius* ``theta_E``. For a dPIE, ``b0`` equals
``theta_E`` only in the SIS limit (``r_core -> 0``, ``r_cut -> infinity``, ``q -> 1``); with finite
core/cut ``b0`` is the lens strength but *not* exactly ``theta_E``. If the BCG is modelled as a
distinct isothermal (SIE) — as is common for galaxy- and group-scale lenses — its ``einstein_radius``
is well defined, but mapping it onto a dPIE member's ``b0`` is only approximate unless the member's
core/cut are negligible. Anchor via a checked conversion, not a straight ``b0 = einstein_radius``
assignment.

__7. Why This is Safe__

The mapping is lossless and invertible, and it does not perturb anything the data constrained:

 - **The fit only ever constrained an angular quantity.** Deflection angles depend on ``b0``, not on
   ``sigma`` and the redshifts separately (``sigma`` and ``D_LS/D_S`` enter only as the product
   ``b0``). Fitting ``b0_ref`` is fitting exactly the identifiable degree of freedom; freeing
   ``sigma`` and the redshifts instead would sample a direction the images cannot pin down (the
   ``sigma``-redshift degeneracy of ``cluster/lenstool/README.md``).

 - **Redshifts enter only as a common rescaling.** Every member shares the one factor ``K``, so
   converting ``b0 -> sigma`` multiplies the whole tier by the same constant (in quadrature). It
   cannot change the *relative* masses of the members, and it turns the ``b0 ~ L^0.5`` relation into
   the ``sigma ~ L^0.25`` Faber-Jackson relation exactly — the physics the reference-anchored
   convention encodes.

 - **You recover the published Lenstool model.** The converted ``sigma_ref`` tier above is, member
   for member, the ``af.Model(dPIEMassSph)`` scaling tier that ``cluster/modeling.py`` and
   ``cluster/start_here.py`` fit directly — so a redshift-free fit followed by this conversion lands
   on the standard Lenstool parameterization, not an approximation of it.

The practical recipe: **fit scaling galaxies in ``dPIEMassB0Sph`` (angular, one free ``b0_ref``);
once the redshifts are known, convert with ``sigma = c * sqrt(b0 / K)`` to report physical velocity
dispersions and to reproduce the Lenstool member model.** Radii convert to kpc with
``cosmology.kpc_per_arcsec_from(redshift=redshift_lens)``; the physical central dispersion is
``sigma_0 = sqrt(3/2) * sigma`` (fiducial-vs-central, Eliasdottir et al. 2007).
"""
print("\nAll parameterization-mapping checks passed.")

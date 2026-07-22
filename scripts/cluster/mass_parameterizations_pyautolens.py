"""
Mass Parameterizations II — Mapping Lenstool onto PyAutoLens
============================================================

**For Lenstool users: how and why PyAutoLens re-expresses the standard Lenstool cluster / group model
in its native parameterization for multi-galaxy lenses (MGLs), and how to keep the dPIE's truncation on
the outskirts if you want it.**

The companion script ``cluster/mass_parameterizations.py`` builds the standard Lenstool model. This
script takes its **Model 1** as the starting point and maps it, component by component, onto what
PyAutoLens uses at galaxy / group scale. Both are supported; this is an *alternative*, not a
replacement, and the focus is the **mass**.

Three sections:

 - **Section 1 — Lenstool Model 1** (dPIE throughout, velocity dispersions, a magnitude-anchored
   Faber-Jackson relation), the starting point.
 - **Section 2 — The standard PyAutoLens model**, mapping every component and justifying each change.
 - **Section 3 — Keeping truncation**: dPIE for the scaling galaxies while the main and extra galaxies
   stay Isothermal, and how the Faber-Jackson linking bridges the two parameterizations.

Every model has four tiers, and this is the structure of a PyAutoLens MGL:

    Tier              Role                                    Coupling to the BGC
    ----------------  --------------------------------------  --------------------------------
    cluster halo      the smooth large-scale mass             none (its own free profile)
    main galaxies     the primary lens galaxies (incl. BGC)   none — free
    extra galaxies    bright members worth freeing            free, but luminosity-BOUNDED
    scaling galaxies  the faint outskirts ("potfile")         TIED — mass set by the relation

The luminosity that drives the relation is computed from each galaxy's fitted MGE light,
``L = sum_g 2*pi*sigma_g^2/q_g * intensity_g`` (the ``2*np.pi*g.sigma**2/g.axis_ratio()*g.intensity``
you have seen); only *ratios* matter, so we drive everything off illustrative magnitudes here
(``L/L_BGC = 10 ** (0.4 * (mag0 - m))``).

Everything uses the model-composition API and prints ``model.info`` so free vs fixed is explicit.

__What changes, at a glance__

    Component         Lenstool (Section 1)          PyAutoLens (Sections 2-3)
    ----------------  ----------------------------  --------------------------------------
    cluster halo      elliptical dPIE (sigma)       NFW  (the CDM halo profile)
    main galaxies     dPIE (sigma, r_core, r_cut)   Isothermal (einstein_radius)
    extra galaxies    dPIE, free                    Isothermal, free but luminosity-bounded
    scaling galaxies  dPIE, free sigma_ref          Isothermal, TIED to the BGC's Einstein radius
    mass parameter    sigma (needs redshifts)       einstein_radius (arcsec; redshift-free)
    luminosity        input magnitude               computed from the fitted MGE light
"""

import numpy as np

import autofit as af
import autolens as al
import autogalaxy as ag

"""
__Setup__

Redshifts / cosmology are needed by the *Lenstool* (sigma) side and again to *check* the
sigma <-> einstein_radius bridge — they do not appear in the PyAutoLens mass model.
"""
redshift_lens = 0.5
redshift_source = 2.0
H0 = 67.66
Om0 = 0.30966

main_centres = [
    (0.0, 0.0),
    (12.0, 8.0),
]  # primary lens galaxies (the brightest is the BGC)
main_magnitudes = [17.8, 18.9]

extra_centres = [
    (8.5, 5.5)
]  # a bright member worth freeing (e.g. near a multiple image)
extra_magnitudes = [18.5]

scaling_centres = [(5.5, -6.5), (-7.5, 3.0), (3.0, 13.0)]  # the faint outskirts
scaling_magnitudes = [19.2, 20.0, 21.0]

bgc_index = int(
    np.argmin(main_magnitudes)
)  # brightest (smallest magnitude) main galaxy
mag0 = main_magnitudes[bgc_index]  # the BGC magnitude anchors the luminosity ratios


def luminosity_ratio(magnitude):
    """L / L_BGC from a magnitude, relative to the BGC (mag0)."""
    return 10.0 ** (0.4 * (mag0 - magnitude))


"""
================================================================================
__Section 1 — Lenstool Model 1__
================================================================================

The full standard Lenstool model, dPIE throughout (see ``cluster/mass_parameterizations.py`` Model 1).
The main and extra galaxies are both individually-modelled dPIEs (free ``sigma`` + ``r_cut``); the
distinction between them is a PyAutoLens refinement (Section 2) — in Lenstool they are both just
"potentiel" sections. The scaling galaxies use a magnitude-anchored Faber-Jackson relation with a free
normalization ``sigma_ref``: ``sigma_i = sigma_ref * (L/L_ref) ** 0.25``.

Total: 11 free (halo 4 + 2 main x 2 + 1 extra x 2 + scaling 1).
"""


def dpie_galaxy(centre, sigma_lo, sigma_hi, rcut_lo, rcut_hi):
    """An individually-modelled Lenstool galaxy: free dPIE dispersion + truncation."""
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre
    mass.sigma = af.UniformPrior(
        lower_limit=sigma_lo, upper_limit=sigma_hi
    )  # [FREE] km/s
    mass.r_core = 0.3
    mass.r_cut = af.UniformPrior(
        lower_limit=rcut_lo, upper_limit=rcut_hi
    )  # [FREE] arcsec
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    return af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)


# Cluster halo — elliptical dPIE, fully free.
halo_mass = af.Model(al.mp.dPIEMass)
halo_mass.centre = (0.0, 0.0)  # [FIXED]
halo_mass.ellipticity = af.UniformPrior(lower_limit=0.0, upper_limit=0.7)  # [FREE]
halo_mass.angle_pos = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)  # [FREE]
halo_mass.sigma = af.UniformPrior(lower_limit=500.0, upper_limit=1500.0)  # [FREE] km/s
halo_mass.r_core = af.UniformPrior(lower_limit=20.0, upper_limit=150.0)  # [FREE] arcsec
halo_mass.r_cut = 1000.0  # [FIXED]
halo_mass.redshift_object = redshift_lens
halo_mass.redshift_source = redshift_source
halo_mass.H0 = H0
halo_mass.Om0 = Om0
cluster_halo_lt = af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass)

main_galaxies_lt = [dpie_galaxy(c, 100.0, 600.0, 20.0, 200.0) for c in main_centres]
extra_galaxies_lt = [dpie_galaxy(c, 100.0, 450.0, 20.0, 150.0) for c in extra_centres]

# Scaling galaxies — Faber-Jackson on a FREE normalization sigma_ref (independent of the BGC).
sigma_ref = af.UniformPrior(lower_limit=100.0, upper_limit=400.0)  # [FREE] km/s
r_core_ref, r_cut_ref = 0.15, 20.0  # [FIXED] arcsec
scaling_galaxies_lt = []
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    ratio = luminosity_ratio(magnitude)
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = centre  # [FIXED]
    mass.sigma = sigma_ref * ratio**0.25  # tied to the one free sigma_ref
    mass.r_core = r_core_ref * ratio**0.5
    mass.r_cut = r_cut_ref * ratio**0.5
    mass.redshift_object = redshift_lens
    mass.redshift_source = redshift_source
    mass.H0 = H0
    mass.Om0 = Om0
    scaling_galaxies_lt.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

model_lenstool = af.Collection(
    cluster_halo=cluster_halo_lt,
    main_galaxies=af.Collection(main_galaxies_lt),
    extra_galaxies=af.Collection(extra_galaxies_lt),
    scaling_galaxies=af.Collection(scaling_galaxies_lt),
)

print("=" * 80)
print("Section 1 — Lenstool Model 1 (dPIE throughout)")
print("=" * 80)
print(
    f"Free parameters: {model_lenstool.prior_count}  (halo 4 + 2 main x 2 + 1 extra x 2 + scaling 1 = 11)"
)
assert model_lenstool.prior_count == 11


"""
================================================================================
__Section 2 — The Standard PyAutoLens Model__
================================================================================

We map Model 1 onto the PyAutoLens MGL parameterization. Component by component:

**(1) The halo: dPIE -> NFW.** Lenstool models the smooth halo as a dPIE for analytic convenience;
PyAutoLens uses the **NFW**, the CDM-motivated halo profile. A genuine *profile change*, not a
reparameterization — you refit ``kappa_s`` / ``scale_radius`` rather than convert. (Keep a dPIE halo if
you want to reproduce a Lenstool halo; both are supported.)

**(2) Galaxies: dPIE -> Isothermal (SIE).** One mass parameter, ``einstein_radius`` — the reduced
deflection, in arcsec, redshift-free, exactly what the images constrain. It equals the dPIE ``b0`` in
the SIS limit (checked below). No core, no truncation.

**(3) The coupling spectrum.** The individually-modelled galaxies split into two tiers, and the scaling
tier is anchored to the brightest galaxy (the BGC):

 - **main galaxies** — free ``einstein_radius``, no coupling.
 - **extra galaxies** — free ``einstein_radius``, but with a luminosity-informed *upper bound*
   ``min(2 * (upper_einstein_radius / L_BGC^0.5) * L^0.5, 5.0)`` — the BGC-scaled Faber-Jackson
   prediction (x2, capped). It prevents a runaway mass while keeping the galaxy free. This is the
   pipeline's middle tier.
 - **scaling galaxies** — *tied*: ``einstein_radius_i = einstein_radius_BGC * (L/L_BGC) ** 0.5``. Zero
   free parameters; the members inherit the BGC's mass (mass-anchored coupling, companion Model 3).

So the magnitude of Model 1 splits its two jobs: the luminosity *ratio* still enters (now from the
fitted light), but the *normalization* is no longer a free ``sigma_ref`` — it is the BGC's own
``einstein_radius``.

Total: 7 free (NFW halo 4 + 2 main einstein_radii + 1 extra einstein_radius + scaling 0).
"""

# (1) Halo -> elliptical NFW.
halo_mass = af.Model(al.mp.NFW)
halo_mass.centre = (0.0, 0.0)  # [FIXED]
halo_mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=-0.5, upper_limit=0.5
)  # [FREE]
halo_mass.ell_comps.ell_comps_1 = af.UniformPrior(
    lower_limit=-0.5, upper_limit=0.5
)  # [FREE]
halo_mass.kappa_s = af.UniformPrior(lower_limit=0.05, upper_limit=0.5)  # [FREE]
halo_mass.scale_radius = af.UniformPrior(
    lower_limit=10.0, upper_limit=60.0
)  # [FREE] arcsec
cluster_halo_pa = af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass)

# (2,3a) Main galaxies -> SIE, free einstein_radius. The BGC is the brightest.
main_galaxies_pa = []
for centre in main_centres:
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre  # [FIXED]
    mass.einstein_radius = af.UniformPrior(
        lower_limit=0.0, upper_limit=3.0
    )  # [FREE] arcsec
    main_galaxies_pa.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

einstein_radius_bgc = main_galaxies_pa[bgc_index].mass.einstein_radius  # the BGC anchor
upper_einstein_radius = (
    3.0  # fixed proxy for the BGC Einstein radius (the main-lens prior upper bound)
)

# (3b) Extra galaxies -> SIE, free einstein_radius with a luminosity-informed upper bound.
extra_galaxies_pa = []
for centre, magnitude in zip(extra_centres, extra_magnitudes):
    ratio = luminosity_ratio(magnitude)  # L / L_BGC
    upper = min(
        2.0 * (upper_einstein_radius / 1.0**0.5) * ratio**0.5, 5.0
    )  # L_BGC = 1 by construction
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre  # [FIXED]
    mass.einstein_radius = af.UniformPrior(
        lower_limit=0.0, upper_limit=upper
    )  # [FREE, BOUNDED]
    extra_galaxies_pa.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

# (3c) Scaling galaxies -> SIE, einstein_radius tied to the BGC.
scaling_galaxies_pa = []
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    ratio = luminosity_ratio(magnitude)
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre  # [FIXED]
    mass.einstein_radius = (
        einstein_radius_bgc * ratio**0.5
    )  # tied to the BGC -> NO new parameter
    scaling_galaxies_pa.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

model_pyautolens = af.Collection(
    cluster_halo=cluster_halo_pa,
    main_galaxies=af.Collection(main_galaxies_pa),
    extra_galaxies=af.Collection(extra_galaxies_pa),
    scaling_galaxies=af.Collection(scaling_galaxies_pa),
)

print("\n" + "=" * 80)
print("Section 2 — Standard PyAutoLens model (NFW halo, Isothermal galaxies)")
print("=" * 80)
print(
    f"Free parameters: {model_pyautolens.prior_count}  (NFW 4 + 2 main + 1 extra + scaling 0 = 7)"
)
assert model_pyautolens.prior_count == 7

# --- Numerical bridge: the scaling-galaxy parameter change is exact ---
C_KM_S = 299792.458
cosmology = ag.cosmo.FlatLambdaCDM(H0=H0, Om0=Om0)
d_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_source)
d_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
    redshift_0=redshift_lens, redshift_1=redshift_source
)
grid = al.Grid2DIrregular([[1.5, 2.0], [-1.0, 0.8], [3.0, -2.5]])

fj = [
    (4.0 * np.pi * (s / C_KM_S) ** 2 * (d_ls / d_s) * (648000.0 / np.pi)) / s**2
    for s in [150.0, 250.0, 350.0]
]
print(
    f"(a) einstein_radius / sigma^2 = {fj[0]:.3e}  (constant => einstein_radius ~ sigma^2, Faber-Jackson carries over)"
)
assert np.allclose(fj, fj[0], rtol=1e-12)

theta = 1.4
diff = np.max(
    np.abs(
        np.asarray(
            al.mp.IsothermalSph(
                centre=(0.0, 0.0), einstein_radius=theta
            ).deflections_yx_2d_from(grid=grid)
        )
        - np.asarray(
            al.mp.dPIEMassB0Sph(
                centre=(0.0, 0.0), ra=1e-6, rs=1e7, b0=theta
            ).deflections_yx_2d_from(grid=grid)
        )
    )
)
print(
    f"(b) IsothermalSph(einstein_radius={theta}) vs dPIE b0 (SIS limit):  max defl diff = {diff:.2e}"
)
assert diff < 1e-5
print("Section 2 checks passed.")


"""
================================================================================
__Section 3 — Keeping the Truncation (dPIE outskirts, Isothermal main + extra)__
================================================================================

Section 2 dropped the members' truncation. Here is how to keep it, changing as little as possible: the
halo (NFW), the main galaxies and the extra galaxies (all Isothermal) are unchanged from Section 2 —
**only the scaling galaxies become truncated dPIEs**.

**The difference truncation makes (all at large radius).**

    R (arcsec)     SIE deflection      dPIE (rs=20) deflection
        1              1.40                  1.36
       20              1.40                  0.82
      300              1.40                  0.09

The SIE has ``rho ~ r^-2`` forever: deflection stays at ``einstein_radius``, mass diverges (``M ~ R``).
The dPIE falls as ``r^-4`` beyond ``rs``: deflection dies away, total mass is *finite*. Inside the arc
region (R ~ 1-5") they agree to a few percent — which is why the SIE is fine where the constraints live.

**When MGLs do not need it.** ``rs`` models tidal *stripping* of a galaxy's outer halo by the host
potential. In a rich cluster stripping is strong and members are packed, so untruncated SIEs would each
contribute unbounded, overlapping mass — truncation is needed. In an MGL that is *not* a rich group the
potential is shallower, stripping weaker, and members are subdominant perturbers with the arcs near the
main lens — so the untruncated SIE is a defensible simplification. Keep truncation if the members are
numerous / massive enough to contribute meaningfully at large radius.

**How, staying closest to the parameterization.** There is no separate "truncated isothermal" class —
the dPIE *is* it (with ``ra -> 0`` it is a Pseudo-Jaffe). So the scaling galaxies become
``dPIEMassB0Sph`` with a tiny core and a truncation ``rs``, keeping ``b0`` (the angular lens strength) as
the mass parameter. The main and extra galaxies stay Isothermal.

**The linking across parameterizations.** Both the SIE ``einstein_radius`` and the dPIE ``b0`` are the
*angular lens strength* in arcsec, both ``~ sigma^2``, equal in the SIS limit. So a dPIE scaling
galaxy's ``b0`` anchors *directly* to the SIE BGC's ``einstein_radius`` with the identical relation — no
conversion:

    b0_i = einstein_radius_BGC * (L / L_BGC) ** 0.5,     rs_i = rs_ref * (L / L_BGC) ** 0.5

The truncation ``rs`` is a *separate* parameter (the outer fall-off); it does not enter the anchoring.
Caveat: with finite ``rs`` the actual central deflection is a few percent below ``b0`` (truncation
removes outer mass) — negligible for subdominant members; you anchor on ``b0`` regardless.
"""

# Halo (NFW), main and extra galaxies (SIE) — identical to Section 2.
halo_mass = af.Model(al.mp.NFW)
halo_mass.centre = (0.0, 0.0)
halo_mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=-0.5, upper_limit=0.5
)  # [FREE]
halo_mass.ell_comps.ell_comps_1 = af.UniformPrior(
    lower_limit=-0.5, upper_limit=0.5
)  # [FREE]
halo_mass.kappa_s = af.UniformPrior(lower_limit=0.05, upper_limit=0.5)  # [FREE]
halo_mass.scale_radius = af.UniformPrior(lower_limit=10.0, upper_limit=60.0)  # [FREE]
cluster_halo_trunc = af.Model(al.Galaxy, redshift=redshift_lens, mass=halo_mass)

main_galaxies_trunc = []
for centre in main_centres:
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre  # [FIXED]
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)  # [FREE]
    main_galaxies_trunc.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

einstein_radius_bgc = main_galaxies_trunc[
    bgc_index
].mass.einstein_radius  # SIE Einstein radius = anchor

extra_galaxies_trunc = []
for centre, magnitude in zip(extra_centres, extra_magnitudes):
    ratio = luminosity_ratio(magnitude)
    upper = min(2.0 * (3.0 / 1.0**0.5) * ratio**0.5, 5.0)
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre  # [FIXED]
    mass.einstein_radius = af.UniformPrior(
        lower_limit=0.0, upper_limit=upper
    )  # [FREE, BOUNDED]
    extra_galaxies_trunc.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

# Scaling galaxies -> dPIE (truncated), b0 anchored to the SIE BGC's einstein_radius.
rs_ref = (
    30.0  # [FIXED] reference truncation (arcsec); scales ~ L^0.5 like Lenstool's r_cut
)
scaling_galaxies_trunc = []
for centre, magnitude in zip(scaling_centres, scaling_magnitudes):
    ratio = luminosity_ratio(magnitude)
    mass = af.Model(al.mp.dPIEMassB0Sph)
    mass.centre = centre  # [FIXED]
    mass.ra = 0.01  # [FIXED] ~0: a pure truncated isothermal (no core)
    mass.b0 = (
        einstein_radius_bgc * ratio**0.5
    )  # SIE einstein_radius drives dPIE b0 -> NO new parameter
    mass.rs = rs_ref * ratio**0.5  # [FIXED] truncation, scaled with L^0.5
    scaling_galaxies_trunc.append(
        af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
    )

model_truncated = af.Collection(
    cluster_halo=cluster_halo_trunc,
    main_galaxies=af.Collection(main_galaxies_trunc),
    extra_galaxies=af.Collection(extra_galaxies_trunc),
    scaling_galaxies=af.Collection(scaling_galaxies_trunc),
)

print("\n" + "=" * 80)
print(
    "Section 3 — Truncated scaling galaxies (dPIE outskirts, Isothermal main + extra)"
)
print("=" * 80)
print(
    f"Free parameters: {model_truncated.prior_count}  (same 7 as Section 2; only the scaling profile changed)"
)
assert model_truncated.prior_count == 7

# (a) The cross-parameterization linking: a scaling galaxy's b0 = einstein_radius_BGC * (L/L_BGC)^0.5.
instance = model_truncated.instance_from_prior_medians()
er_bgc = instance.main_galaxies[bgc_index].mass.einstein_radius
print(
    f"(a) BGC einstein_radius = {er_bgc:.3f} arcsec (SIE);  dPIE scaling galaxies b0 = einstein_radius_BGC * (L/L_BGC)^0.5:"
)
worst = 0.0
for i, magnitude in enumerate(scaling_magnitudes):
    ratio = luminosity_ratio(magnitude)
    expected = er_bgc * ratio**0.5
    actual = instance.scaling_galaxies[i].mass.b0
    worst = max(worst, abs(actual - expected))
    print(
        f"    scaling {i} (mag {magnitude}):  b0 = {actual:.3f} arcsec  = {ratio**0.5:.3f} x einstein_radius_BGC"
    )
assert worst < 1e-9

# (b) The truncation bites: the deflection dies away at large radius (an SIE would stay flat).
member0 = instance.scaling_galaxies[0].mass
c = scaling_centres[0]
print(
    "(b) scaling galaxy 0 deflection vs radius (truncated dPIE — dies away; an SIE would stay flat):"
)
for R in [1.0, 10.0, 50.0, 200.0]:
    g = al.Grid2DIrregular([[c[0], c[1] + R]])
    a = float(np.max(np.abs(np.asarray(member0.deflections_yx_2d_from(grid=g)))))
    print(f'    R={R:>4.0f}":  alpha = {a:.4f} arcsec')
print("Section 3 checks passed.")

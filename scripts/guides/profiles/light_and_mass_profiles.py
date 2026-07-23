"""
Light-and-Mass Profiles
=======================

This guide is the third and final entry in the `scripts/guides/profiles/` trilogy.  It
covers the *stellar*, *dark-matter*, and *combined light-and-mass* profiles — the ones used
to decompose a lens galaxy into its constituent matter components rather than to model the
total mass with a single parametric profile.

It pairs with:

- `scripts/guides/profiles/light.py` — pure light profiles in `al.lp.*` and friends.
- `scripts/guides/profiles/mass.py` — parametric lensing mass profiles in `al.mp.*`
  (Total, Mass Sheets, Multipoles, Point Mass).

Where `mass.py` shows the *total* mass distribution of a lens galaxy parameterised by a
single power-law / isothermal / dPIE profile, this guide shows the *decomposed* picture:
stellar mass (Sersic, Chameleon, ...) plus dark-matter halo (NFW family).  The `al.lmp.*`
and `al.lmp_linear.*` namespaces tie a galaxy's *visible* light to its *stellar* mass via
a single shared `mass_to_light_ratio` parameter, so one model object produces both an
`image_2d_from` (light) and a `convergence_2d_from` (mass) consistently.

__Contents__

- **Overview & Docs URL:** The three-guide layout and where the canonical API reference lives.
- **All Profiles (Survey):** A high-level catalogue of every stellar / dark / lmp /
  lmp_linear class.
- **Stellar Mass Detailed Example:** `al.mp.Sersic` — pure stellar mass.  Convergence and
  the lensed source image via `Tracer`.
- **Dark Mass Detailed Example:** `al.mp.NFW` — the standard cuspy dark-matter halo.
  Convergence in linear and log10 scales.
- **NFW Variants:** `gNFW`, `cNFW`, `NFWTruncated`, plus the MCR (mass-concentration), Virial-
  mass, and Scatter reparameterisations.  One-liner construction and when to reach for each.
- **Combined Light + Mass Profiles (`al.lmp`):** The headline feature — one Sersic-shaped
  object emits BOTH `image_2d_from` (light) AND `convergence_2d_from` (mass) via a shared
  `mass_to_light_ratio`.
- **Linear Combined Light + Mass (`al.lmp_linear`):** The inversion-aware variant of the
  lmp family; the intensity-via-inversion semantics from `al.lp_linear` carry over.
- **Composing a Decomposed Bulge+Halo Model:** `af.Model` with a stellar Sersic mass + NFW
  halo on the lens galaxy.  The standard decomposed-lens recipe.
- **Model Instance from Decomposed Model:** `instance_from_prior_medians()` -> `Tracer` ->
  lensed image.
- **Remaining Profiles Walkthrough:** Compact `convergence_2d_from` / `image_2d_from` block
  per remaining stellar / dark / lmp / lmp_linear profile.
- **Back-References:** Pointer to `light.py` and `mass.py`.

__Units__

Spatial coordinates are in arc-seconds; convergence is dimensionless; intensity is in
electrons per second; `mass_to_light_ratio` is dimensionless (in the units of the workspace
config).  The `guides/units_and_cosmology.ipynb` guide covers physical-unit conversion.

__Data Structures__

`convergence_2d_from` and `image_2d_from` both return `Array2D`.  Plotting through
`aplt.plot_array` is direct.  See `mass.py` for the deflection-magnitude pattern when you
need to inspect the vector quantities — this guide focuses on convergence and image.

__Docs URL__

The published API reference for these classes lives at:

    https://pyautolens.readthedocs.io/en/latest/api/mass.html

The autosummary on that page is the authoritative list of every public mass-profile class
(stellar and dark are in the same page under their respective sections).  Light-and-mass
combined profiles in `al.lmp.*` and `al.lmp_linear.*` are not yet documented on the
reference page — refer to the source under
`autogalaxy/profiles/light_and_mass_profiles.py` for the full list.
"""
# ENV: full_datasets
# Guides load committed full-resolution FITS; SMALL_DATASETS would
# mismatch the pre-existing 100x100 data shape.

# from autolens import setup_notebook; setup_notebook()

import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__Grid__

To evaluate any quantity on a profile we need a 2D Cartesian grid of (y,x) coordinates.
We build a 100x100 grid here at a 0.05" pixel scale — used by every section below.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,
)

"""
__All Profiles (Survey)__

**PyAutoLens** organises matter-decomposition profiles into four namespaces:

- `al.mp.*` *stellar* mass — Sersic-like profiles parameterised by light shape plus a
  `mass_to_light_ratio` that converts intensity into surface mass density.
- `al.mp.*` *dark-matter* mass — NFW family halos parameterised by `kappa_s` /
  `scale_radius`, or alternatively by halo mass + concentration via the MCR and Virial
  variants.
- `al.lmp.*` *combined light-and-mass* profiles — one object that emits both
  `image_2d_from` (light) and `convergence_2d_from` (mass) via a shared
  `mass_to_light_ratio`.
- `al.lmp_linear.*` — the inversion-aware version of `al.lmp.*`.  Same classes, intensity
  solved by inversion at fit time.

Below we construct one profile from each family with sensible defaults.  No quantity is
evaluated yet — this is the catalogue.
"""
# Stellar mass — pure mass parameterised by light shape * mass_to_light_ratio
stellar_sersic = al.mp.Sersic()
stellar_chameleon = al.mp.Chameleon()
stellar_gaussian = al.mp.Gaussian()

# Dark mass — NFW family
dark_nfw = al.mp.NFW()
dark_gnfw = al.mp.gNFW()
dark_cnfw = al.mp.cNFW()
dark_nfw_truncated = al.mp.NFWTruncatedSph()
dark_nfw_mcr = al.mp.NFWMCRLudlow()
dark_nfw_virial = al.mp.NFWVirialMassConcSph()

# Combined light-and-mass — one object, two outputs
lmp_sersic = al.lmp.Sersic()
lmp_chameleon = al.lmp.Chameleon()
lmp_gaussian = al.lmp.Gaussian()
lmp_gaussian_gradient = al.lmp.GaussianGradient()

# Linear combined light-and-mass — intensity solved by inversion at fit time
lmp_linear_sersic = al.lmp_linear.Sersic()
lmp_linear_gaussian = al.lmp_linear.Gaussian()

"""
Two things worth knowing about this list before we move on:

1. The *stellar* and *combined* profiles use the **same parametric forms** as the light
   profiles (Sersic, Chameleon, Gaussian, ...).  Their `convergence_2d_from` is the
   profile's light shape multiplied by `mass_to_light_ratio` — this is a fundamentally
   different parameterisation from the dark-matter halos, which are governed by halo-mass
   physics (concentration-mass relations, virial parameters).
2. The dark-matter NFW family has *many* variants that are physically equivalent but
   parameterised differently — they trade `kappa_s` / `scale_radius` for `mass_at_200` +
   concentration (via MCR), or for `virial_mass` + concentration (Virial).  Picking the
   right parameterisation matters for prior choice and degeneracy management; the NFW
   Variants section below is a one-page tour of the menagerie.

__Stellar Mass Detailed Example__

`al.mp.Sersic` is the stellar mass profile.  It is parameterised by exactly the same
light-shape arguments as `al.lp.Sersic` (`intensity`, `effective_radius`, `sersic_index`)
plus a `mass_to_light_ratio` that converts the surface brightness profile into a surface
mass density.  Crucially it does NOT emit light — `al.mp.Sersic` only carries mass.

Build a stellar Sersic mass profile and plot its convergence:
"""
stellar_sersic = al.mp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
    mass_to_light_ratio=2.0,
)

aplt.plot_array(
    array=stellar_sersic.convergence_2d_from(grid=grid),
    title="Stellar Sersic Mass Convergence",
)

"""
The convergence map looks exactly like a Sersic image because, geometrically, it is one
— the same Sersic profile, normalised by `mass_to_light_ratio`.

To see what this stellar mass does to a background source, drop it into a `Tracer` with a
source-plane galaxy carrying a small Sersic light:
"""
lens_galaxy = al.Galaxy(redshift=0.5, mass=stellar_sersic)
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.05, 0.05),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.5,
    ),
)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Tracer Image (Stellar Sersic Lens + Sersic Source)",
)

"""
Note that the lens galaxy carries *only* mass — no light is emitted from the lens plane in
the tracer image.  To add lens-galaxy light you would attach a separate `al.lp.*` light
profile to the same `Galaxy`, OR use a `al.lmp.*` profile (covered later in this guide)
which carries both light and mass in a single object.

__Dark Mass Detailed Example__

`al.mp.NFW` is the canonical dark-matter halo — the elliptical Navarro-Frenk-White profile
parameterised by:

- `kappa_s` — the dimensionless characteristic convergence at the scale radius.
- `scale_radius` — the radius (arc-seconds) at which the density slope transitions from
  ~r^-1 (inner cusp) to ~r^-3 (outer fall-off).

This is the "natural" lensing parameterisation; physically equivalent reparameterisations
in terms of halo mass + concentration are covered in the NFW Variants section below.

Build an `al.mp.NFW` and plot its convergence on linear and log10 scales:
"""
dark_nfw = al.mp.NFW(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
    kappa_s=0.1,
    scale_radius=2.0,
)

aplt.plot_array(
    array=dark_nfw.convergence_2d_from(grid=grid),
    title="NFW Convergence",
)

aplt.plot_array(
    array=dark_nfw.convergence_2d_from(grid=grid),
    title="NFW Convergence (log10 — central cusp visible)",
    use_log10=True,
)

"""
The log10 view shows the characteristic central cusp that distinguishes NFW from the cored
profiles (`Isothermal`, `dPIEMass`).  In a real lens model the cusp is what makes dark-
matter mass distinguishable from a steep stellar profile.

__NFW Variants__

The NFW family has several reparameterisations.  All produce identical convergence maps
for matched halo parameters — they differ only in which parameters the *user* specifies
and which are derived.  Picking the right one is mostly about prior choice and parameter
degeneracies.

- **Generalised NFW (`al.mp.gNFW`)** — adds a free `inner_slope` parameter.  Reduces to
  `NFW` when `inner_slope=1.0`.  Useful when you want to test whether the inner cusp is
  steeper or shallower than the NFW prediction.
- **Cored NFW (`al.mp.cNFW`)** — adds a `core_radius` parameter that flattens the central
  cusp.  Useful for self-interacting dark matter scenarios.
- **Truncated NFW (`al.mp.NFWTruncatedSph`)** — adds a `truncation_radius` beyond which
  the density is forced to zero.  Useful for satellite halos with stripped outer envelopes.
- **MCR variants (`NFWMCR*` / `cNFWMCR*` / `gNFWMCR*`)** — replace `kappa_s` /
  `scale_radius` with `mass_at_200` and use a fixed concentration-mass relation (Duffy
  2008 or Ludlow 2016) to derive the concentration.  Removes one free parameter at the
  cost of assuming the concentration follows the relation.
- **Scatter variants (`NFWMCRScatter*`)** — the Ludlow MCR plus a `scatter_sigma` knob
  that adds Gaussian scatter on the log-concentration.  Useful when you want to *fit* the
  concentration but anchor it to the Ludlow relation with finite tolerance.
- **Virial-mass variants (`*VirialMassConc*`)** — parameterise the halo by its virial mass
  and concentration directly, instead of the `mass_at_200` convention.  Choose whichever
  matches the units used in your cosmology pipeline.

One-liner construction of the major variants:
"""
nfw_gnfw = al.mp.gNFW(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
    kappa_s=0.1,
    inner_slope=1.2,
    scale_radius=2.0,
)

nfw_cnfw = al.mp.cNFW(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
    kappa_s=0.1,
    scale_radius=2.0,
    core_radius=0.2,
)

nfw_truncated = al.mp.NFWTruncatedSph(
    centre=(0.0, 0.0),
    kappa_s=0.1,
    scale_radius=2.0,
    truncation_radius=4.0,
)

nfw_mcr = al.mp.NFWMCRLudlowSph(
    centre=(0.0, 0.0),
    mass_at_200=1.0e12,
    redshift_object=0.5,
    redshift_source=1.0,
)

nfw_mcr_scatter = al.mp.NFWMCRScatterLudlowSph(
    centre=(0.0, 0.0),
    mass_at_200=1.0e12,
    scatter_sigma=0.1,
    redshift_object=0.5,
    redshift_source=1.0,
)

nfw_virial = al.mp.NFWVirialMassConcSph(
    centre=(0.0, 0.0),
    virial_mass=1.0e12,
    concentration=10.0,
    virial_overdens=200.0,
    redshift_object=0.5,
    redshift_source=1.0,
)

"""
For comparison, here is the cored variant's convergence — note the flat central plateau:
"""
aplt.plot_array(
    array=nfw_cnfw.convergence_2d_from(grid=grid),
    title="cNFW Convergence (cored central region)",
    use_log10=True,
)

"""
And the MCR variant produced from a halo mass rather than `kappa_s` / `scale_radius`:
"""
aplt.plot_array(
    array=nfw_mcr.convergence_2d_from(grid=grid),
    title="NFWMCRLudlowSph Convergence (mass=1e12, concentration via Ludlow)",
    use_log10=True,
)

"""
__Combined Light + Mass Profiles (`al.lmp`)__

The `al.lmp.*` namespace is the headline feature of this guide.  Each `al.lmp` class
represents a galaxy component that emits BOTH light AND mass — a single object with one
set of parameters governing both `image_2d_from` and `convergence_2d_from` via a shared
`mass_to_light_ratio`.

This is the standard way to model a lens galaxy's bulge: one `al.lmp.Sersic` carries the
visible Sersic light and contributes the stellar mass component to the deflection, tied
together by `mass_to_light_ratio`.  Pair it with a separate `al.mp.NFW` (or one of the
dark variants above) for the dark-matter halo and you have the canonical bulge+halo
decomposition.

Build one `al.lmp.Sersic` and plot both its image AND its convergence:
"""
lmp_sersic = al.lmp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
    mass_to_light_ratio=2.0,
)

aplt.plot_array(
    array=lmp_sersic.image_2d_from(grid=grid),
    title="lmp.Sersic Image (light side)",
)

aplt.plot_array(
    array=lmp_sersic.convergence_2d_from(grid=grid),
    title="lmp.Sersic Convergence (mass side, mass_to_light_ratio=2.0)",
)

"""
The two maps are *the same Sersic profile*, with the mass map scaled by
`mass_to_light_ratio`.  This is the geometric guarantee of the `al.lmp.*` family — the
stellar light and stellar mass cannot diverge from each other except through this one
parameter.

`al.lmp.SersicGradient` and `al.lmp.GaussianGradient` extend this with a *radial gradient*
in the mass-to-light ratio (one extra parameter), useful when the inner regions of a
galaxy have a systematically different M/L from the outskirts.

__Linear Combined Light + Mass (`al.lmp_linear`)__

`al.lmp_linear.*` mirrors `al.lmp.*` class-for-class.  The only difference is that the
`intensity` parameter is solved analytically via the linear inversion at each likelihood
evaluation (the same mechanism that `al.lp_linear.*` uses for pure light profiles).  This
means a lens model with a `al.lmp_linear.Sersic` bulge has one fewer free non-linear
parameter than the corresponding `al.lmp.Sersic` model — the bulge intensity is no longer
sampled by the search.

The construction API is identical to `al.lmp.*`:
"""
lmp_linear_sersic = al.lmp_linear.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
    mass_to_light_ratio=2.0,
)

"""
The full workflow for linear profiles in lens models — including how the inversion
combines light and mass contributions — is documented in:

    scripts/imaging/features/linear_light_profiles/

That folder is the canonical reference; this guide stops at the construction API.

__Composing a Decomposed Bulge+Halo Model__

The standard decomposed lens-mass model attaches:

- one `al.mp.Sersic` (or `al.lmp.Sersic`) as the lens-galaxy stellar component, AND
- one `al.mp.NFW` (or another dark variant) as the lens-galaxy dark-matter halo,

so the total convergence is the sum of stellar + dark contributions.  Compose this via
`af.Model` exactly as you would for a single-profile lens — `Galaxy` accepts arbitrary
kwargs for its mass components.
"""
lens_bulge_mass_model = af.Model(al.mp.Sersic)
lens_dark_model = af.Model(al.mp.NFW)
source_bulge_model = af.Model(al.lp.Sersic)

lens_galaxy_model = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=lens_bulge_mass_model,
    dark=lens_dark_model,
)
source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_model)
model = af.Collection(
    galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
)

print(model.info)

"""
Printing `model.info` shows the priors-and-defaults summary.  Notice the lens galaxy has
*two* mass components (`bulge` and `dark`) whose convergences will be summed during the
fit.  This is how a decomposed lens model differs from the single-profile `Isothermal`
lens shown in `mass.py` — the model is more flexible but also higher-dimensional.

Swapping `al.mp.Sersic` for `al.lmp.Sersic` on the lens bulge would additionally model the
lens-galaxy *light* through the same object — useful when the lens galaxy is visible in
the data and you want to tie its light to its stellar mass.

Full end-to-end lens fits with this model live under `scripts/imaging/features/` and the
SLaM pipeline guides; this section just shows the model spec.

__Model Instance from Decomposed Model__

To realise an instance from the model's prior medians and visualise it, call
`instance_from_prior_medians()` on the collection and feed the result into a `Tracer`:
"""
model_instance = model.instance_from_prior_medians()

tracer = al.Tracer(
    galaxies=[model_instance.galaxies.lens, model_instance.galaxies.source]
)

aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Tracer Image from Decomposed Model (Stellar + NFW)",
)

aplt.plot_array(
    array=tracer.convergence_2d_from(grid=grid),
    title="Tracer Total Convergence (Stellar + NFW summed)",
)

"""
The total convergence map shows the summed stellar + dark contributions — exactly the
lensing mass distribution that the search would optimise.

After a fit completes, `result.max_log_likelihood_tracer` returns the same shape of object
with the prior medians replaced by the fitted parameters.  See
`scripts/guides/results/start_here.py` for the full results-introspection guide.

__Remaining Profiles Walkthrough__

We have shown the full flow for `al.mp.Sersic` (stellar), `al.mp.NFW` (dark), and
`al.lmp.Sersic` (combined).  Every remaining stellar / dark / lmp / lmp_linear profile uses
the same API — only the constructor parameters change.

The compact tour below builds each remaining profile with sensible defaults.  Stellar and
dark profiles show `convergence_2d_from`; `lmp` profiles show both `image_2d_from` and
`convergence_2d_from` so the dual-output property is visible per class.

Stellar mass:
"""
aplt.plot_array(
    array=al.mp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        radius_break=0.05,
        gamma=0.25,
        alpha=3.0,
        mass_to_light_ratio=2.0,
    ).convergence_2d_from(grid=grid),
    title="Stellar SersicCore Convergence",
)

aplt.plot_array(
    array=al.mp.SersicGradient(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.3,
    ).convergence_2d_from(grid=grid),
    title="Stellar SersicGradient Convergence",
)

aplt.plot_array(
    array=al.mp.DevVaucouleurs(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    ).convergence_2d_from(grid=grid),
    title="Stellar DevVaucouleurs Convergence",
)

aplt.plot_array(
    array=al.mp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.5,
        effective_radius=1.6,
        mass_to_light_ratio=2.0,
    ).convergence_2d_from(grid=grid),
    title="Stellar Exponential Convergence",
)

aplt.plot_array(
    array=al.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        sigma=0.4,
        mass_to_light_ratio=2.0,
    ).convergence_2d_from(grid=grid),
    title="Stellar Gaussian Convergence",
)

aplt.plot_array(
    array=al.mp.GaussianGradient(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        sigma=0.4,
        mass_to_light_ratio_base=2.0,
        mass_to_light_gradient=0.2,
        mass_to_light_radius=0.5,
    ).convergence_2d_from(grid=grid),
    title="Stellar GaussianGradient Convergence",
)

aplt.plot_array(
    array=al.mp.Chameleon(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.0,
        core_radius_0=0.05,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    ).convergence_2d_from(grid=grid),
    title="Stellar Chameleon Convergence",
)

"""
Dark mass — the variant menagerie, one block each:
"""
aplt.plot_array(
    array=al.mp.gNFW(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        kappa_s=0.1,
        inner_slope=1.2,
        scale_radius=2.0,
    ).convergence_2d_from(grid=grid),
    title="gNFW Convergence (inner_slope=1.2)",
    use_log10=True,
)

aplt.plot_array(
    array=al.mp.NFWTruncatedSph(
        centre=(0.0, 0.0),
        kappa_s=0.1,
        scale_radius=2.0,
        truncation_radius=4.0,
    ).convergence_2d_from(grid=grid),
    title="NFWTruncatedSph Convergence",
    use_log10=True,
)

aplt.plot_array(
    array=al.mp.NFWMCRDuffySph(
        centre=(0.0, 0.0),
        mass_at_200=1.0e12,
        redshift_object=0.5,
        redshift_source=1.0,
    ).convergence_2d_from(grid=grid),
    title="NFWMCRDuffySph Convergence (Duffy 2008 MCR)",
    use_log10=True,
)

aplt.plot_array(
    array=al.mp.NFWMCRScatterLudlowSph(
        centre=(0.0, 0.0),
        mass_at_200=1.0e12,
        scatter_sigma=0.1,
        redshift_object=0.5,
        redshift_source=1.0,
    ).convergence_2d_from(grid=grid),
    title="NFWMCRScatterLudlowSph Convergence (Ludlow + sigma=0.1)",
    use_log10=True,
)

aplt.plot_array(
    array=al.mp.NFWVirialMassConcSph(
        centre=(0.0, 0.0),
        virial_mass=1.0e12,
        concentration=10.0,
        virial_overdens=200.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ).convergence_2d_from(grid=grid),
    title="NFWVirialMassConcSph Convergence",
    use_log10=True,
)

aplt.plot_array(
    array=al.mp.cNFWMCRLudlowSph(
        centre=(0.0, 0.0),
        mass_at_200=1.0e12,
        f_c=0.05,
        redshift_object=0.5,
        redshift_source=1.0,
    ).convergence_2d_from(grid=grid),
    title="cNFWMCRLudlowSph Convergence (cored + Ludlow MCR)",
    use_log10=True,
)

"""
Combined light-and-mass — every remaining `al.lmp` profile, with both image and convergence
shown side-by-side via stacked `plot_array` calls:
"""
lmp_chameleon = al.lmp.Chameleon(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    core_radius_0=0.05,
    core_radius_1=0.3,
    mass_to_light_ratio=2.0,
)
aplt.plot_array(
    array=lmp_chameleon.image_2d_from(grid=grid), title="lmp.Chameleon Image"
)
aplt.plot_array(
    array=lmp_chameleon.convergence_2d_from(grid=grid),
    title="lmp.Chameleon Convergence",
)

lmp_devvaucouleurs = al.lmp.DevVaucouleurs(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    mass_to_light_ratio=2.0,
)
aplt.plot_array(
    array=lmp_devvaucouleurs.image_2d_from(grid=grid),
    title="lmp.DevVaucouleurs Image",
)
aplt.plot_array(
    array=lmp_devvaucouleurs.convergence_2d_from(grid=grid),
    title="lmp.DevVaucouleurs Convergence",
)

lmp_exponential = al.lmp.Exponential(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=0.5,
    effective_radius=1.6,
    mass_to_light_ratio=2.0,
)
aplt.plot_array(
    array=lmp_exponential.image_2d_from(grid=grid),
    title="lmp.Exponential Image",
)
aplt.plot_array(
    array=lmp_exponential.convergence_2d_from(grid=grid),
    title="lmp.Exponential Convergence",
)

lmp_gaussian = al.lmp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    sigma=0.4,
    mass_to_light_ratio=2.0,
)
aplt.plot_array(array=lmp_gaussian.image_2d_from(grid=grid), title="lmp.Gaussian Image")
aplt.plot_array(
    array=lmp_gaussian.convergence_2d_from(grid=grid),
    title="lmp.Gaussian Convergence",
)

lmp_sersic_gradient = al.lmp.SersicGradient(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
    mass_to_light_ratio=2.0,
    mass_to_light_gradient=0.3,
)
aplt.plot_array(
    array=lmp_sersic_gradient.image_2d_from(grid=grid),
    title="lmp.SersicGradient Image",
)
aplt.plot_array(
    array=lmp_sersic_gradient.convergence_2d_from(grid=grid),
    title="lmp.SersicGradient Convergence",
)

"""
The spherical variants (`SersicSph`, `DevVaucouleursSph`, `ExponentialSph`, `ChameleonSph`,
etc., across all three namespaces) are constructed identically with the `ell_comps`
argument removed.  Each looks like a rotationally symmetric version of its elliptical
counterpart.

`al.lmp_linear.*` shares the construction API with `al.lmp.*` — instantiate any
`al.lmp_linear.X` exactly as you would `al.lmp.X`.  Their distinct behaviour only shows up
inside a model fit, when the inversion solves for `intensity` rather than treating it as
a free non-linear parameter.

__Back-References__

This guide completes the three-guide tour of the `scripts/guides/profiles/` folder:

- `scripts/guides/profiles/light.py` — pure light profiles, the `al.lp.*` /
  `al.lp_linear.*` / `al.lp_operated.*` / `al.lp_basis.*` namespaces.
- `scripts/guides/profiles/mass.py` — parametric lensing mass profiles, the *Total* /
  *Mass Sheets* / *Multipoles* / *Point Mass* families of `al.mp.*`.
- `scripts/guides/profiles/light_and_mass_profiles.py` — this guide; stellar / dark / lmp /
  lmp_linear matter decomposition.

For the next step into actual lens modelling, see `scripts/imaging/modeling/start_here.py`
(strong-lens fit end-to-end) and the topic-specific feature packages under
`scripts/imaging/features/`.  The SLaM pipeline guide (`scripts/guides/modeling/slam_start_here.py`)
shows the production decomposed-lens workflow built on these profiles.
"""

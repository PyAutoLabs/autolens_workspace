"""
Mass Profiles
=============

This guide is the single-page tour of every lensing mass profile available in **PyAutoLens**
(re-exported from **PyAutoGalaxy**): how to construct each one, how to evaluate its
convergence and deflections on a grid, how to compose it into a model, and how to pull an
instance back out of that model.

It is the companion to `scripts/guides/profiles/light.py` — the section flow, prose style
and "detailed example then walkthrough" rhythm intentionally mirror that guide so the two
can be read as a pair.  Where `light.py` shows `image_2d_from`, this guide shows
`convergence_2d_from` and `deflections_yx_2d_from`; mass profiles do not produce images of
their own (a mass profile only deflects light from a *source* through ray-tracing — that is
what the `Tracer` is for, and we use it for the detailed example).

This guide covers the *lensing* mass profiles — Total, Mass Sheets, Multipoles, and Point
Mass.  The *stellar* (`al.mp.Sersic`, `al.mp.Chameleon`, ...) and *dark-matter* (NFW family)
mass profiles, along with the combined light+mass profiles in `al.lmp.*` / `al.lmp_linear.*`,
get their own dedicated guide at `scripts/guides/profiles/light_and_mass_profiles.py` where
the stellar-plus-dark decomposition story is told properly.

__Contents__

- **Overview & Docs URL:** Where the canonical API reference lives.
- **All Mass Profiles (Survey):** A high-level run-through of every profile in `al.mp.*`
  (Total / Mass Sheets / Multipoles / Point Mass), without yet evaluating any quantities.
- **Detailed Example: Isothermal:** Build a `Grid2D`, instantiate `al.mp.Isothermal`, plot
  convergence, potential, deflection-magnitude, and the lensed source image produced when
  the isothermal mass is dropped into a `Tracer`.
- **Mass Sheets:** `ExternalShear`, `MassSheet`, `ExternalPotential` — global perturbations
  rather than parametric matter distributions.
- **Point Mass:** `PointMass`, `SMBH`, `SMBHBinary` — delta-function-like mass for
  microlensing and supermassive black hole lensing.
- **Mass Profile in a Model:** Wrap a profile in `af.Model`, compose lens + source via
  `af.Collection`, inspect the model info.
- **Model Instance from Mass Profile:** Realise an instance from the model's prior medians
  and drop it into a `Tracer` to produce a lensed image.
- **Multipole Mass Profile:** `PowerLawMultipole` — m=3 / m=4 Fourier perturbation on the
  power-law convergence, the lensing counterpart of `lp.SersicMultipole`.
- **Remaining Profiles Walkthrough:** Compact `convergence_2d_from` block for every total
  profile not yet shown.
- **Follow-Up:** Pointer to `light_and_mass_profiles.py` for stellar / dark / lmp coverage.

__Units__

Spatial coordinates are in arc-seconds, mass quantities are dimensionless (convergence) or
in their natural lensing units (potential, deflection angles in arc-seconds).  The
`guides/units_and_cosmology.ipynb` guide covers conversion to physical units.

__Data Structures__

`convergence_2d_from` and `potential_2d_from` return `Array2D`.  `deflections_yx_2d_from`
returns `VectorYX2D` (a 2D vector field).  `aplt.plot_array` accepts `Array2D` directly;
to plot a vector field we either pull out the y / x components individually or compute the
magnitude and wrap it in an `Array2D`.  This guide uses the magnitude approach for the
detailed example.

__Docs URL__

The published API reference for these classes lives at:

    https://pyautolens.readthedocs.io/en/latest/api/mass.html

The autosummary on that page is the authoritative list of every public mass-profile class.
Note that the reference uses the `ag.mp` namespace label because the classes are defined in
PyAutoGalaxy and re-exported here as `al.mp`.
"""

# from autolens import setup_notebook; setup_notebook()

import numpy as np

import autofit as af
import autoarray as aa
import autolens as al
import autolens.plot as aplt


"""
__Grid__

To evaluate any quantity on a mass profile we need a 2D Cartesian grid of (y,x) coordinates.
We build a 100x100 grid here at a 0.05" pixel scale — used by every section below.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,
)

"""
__All Mass Profiles (Survey)__

**PyAutoLens** groups mass profiles into four families relevant to lensing:

- `al.mp.*` total profiles — the canonical parametric lens mass distributions
  (`Isothermal`, `PowerLaw`, their cored / broken / dPIE variants).
- `al.mp.*` mass sheets — `ExternalShear`, `MassSheet`, `ExternalPotential`.  Global
  perturbations representing line-of-sight contributions.
- `al.mp.*` multipoles — `PowerLawMultipole`, the m=3 / m=4 angular extension of the power-
  law family.
- `al.mp.*` point masses — `PointMass`, `SMBH`, `SMBHBinary` — delta-function-like sources
  for microlensing and central black holes.

The *stellar* and *dark-matter* mass profiles (e.g. `al.mp.Sersic`, the NFW family) and the
combined `al.lmp.*` / `al.lmp_linear.*` light-and-mass namespaces are covered in
`scripts/guides/profiles/light_and_mass_profiles.py`.  They share the same API but tell a
different story (stellar-plus-dark decomposition).

Below we construct each lensing profile with default parameters.  No quantity is evaluated
yet — that comes in the next section.  The goal here is purely a catalogue of what is
available.
"""
# Power-law family (Isothermal is PowerLaw with slope = 2)
isothermal = al.mp.Isothermal()
isothermal_sph = al.mp.IsothermalSph()
isothermal_core = al.mp.IsothermalCore()
isothermal_core_sph = al.mp.IsothermalCoreSph()
power_law = al.mp.PowerLaw()
power_law_sph = al.mp.PowerLawSph()
power_law_core = al.mp.PowerLawCore()
power_law_core_sph = al.mp.PowerLawCoreSph()
power_law_broken = al.mp.PowerLawBroken()
power_law_broken_sph = al.mp.PowerLawBrokenSph()

# Pseudo-isothermal family. The default dPIEMass / dPIEMassSph are parameterized in
# Lenstool's native convention (ellipticity, angle_pos, sigma = fiducial v_disp in km/s,
# r_core, r_cut, plus the redshifts entering the D_LS/D_S normalization) — the same numbers
# that appear in published cluster papers' results tables. The internal (ra, rs, b0)
# parameterization is the non-standard dPIEMassB0 / dPIEMassB0Sph.
dpie_mass = al.mp.dPIEMass(ellipticity=0.1)
dpie_mass_sph = al.mp.dPIEMassSph()
dpie_mass_b0 = al.mp.dPIEMassB0(ell_comps=(0.05, 0.0))
# Note: PIEMass with ell_comps=(0,0) triggers a divide-by-zero in the complex-plane
# formula; we use a small ellipticity here so the survey constructions succeed cleanly.
pie_mass = al.mp.PIEMass(ell_comps=(0.05, 0.0))
dpie_potential = al.mp.dPIEPotential()
dpie_potential_sph = al.mp.dPIEPotentialSph()

# Mass sheets — global perturbations
external_shear = al.mp.ExternalShear()
mass_sheet = al.mp.MassSheet()
external_potential = al.mp.ExternalPotential()

# Multipole — power-law angular extension
power_law_multipole = al.mp.PowerLawMultipole()

# Point masses — microlensing and supermassive black holes
point_mass = al.mp.PointMass()
smbh = al.mp.SMBH()
smbh_binary = al.mp.SMBHBinary()

"""
Two things worth knowing about this list before we move on:

1. Every elliptical lensing profile (e.g. `Isothermal`, `PowerLaw`, `PowerLawCore`) has a
   spherical sibling whose name ends in `Sph` (e.g. `IsothermalSph`).  The spherical variant
   fixes the ellipticity components `ell_comps` to `(0, 0)`, which is useful when you want
   to model a circular halo and avoid two redundant parameters in the non-linear search.
2. The `PowerLawMultipole` variant only exists as an *elliptical* profile — the m=3 / m=4
   perturbations are angular distortions and are not meaningful without an underlying
   reference frame, exactly like the light multipoles in `light.py`.

We now move on to seeing what these profiles actually produce when evaluated on a grid.

__Detailed Example: Isothermal__

The `Isothermal` profile is the canonical strong-lens mass profile — equivalent to the
elliptical power-law with `slope = 2.0`.  Its three parameters are:

- `centre` — the (y, x) arc-second coordinate of the profile centre.
- `ell_comps` — the two ellipticity components `(e1, e2)`.  Use
  `al.convert.ell_comps_from(axis_ratio=..., angle=...)` to convert from human-friendly
  axis ratio and position angle.
- `einstein_radius` — the Einstein radius in arc-seconds.  This sets the strength of the
  lens and the size of the Einstein ring for an axisymmetric source on-axis.

Build an isothermal and evaluate every standard lensing quantity on our grid:
"""
isothermal = al.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    einstein_radius=1.6,
)

"""
First the *convergence* — the projected surface mass density in units of the critical
surface density.  This is the simplest visualisation of a mass profile and the most
analogous to the `image_2d_from` plot we used throughout `light.py`.
"""
aplt.plot_array(
    array=isothermal.convergence_2d_from(grid=grid),
    title="Isothermal Convergence",
)

"""
Next the *lensing potential* — the 2D field whose gradient gives the deflection angles.
We plot it in log10 to make the central cusp visible.
"""
aplt.plot_array(
    array=isothermal.potential_2d_from(grid=grid),
    title="Isothermal Potential (log10)",
    use_log10=True,
)

"""
The *deflection angles* are a vector field rather than a scalar, so `deflections_yx_2d_from`
returns a `VectorYX2D` rather than an `Array2D`.  `aplt.plot_array` does not accept vectors
directly; the conventional pattern in this workspace is to either plot the y / x components
separately or compute the magnitude.  Below we use the magnitude — a single map showing
"how much light is bent here", which is the most useful single-figure summary.
"""
deflections = isothermal.deflections_yx_2d_from(grid=grid)
deflection_magnitude = aa.Array2D(
    values=np.hypot(deflections.slim[:, 0], deflections.slim[:, 1]),
    mask=grid.mask,
)
aplt.plot_array(
    array=deflection_magnitude,
    title="Isothermal Deflection Magnitude",
)

"""
None of these maps show what the isothermal mass actually *does* to a background source —
for that we need a `Tracer`.  A `Tracer` groups galaxies by redshift plane, ray-traces the
grid through every plane, and combines the resulting emission into a single observed image.

Below we build a two-plane lens system: a lens galaxy at `z=0.5` carrying our isothermal
mass profile, and a source galaxy at `z=1.0` with a small Sersic light profile.  The
`tracer.image_2d_from` call produces the lensed image you would see at the telescope.
"""
lens_galaxy = al.Galaxy(redshift=0.5, mass=isothermal)
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
    title="Tracer Image (Isothermal Lens + Sersic Source)",
)

"""
The arc / ring you see is the source's Sersic light, deflected by the isothermal mass into
the characteristic strong-lens morphology.  The same `image_2d_from(grid=grid)` call exists
on every `Tracer` regardless of which mass profile is on the lens galaxy — every section
below is a small variation on this one, swapping the lens-galaxy mass for a different
profile.

__Mass Sheets__

Mass sheets are *global* perturbations rather than localised mass distributions.  They show
up in strong-lens models to capture line-of-sight contributions: nearby group / cluster
members, large-scale structure, and any other mass that is not part of the primary lens but
still distorts the light path.

Three mass-sheet profiles are available:

- `al.mp.ExternalShear(gamma_1, gamma_2)` — a constant shear with two components.  The most
  common line-of-sight correction; you will see this routinely added to lens-mass models.
- `al.mp.MassSheet(centre, kappa)` — a uniform convergence (positive or negative).  Useful
  for representing diffuse line-of-sight mass.
- `al.mp.ExternalPotential(centre, gamma_1, gamma_2, tau_1, tau_2, delta_1, delta_2)` — the
  Powell et al. 2022 generalisation of external shear that adds the next-order line-of-sight
  terms.  Reduces to `ExternalShear` when `tau_*` and `delta_*` are zero.
"""
external_shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.03)
deflections = external_shear.deflections_yx_2d_from(grid=grid)
aplt.plot_array(
    array=aa.Array2D(
        values=np.hypot(deflections.slim[:, 0], deflections.slim[:, 1]),
        mask=grid.mask,
    ),
    title="ExternalShear Deflection Magnitude",
)

mass_sheet = al.mp.MassSheet(centre=(0.0, 0.0), kappa=0.1)
aplt.plot_array(
    array=mass_sheet.convergence_2d_from(grid=grid),
    title="MassSheet Convergence (uniform kappa=0.1)",
)

external_potential = al.mp.ExternalPotential(
    centre=(0.0, 0.0),
    gamma_1=0.04,
    gamma_2=0.02,
    tau_1=0.005,
    tau_2=0.0,
    delta_1=0.0,
    delta_2=0.003,
)
deflections = external_potential.deflections_yx_2d_from(grid=grid)
aplt.plot_array(
    array=aa.Array2D(
        values=np.hypot(deflections.slim[:, 0], deflections.slim[:, 1]),
        mask=grid.mask,
    ),
    title="ExternalPotential Deflection Magnitude",
)

"""
A mass sheet alone produces no observable lensing — its deflections are uniform (constant)
across the image and so are absorbed into the global astrometric solution.  Mass sheets only
become visible *in combination* with another mass profile; their job is to perturb the
isothermal or power-law deflections at the few-percent level.

__Point Mass__

Point masses are delta-function-like mass distributions.  Three flavours exist:

- `al.mp.PointMass(centre, einstein_radius)` — a single point mass parameterised by its
  Einstein radius.  Used for microlensing.
- `al.mp.SMBH(centre, mass, redshift_object, redshift_source)` — a supermassive black hole
  parameterised by mass (in solar masses); internally converts to a `PointMass` Einstein
  radius using the input redshifts.
- `al.mp.SMBHBinary(centre, separation, angle_binary, mass, mass_ratio, ...)` — two SMBHs
  separated by a tunable angle and separation.  Used for binary SMBH lensing.

`PointMass.convergence_2d_from` returns a raw numpy array rather than the `Array2D` that
elliptical profiles return — a library quirk.  We wrap the point mass in a `Galaxy` and plot
the galaxy's convergence map instead, which goes through the standard `Array2D` path and is
also the natural usage pattern for a lensing model anyway.
"""
point_mass = al.mp.PointMass(centre=(0.0, 0.0), einstein_radius=0.3)
point_galaxy = al.Galaxy(redshift=0.5, mass=point_mass)

aplt.plot_array(
    array=point_galaxy.convergence_2d_from(grid=grid),
    title="PointMass Convergence (delta-function-like)",
)

smbh = al.mp.SMBH(
    centre=(0.0, 0.0),
    mass=1.0e9,
    redshift_object=0.5,
    redshift_source=1.0,
)
smbh_galaxy = al.Galaxy(redshift=0.5, mass=smbh)
aplt.plot_array(
    array=smbh_galaxy.convergence_2d_from(grid=grid),
    title="SMBH Convergence (M = 1e9 M_sun)",
)

smbh_binary = al.mp.SMBHBinary(
    centre=(0.0, 0.0),
    separation=0.4,
    angle_binary=45.0,
    mass=2.0e9,
    mass_ratio=0.5,
    redshift_object=0.5,
    redshift_source=1.0,
)
smbh_binary_galaxy = al.Galaxy(redshift=0.5, mass=smbh_binary)
aplt.plot_array(
    array=smbh_binary_galaxy.convergence_2d_from(grid=grid),
    title="SMBHBinary Convergence",
)

"""
__Mass Profile in a Model__

So far we have been instantiating mass profiles with concrete parameter values.  When
fitting a real strong-lens dataset we instead build a *model* of the mass profile and let
the non-linear search find the best-fit parameters.  This is what `af.Model` is for.

For a strong-lens system the model typically contains two `Galaxy` objects on different
planes: a foreground lens (light + mass) and a background source (light only).  In this
section we focus on the *mass* side of that picture and show how an isothermal mass profile
plugs in to a lens galaxy that has only mass (no light) — the simplest case.
"""
lens_mass_model = af.Model(al.mp.Isothermal)
source_bulge_model = af.Model(al.lp.Sersic)

"""
The `af.Model` wraps the profile class.  Every constructor argument with a numerical default
becomes a *prior* — by default the priors are sensible distributions for each parameter
(see the autogalaxy config for the configured ranges).

You can override priors before fitting:
"""
lens_mass_model.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)

"""
Wrap each profile in a `Galaxy` at its own redshift and assemble them in an `af.Collection`.
The `Tracer` is not part of the *model* spec itself; it is what `AnalysisImaging` builds out
of the realised instance at fit time.
"""
lens_galaxy_model = af.Model(al.Galaxy, redshift=0.5, mass=lens_mass_model)
source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_model)
model = af.Collection(
    galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)
)

print(model.info)

"""
Printing `model.info` shows the full priors-and-defaults summary — useful before kicking
off a long fit to confirm the model is shaped the way you expect.

The model API is the same for **every** mass profile in this guide — swap
`al.mp.Isothermal` for `al.mp.PowerLaw`, `al.mp.dPIEMass`, `al.mp.PowerLawMultipole`, etc.,
and the rest of the snippet is unchanged.

Full lens-modelling end-to-end examples live in `scripts/imaging/modeling/start_here.py`
and the topic-specific guides under `scripts/imaging/features/`.

__Model Instance from Mass Profile__

A model is a description of *possible* profiles.  To get an actual profile back out — for
example to plot what the prior medians look like before running a fit — call
`instance_from_prior_medians()`:
"""
lens_mass_instance = lens_mass_model.instance_from_prior_medians()
print(type(lens_mass_instance))  # autogalaxy.profiles.mass.total.isothermal.Isothermal

aplt.plot_array(
    array=lens_mass_instance.convergence_2d_from(grid=grid),
    title="Isothermal Instance Convergence",
)

"""
The instance returned is a real `al.mp.Isothermal` — the same class we constructed by hand
in the detailed example above — and supports the full mass-profile API.

The same flow works at the full-model level.  We realise an instance of the lens-and-source
collection and drop the resulting galaxies into a `Tracer`, which combines the lens mass
deflections with the source light to produce the lensed image.
"""
model_instance = model.instance_from_prior_medians()

tracer = al.Tracer(
    galaxies=[model_instance.galaxies.lens, model_instance.galaxies.source]
)

aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Tracer Image from Model Instance",
)

"""
After a fit completes, `result.max_log_likelihood_tracer` returns the same shape of object,
with the prior medians replaced by the fitted parameter values.  See
`scripts/guides/results/start_here.py` for the full results-introspection guide.

__Multipole Mass Profile__

`PowerLawMultipole` is the lensing counterpart of `lp.SersicMultipole` — an m=3 / m=4
Fourier angular perturbation, here on the eccentric radius of the power-law convergence
field rather than the Sersic intensity field.  The signature is

    al.mp.PowerLawMultipole(
        m=4,                                # multipole order (3 or 4)
        centre=(0.0, 0.0),
        einstein_radius=1.0,
        slope=2.0,
        multipole_comps=(c, s),             # cosine / sine components of the m-th term
    )

A `PowerLawMultipole` produces only the *perturbation* — not the underlying power-law mass
distribution.  The standard usage is therefore to add the multipole alongside a `PowerLaw`
or `Isothermal` profile sharing the same `centre`, `einstein_radius`, and `slope`, so that
the combined convergence reads as a power law with boxy / discy / lopsided distortions.

Build a `PowerLawMultipole` with non-trivial multipole components and plot its convergence
alongside the underlying `PowerLaw` for comparison:
"""
power_law_base = al.mp.PowerLaw(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    einstein_radius=1.6,
    slope=2.0,
)

power_law_multipole_m4 = al.mp.PowerLawMultipole(
    m=4,
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    slope=2.0,
    multipole_comps=(0.05, 0.02),
)

aplt.plot_array(
    array=power_law_base.convergence_2d_from(grid=grid),
    title="PowerLaw Convergence (base profile)",
)

aplt.plot_array(
    array=power_law_multipole_m4.convergence_2d_from(grid=grid),
    title="PowerLawMultipole Convergence (m=4 perturbation only)",
)

"""
Two practical notes on the multipole:

- The multipole profile carries the *perturbation*, not the base mass distribution.  In a
  model you typically wrap both `PowerLaw` and `PowerLawMultipole` on the same lens galaxy
  and link their shared parameters (`centre`, `einstein_radius`, `slope`).
- There is **no spherical (`*Sph`) variant** of `PowerLawMultipole`.  The perturbation is
  an angular distortion measured in the lens reference frame, so it only makes sense for an
  elliptical (or quasi-elliptical) host.

Plugging `PowerLawMultipole` into the `af.Model` / `af.Collection` / `Galaxy` / `Tracer`
pattern shown above works exactly as it did for the plain `Isothermal` — the multipole
components are picked up as priors automatically.

__Remaining Profiles Walkthrough__

We have shown the full `convergence_2d_from` → `af.Model` → `instance` → `Tracer` flow for
the `Isothermal` profile.  Every remaining lensing mass profile uses the **same API** — the
only thing that changes is which parameters appear in the constructor.

The compact tour below builds each remaining total profile with sensible parameter values
and plots its convergence.  When you want to use any of these in a lens model, repeat the
`af.Model(...)` / `af.Collection(...)` / `Tracer(...)` pattern from the previous section.
"""

aplt.plot_array(
    array=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        einstein_radius=1.6,
        slope=2.2,
    ).convergence_2d_from(grid=grid),
    title="PowerLaw Convergence (slope=2.2)",
)

aplt.plot_array(
    array=al.mp.PowerLawCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        einstein_radius=1.6,
        slope=2.0,
        core_radius=0.05,
    ).convergence_2d_from(grid=grid),
    title="PowerLawCore Convergence",
)

aplt.plot_array(
    array=al.mp.PowerLawBroken(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        einstein_radius=1.6,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.5,
    ).convergence_2d_from(grid=grid),
    title="PowerLawBroken Convergence",
)

aplt.plot_array(
    array=al.mp.IsothermalCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        einstein_radius=1.6,
        core_radius=0.05,
    ).convergence_2d_from(grid=grid),
    title="IsothermalCore Convergence",
)

aplt.plot_array(
    array=al.mp.dPIEMass(
        centre=(0.0, 0.0),
        ellipticity=0.1,
        angle_pos=45.0,
        sigma=200.0,
        r_core=0.1,
        r_cut=20.0,
    ).convergence_2d_from(grid=grid),
    title="dPIEMass Convergence",
)

aplt.plot_array(
    array=al.mp.PIEMass(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ra=0.1,
        b0=0.5,
    ).convergence_2d_from(grid=grid),
    title="PIEMass Convergence",
)

aplt.plot_array(
    array=al.mp.dPIEPotential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        ra=0.1,
        rs=2.0,
        b0=1.0,
    ).convergence_2d_from(grid=grid),
    title="dPIEPotential Convergence",
)

"""
The spherical variants (`IsothermalSph`, `PowerLawSph`, `PowerLawCoreSph`,
`PowerLawBrokenSph`, `dPIEMassSph`) are constructed identically with the `ell_comps`
argument removed.  Each looks like a rotationally symmetric version of its elliptical
counterpart.

__Follow-Up: Stellar, Dark Matter, and Combined Light+Mass Profiles__

This guide deliberately stops at the parametric lensing mass profiles.  The remaining mass
families — *stellar* mass profiles (e.g. `al.mp.Sersic`, `al.mp.Chameleon`,
`al.mp.GaussianGradient`), the *dark-matter* NFW family (`al.mp.NFW`, `al.mp.gNFW`,
`al.mp.cNFW`, all their MCR / virial / scatter variants), and the combined *light-and-mass*
namespaces `al.lmp.*` and `al.lmp_linear.*` — are the subject of a separate companion
guide:

    scripts/guides/profiles/light_and_mass_profiles.py

That guide tells the stellar-plus-dark decomposition story: how a Sersic mass component
representing the stellar matter combines with an NFW component representing the dark matter
halo, how the `lmp` profiles tie a Sersic *light* to a Sersic *mass* through a shared
mass-to-light ratio, and how those compositions feed into the standard model / instance /
Tracer flow shown here.

If you arrived at this guide from the API reference and now want to use any of these mass
profiles in an actual lens fit, the next step is `scripts/imaging/modeling/start_here.py`,
which sets up an `AnalysisImaging` and runs a non-linear search end-to-end on a strong-lens
dataset.  For a deeper walk-through of how mass profiles combine with light profiles in the
`Tracer` to produce lensed images, see `scripts/guides/tracer.py`.
"""

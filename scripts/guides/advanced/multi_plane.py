"""
Misc: Multi-Plane
=================

Multi-plane ray-tracing is used when a lens system has more planes than just an image-plane and source-plane. When
tracing from one plane to the next, the redshifts of the planes are used to compute scaling factors that are applied
to the deflection angles, so that every plane's lensing effect is combined consistently.

PyAutoLens follows the standard multiple lens-plane formalism of Schneider, Ehlers & Falco 1992
(https://ui.adsabs.harvard.edu/abs/1992grle.book.....S/abstract, section 9.1): equation (9.6) defines each plane's
scaled (dimensionless) deflection angles and equation (9.7b) is the recursive ray-tracing equation implemented in
the source code shown in this guide.

Examples of multi-plane lensing systems include:

 - A standard lens galaxy and source galaxy system, but where there is also a dark matter subhalo whose redshift is
 not at the redshift of the lens galaxy.

 - A strong lens system where the deflections due to many dark matter halos down the line-of-sight are included,
 which may be at a large range of different redshifts.

 - A galaxy cluster, where the observed background source galaxies are at a range of different redshifts and the
 deflections between all planes must be included.

This guide has two halves. The first half walks through the multi-plane ray-tracing algorithm itself, using
simplified copies of the PyAutoLens source code. The second half addresses a question every cluster-scale modeler
eventually asks: when mass profiles are defined in dimensionless "lensing units", and there are multiple source
planes at different redshifts, which source redshift gives those units meaning? Answering it reveals a genuine
division between how galaxy-scale and cluster-scale lens modeling define their units.

__Contents__

- **Example:** Set up a simple 3-plane lens system to illustrate multi-plane ray-tracing.
- **Ray Tracing:** Simplified copies of the multi-plane source code, with print statements showing each step.
- **Trace:** Ray-trace a coordinate through the 3-plane system and verify against the `Tracer`.
- **Lensing Units vs Physical Units:** Why galaxy-scale modeling samples dimensionless units but cluster-scale
  modeling samples physical masses.
- **The PyAutoLens Convention:** Deflections are normalized to the final plane; `redshift_source` of every physical
  profile is the highest redshift plane; the scaling factor is a ratio of critical surface densities.
- **Profiles With Physical Units:** A worked 3-plane example using the `NFWMCRLudlow` profile, with numerical checks
  of the convention.
- **Science Corollaries:** Mass-sheet degeneracy breaking and multi-plane cosmography.
- **Cross Validation:** Independent oracles for the recursion — astropy distances, the papers' equations,
  numerical and exact-autodiff Jacobians, a double Einstein ring — and the two convention traps that make an
  obvious sanity check give a confidently wrong answer.
- **Attribution:** The Slack discussion this guide distills.

__Example__

To illustrate multi-plane ray-tracing, we first set up a simple lens system.

We'll make things simple and assume 3 galaxies at redshifts 0.5, 1.0 and 2.0. We'll use a singular isothermal sphere
for each galaxy's mass profile.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from typing import List, Optional

import numpy as np

import autoarray as aa
import autogalaxy as ag
import autolens as al

lens_0 = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0))
lens_1 = al.Galaxy(redshift=1.0, mass=al.mp.IsothermalSph(einstein_radius=1.0))
lens_2 = al.Galaxy(redshift=2.0, mass=al.mp.IsothermalSph(einstein_radius=1.0))

"""
Multi-plane ray tracing is based on the redshifts of the planes that make up the lens system, as opposed to the
redshifts of the galaxies.

These two things are equivalent — there is a unique plane at each redshift containing every galaxy at that
redshift — but it means we group the galaxies above into planes in order to perform multi-plane ray-tracing.
"""
plane_0 = [lens_0]
plane_1 = [lens_1]
plane_2 = [lens_2]

"""
__Ray Tracing__

Multi-plane ray tracing is implemented in the `tracer_util.py` module of PyAutoLens:

https://github.com/PyAutoLabs/PyAutoLens/blob/main/autolens/lens/tracer_util.py

It uses the function `traced_grid_2d_list_from`, which relies on the `scaling_factor_between_redshifts_from` method
of the cosmology objects in the `cosmology` package of PyAutoGalaxy.

Simplified copies of both are below (the JAX / `xp` backend plumbing of the source code is removed for clarity, and
print statements are added to show how each step of the calculation works).

The scaling factor, written as $\\beta$ in the literature (equation 9.5 of Schneider, Ehlers & Falco 1992), rescales
the deflection angles of an earlier plane when computing where a ray pierces a later plane:

$\\beta_{ij} = \\frac{D_{ij} D_{s}}{D_{j} D_{is}}$

D_ij = Angular diameter distance between plane i and plane j.
D_s = Angular diameter distance from Earth to the final (highest redshift) plane.
D_j = Angular diameter distance from Earth to plane j.
D_is = Angular diameter distance between plane i and the final plane.

Two limits give intuition for what $\\beta$ does: $\\beta_{ij} = 1$ when plane j is the final plane (the deflections
are used exactly as computed) and $\\beta_{ij} = 0$ when plane j is plane i itself (a plane does not deflect light
before the light reaches it).
"""


def scaling_factor_between_redshifts_from(
    cosmology, redshift_0: float, redshift_1: float, redshift_final: float
) -> float:
    """
    For strong lens systems with more than 2 planes, the deflection angles between different planes must be scaled
    by the angular diameter distances between the planes in order to properly perform multi-plane ray-tracing.

    This function computes the factor which scales deflections between `redshift_0` and `redshift_final` to
    deflections between `redshift_0` and `redshift_1`.

    For a system with a first lens galaxy l0 at `redshift_0`, second lens galaxy l1 at `redshift_1` and final
    source galaxy at `redshift_final` this scaling factor is given by:

    (D_l0l1 * D_s) / (D_l1 * D_l0s)

    D_l0l1 = Angular diameter distance between the first and second lens redshifts.
    D_s = Angular diameter distance from Earth to the final source redshift.
    D_l1 = Angular diameter distance from Earth to the second lens redshift.
    D_l0s = Angular diameter distance between the first lens redshift and the final source redshift.

    For systems with more planes this scaling factor is computed multiple times for the different redshift
    combinations and applied recursively when scaling the deflection angles.

    Parameters
    ----------
    redshift_0
        The redshift of the first strong lens galaxy.
    redshift_1
        The redshift of the second strong lens galaxy.
    redshift_final
        The redshift of the final source galaxy.
    """
    D_l0l1 = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=redshift_0,
        redshift_1=redshift_1,
    )

    D_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_final,
    )

    D_l1 = cosmology.angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_1,
    )

    D_l0s = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=redshift_0,
        redshift_1=redshift_final,
    )

    return (D_l0l1 * D_s) / (D_l1 * D_l0s)


def traced_grid_2d_list_from(
    planes: List[List[al.Galaxy]],
    grid: aa.type.Grid2DLike,
    cosmology: al.cosmo.LensingCosmology = None,
    plane_index_limit: Optional[int] = None,
):
    """
    Returns a ray-traced grid of 2D Cartesian (y,x) coordinates which accounts for multi-plane ray-tracing.

    This uses the redshifts and mass profiles of the galaxies contained within the tracer to perform the multi-plane
    ray-tracing calculation.

    This function returns a list of 2D (y,x) grids, corresponding to each redshift in the input list of planes. The
    plane redshifts are determined from the redshifts of the galaxies in each plane, whereby there is a unique plane
    at each redshift containing all galaxies at the same redshift.

    For example, if the `planes` list contains three lists of galaxies with `redshift`'s z=0.5, z=1.0 and z=2.0, the
    returned list of traced grids will contain three entries corresponding to the input grid after ray-tracing to
    redshifts 0.5, 1.0 and 2.0.

    An input cosmology object can change the cosmological model, which is used to compute the scaling factors between
    planes (which are derived from their redshifts and angular diameter distances). It is these scaling factors that
    account for multi-plane ray tracing effects.

    The calculation can be terminated early by inputting a `plane_index_limit`. All planes whose integer indexes are
    above this value are omitted from the calculation and not included in the returned list of grids (the size of
    this list is reduced accordingly).

    For example, if `planes` has 3 lists of galaxies, but `plane_index_limit=1`, the third plane (corresponding to
    index 2) will not be calculated. The `plane_index_limit` is used to avoid unnecessary ray tracing calculations
    of higher redshift planes whose galaxies do not have mass profiles (and only have light profiles).

    Parameters
    ----------
    planes
        The galaxies whose mass profiles are used to perform multi-plane ray-tracing, where the list of galaxies
        has an index for each plane, corresponding to each unique redshift in the multi-plane system.
    grid
        The 2D (y, x) coordinates on which multi-plane ray-tracing calculations are performed.
    cosmology
        The cosmology used for ray-tracing from which angular diameter distances between planes are computed.
    plane_index_limit
        The integer index of the last plane which is used to perform ray-tracing, all planes with an index above
        this value are omitted.

    Returns
    -------
    traced_grid_list
        A list of 2D (y,x) grids each of which are the input grid ray-traced to a redshift of the input list of planes.
    """
    cosmology = cosmology or al.cosmo.Planck15()

    traced_grid_list = []
    traced_deflection_list = []

    redshift_list = [galaxies[0].redshift for galaxies in planes]

    for plane_index, galaxies in enumerate(planes):
        scaled_grid = np.asarray(grid.array)

        if plane_index > 0:
            for previous_plane_index in range(plane_index):
                scaling_factor = scaling_factor_between_redshifts_from(
                    cosmology=cosmology,
                    redshift_0=redshift_list[previous_plane_index],
                    redshift_1=galaxies[0].redshift,
                    redshift_final=redshift_list[-1],
                )

                print(
                    f"  beta(plane {previous_plane_index} -> plane {plane_index}) = {scaling_factor:.4f}"
                )

                scaled_deflections = (
                    scaling_factor * traced_deflection_list[previous_plane_index].array
                )

                scaled_grid = scaled_grid - scaled_deflections

        scaled_grid = al.Grid2DIrregular(values=scaled_grid)

        traced_grid_list.append(scaled_grid)

        print(
            f"plane {plane_index} (z={redshift_list[plane_index]}): "
            f"traced (y,x) = {np.asarray(scaled_grid.array)}"
        )

        if plane_index_limit is not None:
            if plane_index == plane_index_limit:
                return traced_grid_list

        deflections_yx_2d = sum(
            g.deflections_yx_2d_from(grid=scaled_grid) for g in galaxies
        )

        traced_deflection_list.append(deflections_yx_2d)

    return traced_grid_list


"""
__Trace__

The code below ray-traces a Cartesian coordinate y=1.0", x=0.0" to redshifts 0.5, 1.0 and 2.0 via multi-plane
ray-tracing.

The print statements show how the coordinate is transformed as it is ray-traced through each plane and therefore how
the multi-plane ray-tracing algorithm works: each plane's grid starts from the image-plane grid and subtracts every
earlier plane's deflection angles, scaled by the $\\beta$ factor appropriate to that pair of planes.
"""
grid = al.Grid2DIrregular(values=[(1.0, 0.0)])

traced_grid_list = traced_grid_2d_list_from(
    planes=[plane_0, plane_1, plane_2],
    grid=grid,
)

"""
The `Tracer` performs this same calculation internally (via `tracer_util.py`), so we can verify the simplified code
above against it.
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, lens_2])

traced_grid_list_via_tracer = tracer.traced_grid_2d_list_from(grid=grid)

for traced_grid, traced_grid_via_tracer in zip(
    traced_grid_list, traced_grid_list_via_tracer
):
    assert np.allclose(
        np.asarray(traced_grid.array), np.asarray(traced_grid_via_tracer.array)
    )

print("Simplified multi-plane code agrees with the Tracer.")

"""
__Lensing Units vs Physical Units__

The ray-tracing above used dimensionless angular units: the grid was in arc-seconds and the mass profile quantities
(e.g. the `einstein_radius`) were angles in arc-seconds. Understanding why PyAutoLens (and galaxy-scale lens modeling
in general) works this way — and when it stops being convenient — is the key to using multi-plane systems correctly.

Gravitational lensing does not measure mass directly; it measures the dimensionless convergence

$\\kappa(\\theta) = \\Sigma(\\theta) / \\Sigma_{\\rm crit}(z_{\\rm l}, z_{\\rm s})$

where $\\Sigma$ is the physical projected surface mass density and the critical surface density

$\\Sigma_{\\rm crit} = \\frac{c^{2} D_{\\rm s}}{4 \\pi G \\, D_{\\rm l} D_{\\rm ls}}$

depends on the lens redshift, the source redshift and the cosmology. The convergence folds the physical mass and
all cosmological distances into a single dimensionless field, from which the deflection angles follow.

**Galaxy-scale systems (one lens plane, one source plane):** there is a single $\\Sigma_{\\rm crit}$, so the
dimensionless "lensing units" parameterization — e.g. an NFW's `(kappa_s, scale_radius)` with the scale radius as an
angle — is complete and unambiguous. It is also exactly what the data constrain: the lensing observables depend only
on these dimensionless quantities, and many different physical systems map onto the same ones (a halo of mass M in
one system produces identical lensing to a halo of mass 2M in a system whose $\\Sigma_{\\rm crit}$ is twice as
large). Sampling lensing units therefore fits the parameters the data actually measure, works even when the
redshifts are unknown, and physical masses can be derived afterwards once redshifts and a cosmology are adopted.
This is the convention of galaxy-scale lens modeling.

**Cluster-scale systems (many mass components, multiple source planes):** each source plane has its own
$\\Sigma_{\\rm crit}(z_{\\rm l}, z_{\\rm s})$, so a lens no longer has "a" convergence field — a value of `kappa_s`
is meaningless until you state which source redshift it is defined relative to. Worse, the degeneracy that made
lensing units attractive is broken: changing a halo's redshift rescales its $\\Sigma_{\\rm crit}$ to two different
source planes by *different* ratios, so no single set of dimensionless parameters can absorb the redshift dependence
any more. Cluster-scale modeling therefore samples physical parameters directly — e.g. an NFW by
(log10(M200), c200) — with a fixed cosmology and known (or explicitly sampled) redshifts, and lets the ray-tracing
code derive the lensing quantities for every source plane under the hood.

Neither convention is "right": lensing units are minimal and redshift-agnostic where they are well defined, and
physical units are unavoidable where they are not. What matters is that a code states its convention precisely —
which is what the next section does for PyAutoLens.

__The PyAutoLens Convention__

PyAutoLens performs all internal calculations in dimensionless units, whether a mass profile was defined in angular
dimensionless units or physical units. The convention that removes the multiple-source-plane ambiguity is:

**Every mass profile's deflection angles are interpreted as reduced deflections to the final (highest redshift)
plane of the multi-plane system.**

Concretely:

 - For a profile defined in physical units (e.g. the dark matter profile `NFWMCRLudlow`, defined by a `mass_at_200`
 in solar masses), the constructor converts the physical parameters to dimensionless ones (`kappa_s`,
 `scale_radius`) using $\\Sigma_{\\rm crit}$(`redshift_object`, `redshift_source`). In a multi-plane system you must
 therefore set `redshift_source` to the redshift of the **highest redshift plane** for *every* profile, whichever
 plane the profile itself sits in.

 - The multi-plane ray-tracing then automatically produces the correct deflections at every intermediate plane,
 because the scaling factor is exactly a ratio of critical surface densities:

 $\\beta_{ij} = \\frac{D_{ij} D_{s}}{D_{j} D_{is}} =
 \\frac{\\Sigma_{\\rm crit}(z_{i}, z_{\\rm final})}{\\Sigma_{\\rm crit}(z_{i}, z_{j})}$

 Multiplying deflections normalized to the final plane by $\\beta_{ij}$ re-normalizes them to the plane the light is
 currently being traced to. This is why $\\beta = 1$ when tracing to the final plane and $\\beta = 0$ at the
 deflector's own plane.

 - To recover a physical projected mass from any profile, multiply its convergence by the critical surface density
 between the profile's redshift and the final plane: $\\Sigma(\\theta) = \\kappa(\\theta) \\,
 \\Sigma_{\\rm crit}(z_{\\rm profile}, z_{\\rm max})$.

A useful consequence: whether a plane hosts an observed "source galaxy" is irrelevant to the ray-tracing. The
deflection angles do not change if a light-emitting galaxy is added to a plane that previously contained only mass,
and a plane with no mass deflects nothing — the algorithm only cares about redshifts and mass profiles.

__Profiles With Physical Units__

The example below puts the convention into practice for a 3-plane system with two dark matter halos, defined
physically via `NFWMCRLudlow`, at z=0.5 and z=1.0, and a source plane at z=2.0.

Note that both halos receive `redshift_source=2.0` — the highest redshift plane of the system — even though one of
them sits at z=1.0.
"""
cosmology = al.cosmo.Planck15()

halo_0 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.NFWMCRLudlow(
        centre=(0.0, 0.0),
        mass_at_200=1e13,
        redshift_object=0.5,
        redshift_source=2.0,
    ),
)

halo_1 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.NFWMCRLudlow(
        centre=(0.1, 0.1),
        mass_at_200=5e12,
        redshift_object=1.0,
        redshift_source=2.0,
    ),
)

source = al.Galaxy(redshift=2.0)

tracer = al.Tracer(galaxies=[halo_0, halo_1, source], cosmology=cosmology)

traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

for plane_index, traced_grid in enumerate(traced_grid_list):
    print(f"plane {plane_index}: traced (y,x) = {np.asarray(traced_grid.array)}")

"""
The constructor of each `NFWMCRLudlow` has already converted `mass_at_200` to the dimensionless `kappa_s` used
internally, via $\\Sigma_{\\rm crit}$(`redshift_object`, `redshift_source`) (and the critical *density* of the
Universe at the halo's redshift, which sets the NFW's physical normalization and scale radius from the
mass-concentration relation):
"""
print(f"kappa_s of the z=0.5 halo = {halo_0.mass.kappa_s:.4f}")

"""
We now verify numerically that the multi-plane scaling factor is the ratio of critical surface densities claimed
above, using the cosmology object's `critical_surface_density_between_redshifts_solar_mass_per_kpc2_from` method.
"""
beta = cosmology.scaling_factor_between_redshifts_from(
    redshift_0=0.5, redshift_1=1.0, redshift_final=2.0
)

sigma_crit_to_final = (
    cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=0.5, redshift_1=2.0
    )
)

sigma_crit_to_next = (
    cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=0.5, redshift_1=1.0
    )
)

print(f"beta(z=0.5 -> z=1.0, final z=2.0) = {beta:.6f}")
print(
    f"sigma_crit(0.5, 2.0) / sigma_crit(0.5, 1.0) = {sigma_crit_to_final / sigma_crit_to_next:.6f}"
)

assert np.isclose(beta, sigma_crit_to_final / sigma_crit_to_next, rtol=1e-8)

"""
We also verify the statement that planes without mass do not alter the ray-tracing: adding an empty (massless,
light-less) plane at z=1.5 leaves the traced grid of the final plane unchanged.
"""
massless = al.Galaxy(redshift=1.5)

tracer_with_massless_plane = al.Tracer(
    galaxies=[halo_0, halo_1, massless, source], cosmology=cosmology
)

traced_grid_list_via_massless = tracer_with_massless_plane.traced_grid_2d_list_from(
    grid=grid
)

assert np.allclose(
    np.asarray(traced_grid_list[-1].array),
    np.asarray(traced_grid_list_via_massless[-1].array),
)

print("Adding a massless plane does not change the ray-tracing.")

"""
Finally, the projected physical mass associated with any profile: multiplying its convergence at some point by
$\\Sigma_{\\rm crit}$(z_profile, z_max) gives the physical surface mass density there.
"""
kappa = halo_0.mass.convergence_2d_from(grid=al.Grid2DIrregular(values=[(0.5, 0.0)]))

sigma = np.asarray(kappa) * sigma_crit_to_final

print(
    f'Surface mass density of the z=0.5 halo at (0.5", 0.0") = {sigma[0]:.3e} Msun / kpc^2'
)

"""
__Science Corollaries__

The breakdown of the single-$\\Sigma_{\\rm crit}$ picture in multiple source-plane systems is not just a bookkeeping
nuisance — it carries real scientific leverage:

 - **The mass-sheet degeneracy is (partially) broken.** In single source-plane lensing, a rescaled mass distribution
 plus a uniform mass sheet reproduces the same observables, which is a major systematic for e.g. time-delay
 cosmography. Sources at different redshifts probe different $\\Sigma_{\\rm crit}$ values, providing a lever arm
 that pins down the physical mass distribution — one reason the mass-sheet degeneracy features less prominently in
 cluster-scale lensing.

 - **Multi-plane cosmography.** The ratio of lensing strengths between source planes — the $\\beta$ factors — is a
 pure function of angular diameter distance ratios, i.e. of geometry and hence cosmology. Comparing predicted and
 measured $\\beta$ values for clusters with many multiply-imaged sources constrains cosmological parameters (e.g.
 $\\Omega_{\\rm m}$ and the dark energy equation of state), analogously to other geometric probes:
 https://arxiv.org/abs/2110.06232

__Cross Validation__

Everything above describes what the code does. This section is about how you check that it is *right*, which for
multi-plane ray-tracing deserves a section of its own: a magnification is just a number, and nothing about a
wrong one announces that the wrong $\\beta$ went into it. PyAutoLens#480 was a magnification that was wrong by a
factor of 15 and survived four months in exactly that way.

The only defence is an oracle that shares no code with the implementation. There are six available here, and all
of them are cheap:

 - the scaling factors recomputed from `astropy`'s angular diameter distances,
 - the multi-plane lens equation written out directly from the paper,
 - the Jacobian obtained by numerically differencing traced positions,
 - the Jacobian obtained by the published plane-by-plane recursion,
 - an exact analytic closed form (two aligned isothermal spheres, which produce a double Einstein ring),
 - exact automatic differentiation via JAX.

Each arm below is a handful of runnable code that prints its own residual against the `Tracer`. They are the same
oracles the library's own cross-validation module uses
(`test_autolens/lens/test_multi_plane_cross_validation.py`), reproduced here so you can point them at your own
configuration when a multi-plane result surprises you.

__The Formalism__

**Multi-plane lens equation** (Schneider, Ehlers & Falco, *Gravitational Lenses*, 1992, section 9.1, equations 9.6
and 9.7b). Writing $\\theta_{1}$ for the observed image-plane position and $\\alpha_{i}$ for the deflection of
plane $i$ evaluated at that plane's own traced position,

$\\theta_{j} = \\theta_{1} - \\sum_{i<j} \\beta_{ij} \\, \\alpha_{i}(\\theta_{i})$

with $\\beta_{ij} = D_{ij} D_{s} / (D_{j} D_{is})$ and $D_{s}$ the distance to the **final** plane of the system.
Two consequences of that normalization are used repeatedly below: $\\beta_{ij} = 1$ when
$z_{j} = z_{\\rm final}$ (then $D_{j} = D_{s}$ and $D_{ij} = D_{is}$), and $\\beta_{ij} = 0$ when $z_{i} = z_{j}$
(then $D_{ij} = 0$) — a deflector sitting *at* plane $j$ cannot bend light that has only reached plane $j$.

**Convergence, shear and magnification** (Narayan & Bartelmann 1996,
https://inspirehep.net/literature/419263, equations 55 and 60). With $A = d\\beta / d\\theta$ the Jacobian of the
lens mapping and $H_{ab} = d\\alpha_{a} / d\\theta_{b}$ the Hessian of the lensing potential,

$A = I - H$, $\\kappa = {\\rm tr}(H) / 2 = 1 - {\\rm tr}(A) / 2$, $\\gamma_{1} = (H_{xx} - H_{yy}) / 2$,
$\\gamma_{2} = H_{xy}$, $\\mu = 1 / \\det(A)$

**Jacobian recursion** (McCully, Keeton, Wong & Zabludoff 2014, https://arxiv.org/abs/1401.0197). Differentiating
the lens equation above with respect to $\\theta_{1}$ gives, with $U_{i} = d\\alpha_{i} / d\\theta$ evaluated at
$\\theta_{i}$,

$A_{1} = I$, $A_{j} = I - \\sum_{i<j} \\beta_{ij} \\, U_{i} A_{i}$

This is a genuinely different numerical route to $\\mu$ than differencing a traced position, because the chain
rule is applied analytically between planes and only the per-plane $U_{i}$ are differenced.

__Two Convention Traps__

Before the oracles, the two mistakes that produce a confidently wrong cross-check. Both come from the same place:
the final-plane normalization described in `__The PyAutoLens Convention__` above.

**(a) `deflections_between_planes_from` is a difference of traced grids, not a deflection.**
`Tracer.deflections_between_planes_from(plane_i=i, plane_j=j)` returns `traced_grids[i] - traced_grids[j]`.
Because every traced grid carries the final-plane normalization, this quantity is the *final-plane-scaled
difference of two positions*. It is **not** the physical deflection at plane $j$, which in the other common
convention would be $\\alpha_{j}$ rescaled by $D_{js} / D_{s}$. For `plane_i=0` it is exactly
$\\theta - \\theta_{j}$.

**(b) Truncating a tracer to the planes up to $j$ is not an oracle for plane $j$.** Dropping the planes above $j$
changes `redshift_final`, and therefore changes *every* $\\beta_{ij}$ left in the recursion, because $D_{s}$ and
$D_{is}$ in the formula above refer to the final plane of whatever system is being traced. This one is worth a
worked example, because it is the natural thing to try and it fails silently.

__Trap (b) In Practice: 1.86 Versus 27.9__

The configuration below is the one from PyAutoLens#480: a main lens at z=0.5, a second deflector at z=1.0 which
is itself compact (it hosts the intermediate source), and a final source plane at z=2.0. We ask for the
magnification at the **intermediate** plane, z=1.0, two ways.

The wrong way is to build a tracer containing only the planes up to z=1.0 and ask for the magnification of that
system. The right way is to keep the full three-plane tracer and select the plane with `plane_j=1`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

intermediate_galaxy = al.Galaxy(
    redshift=1.0,
    mass=al.mp.Isothermal(
        centre=(0.02, 0.03),
        einstein_radius=0.2,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
    ),
)

far_source = al.Galaxy(redshift=2.0)

tracer_480 = al.Tracer(galaxies=[lens_galaxy, intermediate_galaxy, far_source])
tracer_480_truncated = al.Tracer(galaxies=[lens_galaxy, intermediate_galaxy])

positions_480 = [
    (0.89140625, 0.63102580),
    (-0.67578125, -0.76634227),
    (1.07109375, -0.24131437),
    (-0.36875000, 1.04644736),
]

grid_480 = al.Grid2DIrregular(values=positions_480)

magnification_wrong = np.asarray(
    ag.LensCalc.from_tracer(
        tracer_480_truncated, use_multi_plane=True, plane_i=0, plane_j=1
    ).magnification_2d_via_hessian_from(grid=grid_480)
)

magnification_right = np.asarray(
    ag.LensCalc.from_tracer(
        tracer_480, use_multi_plane=True, plane_i=0, plane_j=1
    ).magnification_2d_via_hessian_from(grid=grid_480)
)

print(f"mu at z=1.0, truncated tracer (WRONG) = {magnification_wrong}")
print(f"mu at z=1.0, full tracer, plane_j=1   = {magnification_right}")

"""
The first image position returns 1.86 the wrong way and 27.9 the right way, and the other three are wrong by
factors of 6 to 10 — two of them with the sign flipped, so a parity argument would not have caught it either.
Nothing raised, nothing warned.

The reason is entirely in the scaling factor. In the full system the light travelling from z=0.5 to z=1.0 carries
$\\beta_{01} = 0.674$, because the deflections are normalized to z=2.0 and must be re-scaled down to reach the
nearer plane. Truncate the system and z=1.0 *becomes* the final plane, so $\\beta_{01} = 1$ by construction: the
same mass now deflects the same ray by 48% more.
"""
beta_full = cosmology.scaling_factor_between_redshifts_from(
    redshift_0=0.5, redshift_1=1.0, redshift_final=2.0
)
beta_truncated = cosmology.scaling_factor_between_redshifts_from(
    redshift_0=0.5, redshift_1=1.0, redshift_final=1.0
)

print(f"beta(0.5 -> 1.0) with final plane z=2.0 = {beta_full:.6f}")
print(f"beta(0.5 -> 1.0) with final plane z=1.0 = {beta_truncated:.6f}")

"""
An independent third opinion settles which of the two numbers is the magnification of the real system. The
Jacobian $A = d\\theta_{j} / d\\theta_{1}$ can be obtained by central-differencing the traced position itself,
with no `LensCalc` and no Hessian involved, and $\\mu = 1 / \\det(A)$ follows.

This little oracle is the workhorse of the rest of the section, so it is worth reading: `mapping` is the map
$\\theta \\rightarrow \\theta_{j}$ through the implementation under test, and the Jacobian is two central
differences per position.
"""


def traced_position_mapping_from(tracer, plane_index):
    """
    The map `theta -> theta_j` through the tracer, as a scalar-in / array-out callable suitable for
    central differencing.
    """

    def mapping(y, x):
        grid = al.Grid2DIrregular(values=[(y, x)])
        return np.asarray(tracer.traced_grid_2d_list_from(grid=grid)[plane_index])[0]

    return mapping


def jacobian_via_central_difference(mapping, positions, h=1e-6):
    """
    The (N, 2, 2) Jacobian `d theta_j / d theta_1` of an arbitrary (y, x) -> (y_j, x_j) mapping, by
    two-point central differences.
    """
    jacobians = []

    for y, x in np.asarray(positions, dtype=float):
        d_y = (np.asarray(mapping(y + h, x)) - np.asarray(mapping(y - h, x))) / (
            2.0 * h
        )
        d_x = (np.asarray(mapping(y, x + h)) - np.asarray(mapping(y, x - h))) / (
            2.0 * h
        )

        jacobians.append(np.array([[d_y[0], d_x[0]], [d_y[1], d_x[1]]]))

    return np.array(jacobians)


def magnification_via_jacobian(jacobians):
    """
    `mu = 1 / det(A)`, Narayan & Bartelmann equation 60.
    """
    return np.array([1.0 / np.linalg.det(jacobian) for jacobian in jacobians])


magnification_traced = magnification_via_jacobian(
    jacobian_via_central_difference(
        traced_position_mapping_from(tracer_480, plane_index=1), positions_480
    )
)

print(f"mu at z=1.0, ray-traced Jacobian      = {magnification_traced}")
print(
    f"max relative residual, LensCalc vs ray-traced Jacobian = "
    f"{np.max(np.abs(magnification_traced - magnification_right) / np.abs(magnification_right)):.2e}"
)

"""
The oracle agrees with the full tracer to a few parts in $10^{8}$ and disagrees with the truncated one by an
order of magnitude. The rule to take away: to ask for a quantity at an intermediate plane, keep the whole system
and select the plane index, never rebuild a shorter system.

__Oracle 1: Scaling Factors From Astropy__

The scaling factors are the first thing to check, because every other quantity is downstream of them.
`ag.cosmo.Planck15` is a hand-rolled `FlatLambdaCDM` that integrates $1/E(z)$ with its own Simpson rule, so
recomputing $\\beta_{ij}$ from `astropy.cosmology.Planck15` angular diameter distances is genuinely independent
evidence rather than a tautology.
"""
import astropy.units as u
from astropy.cosmology import Planck15 as astropy_planck15


def angular_diameter_distance_from(redshift_0, redshift_1):
    """
    Angular diameter distance between two redshifts in Mpc, from astropy's `Planck15`. Astropy 7 renamed
    the two-redshift form, so both spellings are tried.
    """
    try:
        distance = astropy_planck15.angular_diameter_distance(redshift_0, redshift_1)
    except TypeError:
        distance = astropy_planck15.angular_diameter_distance_z1z2(
            redshift_0, redshift_1
        )

    return distance.to(u.Mpc).value


def beta_via_astropy(redshift_i, redshift_j, redshift_final):
    """
    `beta_ij = (D_ij D_s) / (D_j D_is)`, SEF 1992 equation 9.7b, from astropy distances.
    """
    D_ij = angular_diameter_distance_from(redshift_i, redshift_j)
    D_s = angular_diameter_distance_from(0.0, redshift_final)
    D_j = angular_diameter_distance_from(0.0, redshift_j)
    D_is = angular_diameter_distance_from(redshift_i, redshift_final)

    return (D_ij * D_s) / (D_j * D_is)


def beta_via_project(redshift_i, redshift_j, redshift_final):
    """
    The same ratio, from PyAutoGalaxy's own cosmology.
    """
    return float(
        cosmology.scaling_factor_between_redshifts_from(
            redshift_0=redshift_i,
            redshift_1=redshift_j,
            redshift_final=redshift_final,
        )
    )


redshift_triples = [
    (0.5, 1.0, 2.0),
    (0.1, 1.0, 3.0),
    (0.5, 1.5, 2.0),
    (1.0, 1.5, 2.0),
    (0.2, 0.8, 1.6),
]

residual_list = []

for redshift_i, redshift_j, redshift_final in redshift_triples:
    beta_astropy = beta_via_astropy(redshift_i, redshift_j, redshift_final)
    beta_project = beta_via_project(redshift_i, redshift_j, redshift_final)

    residual_list.append(abs(beta_astropy - beta_project) / abs(beta_astropy))

    print(
        f"beta({redshift_i} -> {redshift_j}, final {redshift_final}): "
        f"astropy = {beta_astropy:.10f}, PyAutoGalaxy = {beta_project:.10f}"
    )

print(f"max relative difference, astropy vs PyAutoGalaxy = {max(residual_list):.2e}")

"""
The two agree to about $2 \\times 10^{-7}$, and that number is not noise — it is the difference between two
quadratures of the same integral, with all six Planck15 parameters (H0, Om0, Ob0, Tcmb0, Neff, m_nu) matching
astropy exactly.

That floor is why the next arm is run **twice**. An oracle built on astropy scaling factors can only ever test
the ray-tracing to $10^{-6}$, because below that it is measuring the cosmology rather than the recursion. Feeding
the project's *own* scaling factors into the paper's equation removes the cosmology from the comparison entirely
and lets the recursion itself — plane ordering, which deflection is scaled by which $\\beta$, everything — be
checked to $10^{-10}$. The first arm asks "is the cosmology right?"; the second asks "is the algebra right?".
Both are worth asking, and they are different questions.

__Oracle 2: The Lens Equation, Written From The Paper__

The recursion below is transcribed from SEF 1992 equation 9.6, not from `tracer_util.py`. It uses each galaxy's
*single-plane* `deflections_yx_2d_from` — the one piece of shared code, itself cross-validated against analytic
closed forms elsewhere in the library — and nothing else.

The test system is a four-plane one with three different elliptical deflectors, chosen so that no symmetry can
hide an ordering mistake.
"""


def deflections_summed_from(galaxies, positions):
    """
    Summed single-plane deflection angles of a list of galaxies at the given (N, 2) positions.
    """
    grid = al.Grid2DIrregular(values=np.asarray(positions, dtype=float))

    total = np.zeros(np.asarray(positions, dtype=float).shape)

    for galaxy in galaxies:
        total = total + np.asarray(galaxy.deflections_yx_2d_from(grid=grid))

    return total


def traced_positions_via_paper(planes, positions, redshift_final, beta_from):
    """
    `theta_j = theta_1 - sum_{i<j} beta_ij alpha_i(theta_i)`, SEF 1992 equation 9.6, written out here
    from the paper. Returns one (N, 2) array of positions per plane.
    """
    redshifts = [galaxies[0].redshift for galaxies in planes]

    theta_list = []
    alpha_list = []

    for plane_index, galaxies in enumerate(planes):
        theta = np.asarray(positions, dtype=float).copy()

        for previous_index in range(plane_index):
            beta = beta_from(
                redshifts[previous_index], redshifts[plane_index], redshift_final
            )
            theta = theta - beta * alpha_list[previous_index]

        theta_list.append(theta)
        alpha_list.append(deflections_summed_from(galaxies, theta))

    return theta_list


galaxy_0 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, -0.05), einstein_radius=1.2
    ),
)
galaxy_1 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.PowerLaw(
        centre=(0.15, -0.1), ell_comps=(0.05, 0.1), einstein_radius=0.4, slope=2.2
    ),
)
galaxy_2 = al.Galaxy(
    redshift=1.5,
    mass=al.mp.NFW(
        centre=(-0.2, 0.25), ell_comps=(0.03, 0.02), kappa_s=0.08, scale_radius=5.0
    ),
)
galaxy_3 = al.Galaxy(redshift=2.0)

tracer_general = al.Tracer(
    galaxies=[galaxy_0, galaxy_1, galaxy_2, galaxy_3], cosmology=cosmology
)
planes_general = [[galaxy_0], [galaxy_1], [galaxy_2], [galaxy_3]]

positions_general = [(1.7, 0.9), (-1.4, 1.1), (0.8, -1.6)]
grid_general = al.Grid2DIrregular(values=positions_general)

traced_grid_list_general = tracer_general.traced_grid_2d_list_from(grid=grid_general)

for beta_name, beta_from in [
    ("astropy", beta_via_astropy),
    ("project", beta_via_project),
]:
    theta_list = traced_positions_via_paper(
        planes_general, positions_general, redshift_final=2.0, beta_from=beta_from
    )

    residuals = [
        float(
            np.max(
                np.abs(
                    np.asarray(traced_grid_list_general[plane_index])
                    - theta_list[plane_index]
                )
            )
        )
        for plane_index in range(4)
    ]

    print(
        f"paper recursion ({beta_name} betas), max |delta| per plane = "
        f"{['%.2e' % residual for residual in residuals]}"
    )

"""
With astropy scaling factors the residuals sit at the $10^{-7}$ level set by the cosmology; with the project's
own scaling factors they are exactly zero to the last bit, at every one of the four planes. The recursion is not
approximately the paper's equation — it is the paper's equation.

Trap (a) is checked in passing here too: `deflections_between_planes_from(plane_i=0, plane_j=j)` should equal
$\\theta - \\theta_{j}$ from the recursion above, and *not* the summed single-plane deflections of the planes
below $j$, which is what the other convention would give.
"""
theta_list = traced_positions_via_paper(
    planes_general, positions_general, redshift_final=2.0, beta_from=beta_via_project
)

for plane_index in range(1, 4):
    deflections_between = np.asarray(
        tracer_general.deflections_between_planes_from(
            grid=grid_general, plane_i=0, plane_j=plane_index
        )
    )

    residual = np.max(
        np.abs(
            deflections_between
            - (np.asarray(positions_general) - theta_list[plane_index])
        )
    )

    print(
        f"plane {plane_index}: |deflections_between_planes - (theta - theta_j)| = {residual:.2e}"
    )

"""
__Oracle 3: A Numerical Jacobian, And Whether It Is One__

The central-difference Jacobian introduced above settled PyAutoLens#480, but a numerical derivative is only an
oracle if it is stable: too large a step and the second-order truncation error dominates, too small and
floating-point cancellation does. The honest way to present one is with its step sweep, so here is $\\mu$ at the
final plane of the four-plane system for $h$ from $10^{-4}$ to $10^{-7}$.
"""
mapping_general = traced_position_mapping_from(tracer_general, plane_index=3)

print("h        mu (three image positions)")

for h in [1e-4, 1e-5, 1e-6, 1e-7]:
    magnification = magnification_via_jacobian(
        jacobian_via_central_difference(mapping_general, positions_general, h=h)
    )
    print(f"{h:.0e}   {['%.6f' % value for value in magnification]}")

"""
Six significant figures are stable across three decades of step size, which is what licenses the arm to be used
as a reference. Where that stability breaks down — very close to a compact deflector's centre — is exactly the
regime discussed at the end of this section, and the sweep is how you find out you are in it.

__Oracle 4: The Jacobian Recursion__

The McCully et al. recursion propagates the Jacobian plane by plane, $A_{j} = I - \\sum_{i<j} \\beta_{ij} U_{i}
A_{i}$, so no perturbed ray is ever traced: the chain rule between planes is analytic and only each plane's own
$U_{i} = d\\alpha_{i} / d\\theta$ is differenced. From $A$ come the magnification, convergence and shear at
*every* plane, which is what makes this the arm that cross-checks `LensCalc` most broadly.

Note the shear convention: the library returns `[gamma_2, gamma_1]` in that column order, which the helper below
reproduces from $H = I - A$.
"""


def deflection_gradient_from(galaxies, position, h=1e-6):
    """
    `U_ab = d alpha_a / d theta_b` for one plane's galaxies at a single position, by central differences
    of the *profile* deflections (no ray tracing involved).
    """
    y, x = float(position[0]), float(position[1])

    d_y = (
        deflections_summed_from(galaxies, [(y + h, x)])[0]
        - deflections_summed_from(galaxies, [(y - h, x)])[0]
    ) / (2.0 * h)
    d_x = (
        deflections_summed_from(galaxies, [(y, x + h)])[0]
        - deflections_summed_from(galaxies, [(y, x - h)])[0]
    ) / (2.0 * h)

    return np.array([[d_y[0], d_x[0]], [d_y[1], d_x[1]]])


def jacobian_via_recursion(planes, positions, redshift_final, h=1e-6):
    """
    `A_1 = I`, `A_j = I - sum_{i<j} beta_ij U_i A_i` (McCully et al. 2014). Returns one (N, 2, 2) array
    of Jacobians per plane.
    """
    redshifts = [galaxies[0].redshift for galaxies in planes]
    positions = np.asarray(positions, dtype=float)

    jacobian_list = [np.zeros((positions.shape[0], 2, 2)) for _ in range(len(planes))]

    for position_index, position in enumerate(positions):
        theta_list = []
        A_list = []
        U_list = []

        for plane_index, galaxies in enumerate(planes):
            theta = position.copy()
            A = np.eye(2)

            for previous_index in range(plane_index):
                beta = beta_via_astropy(
                    redshifts[previous_index], redshifts[plane_index], redshift_final
                )
                theta = (
                    theta
                    - beta
                    * deflections_summed_from(
                        planes[previous_index], [theta_list[previous_index]]
                    )[0]
                )
                A = A - beta * U_list[previous_index] @ A_list[previous_index]

            theta_list.append(theta)
            A_list.append(A)
            U_list.append(deflection_gradient_from(galaxies, theta, h=h))

            jacobian_list[plane_index][position_index] = A

    return jacobian_list


def convergence_via_jacobian(jacobians):
    """
    `kappa = 1 - tr(A) / 2`, Narayan & Bartelmann equation 55 rearranged for `A = I - H`.
    """
    return np.array([1.0 - np.trace(jacobian) / 2.0 for jacobian in jacobians])


def shear_via_jacobian(jacobians):
    """
    `gamma_1 = (A_yy - A_xx) / 2` and `gamma_2 = -A[1, 0]`, returned in the library's `[gamma_2, gamma_1]`
    column order.
    """
    return np.array(
        [
            [-jacobian[1, 0], 0.5 * (jacobian[0, 0] - jacobian[1, 1])]
            for jacobian in jacobians
        ]
    )


jacobian_list = jacobian_via_recursion(
    planes_general, positions_general, redshift_final=2.0
)

for plane_index in range(1, 4):
    lens_calc = ag.LensCalc.from_tracer(
        tracer_general, use_multi_plane=True, plane_i=0, plane_j=plane_index
    )

    magnification = np.asarray(
        lens_calc.magnification_2d_via_hessian_from(grid=grid_general)
    )
    convergence = np.asarray(
        lens_calc.convergence_2d_via_hessian_from(grid=grid_general)
    )
    shear = np.asarray(lens_calc.shear_yx_2d_via_hessian_from(grid=grid_general))

    jacobians = jacobian_list[plane_index]

    print(
        f"plane {plane_index}: "
        f"mu residual = {np.max(np.abs(magnification_via_jacobian(jacobians) - magnification) / np.abs(magnification)):.2e}, "
        f"kappa residual = {np.max(np.abs(convergence_via_jacobian(jacobians) - convergence)):.2e}, "
        f"gamma residual = {np.max(np.abs(shear_via_jacobian(jacobians) - shear)):.2e}"
    )

"""
All three quantities agree at every plane to better than $10^{-6}$, the floor being the astropy scaling factors
and the central differences rather than either implementation. Two independent Jacobians, two independent
routes to the same lensing observables.

__Oracle 5: The Double Einstein Ring__

The strongest arm available is one with an exact closed form. Put two singular isothermal spheres on the same
axis, both centred on the origin, at z=0.5 and z=1.0 with a source plane at z=2.0. An SIS deflects by exactly
$\\theta_{\\rm E}$ radially outward, so on the y-axis the whole system collapses to a scalar problem:

$\\theta_{2} = \\theta_{1} - \\beta_{01} \\theta_{\\rm E,1} {\\rm sign}(\\theta_{1})$

$\\theta_{3} = \\theta_{1} - \\theta_{\\rm E,1} {\\rm sign}(\\theta_{1}) - \\theta_{\\rm E,2}
{\\rm sign}(\\theta_{2})$

using $\\beta_{02} = \\beta_{12} = 1$ at the final plane. Setting $\\theta_{3} = 0$ — a source exactly on the
axis — has two positive roots whenever $\\theta_{\\rm E,2} > (1 - \\beta_{01}) \\theta_{\\rm E,1}$:

outer ring: $\\theta_{1} = \\theta_{\\rm E,1} + \\theta_{\\rm E,2}$, from the branch with $\\theta_{2} > 0$

inner ring: $\\theta_{1} = \\theta_{\\rm E,1} - \\theta_{\\rm E,2}$, from the branch with $\\theta_{2} < 0$

The sign of $\\theta_{2}$ is the whole story: images inside $\\beta_{01} \\theta_{\\rm E,1}$ have already crossed
the axis by the time they reach the second deflector, so it bends them the other way. That is the double Einstein
ring — and with $\\theta_{\\rm E,1} = 1.0$ and $\\theta_{\\rm E,2} = 0.5$ the radii are exactly 1.5 and 0.5
arcsec.

We compare the closed form against root-finding on the *traced* position, i.e. against the implementation.
"""
from scipy.optimize import brentq

einstein_radius_0 = 1.0
einstein_radius_1 = 0.5

tracer_ring = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.IsothermalSph(
                centre=(0.0, 0.0), einstein_radius=einstein_radius_0
            ),
        ),
        al.Galaxy(
            redshift=1.0,
            mass=al.mp.IsothermalSph(
                centre=(0.0, 0.0), einstein_radius=einstein_radius_1
            ),
        ),
        al.Galaxy(redshift=2.0),
    ],
    cosmology=cosmology,
)

beta_01 = beta_via_astropy(0.5, 1.0, 2.0)

print(
    f"beta_01 = {beta_01:.6f}, two rings exist = {einstein_radius_1 > (1.0 - beta_01) * einstein_radius_0}"
)

radius_outer_analytic = einstein_radius_0 + einstein_radius_1
radius_inner_analytic = einstein_radius_0 - einstein_radius_1


def source_plane_y_from(theta):
    """
    The y coordinate of an on-axis image position after tracing to the final plane. Its roots are the
    Einstein ring radii.
    """
    grid = al.Grid2DIrregular(values=[(theta, 0.0)])

    return float(np.asarray(tracer_ring.traced_grid_2d_list_from(grid=grid)[2])[0, 0])


radius_outer_traced = brentq(source_plane_y_from, 0.9, 2.5, xtol=1e-14)
radius_inner_traced = brentq(source_plane_y_from, 0.05, 0.6, xtol=1e-14)

print(
    f"outer ring: closed form = {radius_outer_analytic:.10f}, root-found = {radius_outer_traced:.10f}"
)
print(
    f"inner ring: closed form = {radius_inner_analytic:.10f}, root-found = {radius_inner_traced:.10f}"
)
print(
    f"max |delta| = "
    f"{max(abs(radius_outer_analytic - radius_outer_traced), abs(radius_inner_analytic - radius_inner_traced)):.2e}"
)

"""
Both radii come back to machine precision. A quick figure of the two rings, with the region interior to
$\\beta_{01} \\theta_{\\rm E,1}$ marked — the images inside it are the ones the second deflector bends backwards:
"""
import matplotlib.pyplot as plt

figure, axis = plt.subplots(figsize=(5, 5))

axis.add_patch(
    plt.Circle(
        (0.0, 0.0), radius_outer_traced, fill=False, color="k", lw=2, label="outer ring"
    )
)
axis.add_patch(
    plt.Circle(
        (0.0, 0.0), radius_inner_traced, fill=False, color="r", lw=2, label="inner ring"
    )
)
axis.add_patch(
    plt.Circle(
        (0.0, 0.0), beta_01 * einstein_radius_0, fill=False, color="b", ls="--", lw=1
    )
)

axis.plot(0.0, 0.0, "k+", markersize=10)
axis.set_xlim(-2.0, 2.0)
axis.set_ylim(-2.0, 2.0)
axis.set_aspect("equal")
axis.set_xlabel('x (")')
axis.set_ylabel('y (")')
axis.set_title("Double Einstein ring (two aligned SIS)")
axis.legend(loc="upper right")

plt.show()
plt.close()

"""
__Oracle 6: Exact Automatic Differentiation__

Every Jacobian so far has been a finite difference. JAX gives the derivative itself: `jax.jacfwd` differentiates
the map $\\theta \\rightarrow \\theta_{j}$ through the library's own ray-tracing code, with no step size to
choose and no truncation error to sweep.

Two details matter. First, the pairing rule of `guides/using_jax.py` applies — inside a traced function the grid
and the library call both need `xp=jnp`, and the returned plane grid is unwrapped with `.array`. Second, and less
obviously, **float64 is not optional here**: in float32 this comparison returns residuals of order $10^{-2}$,
which would look like a bug in the ray-tracing rather than the rounding error it is. The
`from autolens import jax_wrapper` import at the top of this script is what turns 64-bit mode on, which is why it
must come before anything else.

We use the PyAutoLens#480 configuration from the start of this section, at the **final** plane this time.
"""
import jax
import jax.numpy as jnp


def magnification_via_jacfwd(tracer, positions, plane_index):
    """
    `mu = 1 / det(A)` with `A = d theta_j / d theta_1` from exact forward-mode automatic
    differentiation of the library's own multi-plane ray-tracing.
    """

    def traced_position(theta):
        grid = al.Grid2DIrregular(values=jnp.asarray(theta).reshape(1, 2), xp=jnp)

        return tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)[
            plane_index
        ].array.reshape(2)

    magnification = []

    for position in positions:
        jacobian = jax.jacfwd(traced_position)(jnp.asarray(position, dtype=jnp.float64))
        magnification.append(float(1.0 / jnp.linalg.det(jacobian)))

    return np.array(magnification)


magnification_jax = magnification_via_jacfwd(tracer_480, positions_480, plane_index=2)

magnification_traced_last = magnification_via_jacobian(
    jacobian_via_central_difference(
        traced_position_mapping_from(tracer_480, plane_index=2), positions_480, h=1e-7
    )
)

print(f"mu at z=2.0, JAX jacfwd (float64)  = {magnification_jax}")
print(f"mu at z=2.0, ray-traced Jacobian   = {magnification_traced_last}")
print(
    f"max relative residual = "
    f"{np.max(np.abs(magnification_jax - magnification_traced_last) / np.abs(magnification_jax)):.2e}"
)

"""
The two agree to a few parts in $10^{4}$ at the hardest position and far better at the others: an exact
derivative and a finite-difference one, computed by different tools, on a configuration built to be difficult.

__A Warning: The NumPy Hessian Step Is Too Coarse Near A Compact Deflector__

The oracles above have credentials, which lets them be used to make a statement about the library rather than
only about themselves. Here is one.

`LensCalc`'s NumPy path computes its Hessian by Richardson extrapolation with a hardcoded `buffer = 0.01` arcsec
step. That is fine almost everywhere, including at the intermediate plane of the #480 configuration where it
agreed with the ray-traced Jacobian to a few parts in $10^{8}$ at the start of this section. It is not fine at
the final plane of that same configuration, where the rays pass within $\\sim 4 \\times 10^{-4}$ arcsec of a
compact isothermal centre — a step 25 times larger than the distance to the singularity samples a completely
different deflection field.
"""
magnification_hessian_last = np.asarray(
    ag.LensCalc.from_tracer(
        tracer_480, use_multi_plane=True, plane_i=0, plane_j=2
    ).magnification_2d_via_hessian_from(grid=grid_480)
)

print(f"mu at z=2.0, LensCalc Richardson Hessian = {magnification_hessian_last}")
print(f"mu at z=2.0, exact autodiff and ray-traced Jacobian = {magnification_jax}")

"""
The Hessian returns approximately `[-0.00694, -0.00221, 0.00139, 0.00246]` where exact autodiff and the
ray-traced Jacobian both give `[0.04508, 0.01099, -0.08602, -0.01118]`: wrong by 100-120%, with a flipped sign
at all four image positions.

This is **a known, filed defect, and it is not fixed** — it is pinned as a strict expected failure in the
library's cross-validation module and tracked as
`PyAutoMind/draft/bug/autogalaxy/lenscalc_numpy_hessian_step_is_too_coarse.md`. If your multi-plane system is
compact — a deflector at an intermediate plane sitting close to the sightline of the images you care about — do
not take a NumPy Hessian magnification on trust. Use the JAX path shown above, or the ray-traced Jacobian, both
of which resolve the field at the scale the rays actually probe.

More generally, that is what this whole section is for: every arm above is a few lines long, and running two of
them against each other costs a second. When a multi-plane number surprises you, that is much cheaper than four
months.

__Attribution__

This guide distills a discussion on the PyAutoLens Slack, where users modeling cluster-scale lenses in physical
units worked through the source-redshift ambiguity above with the development team. Thanks to all involved — the
questions, wrong turns and worked answers of that thread shaped every section of the second half of this guide.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: jax full_datasets
"""

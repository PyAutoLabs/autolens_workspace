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

ENV: full_datasets
"""

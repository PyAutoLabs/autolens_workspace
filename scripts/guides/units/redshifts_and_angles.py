"""
Units: Redshifts and Angles
===========================

A question we are asked a lot: "Why doesn't changing the lens or source redshift change my lensed image?"

This guide answers it. It explains why **PyAutoLens** describes lenses with angles, how you turn those angles into
physical quantities (kiloparsecs, solar masses) afterwards, and why galaxy cluster modelers usually work the other way
round, starting from physical masses and letting the angles follow.

You do not need to know anything about multi-plane ray-tracing to follow it. If you want that detail afterwards, the
guide `guides/advanced/multi_plane.py` picks up where this one stops.

__Contents__

- **The Question:** Changing the redshifts of a lens and source does not change the image PyAutoLens makes.
- **The Short Answer:** A telescope measures where things are on the sky as angles, so a lens defined by an angle
  looks the same at any distance.
- **A Little Maths:** The lens equation, the reduced deflection angle and the critical surface density.
- **See It For Yourself:** The same tracer at two redshift pairs gives an identical image.
- **Converting Afterwards:** The same image gives a different physical size and mass at each redshift pair.
- **The Opposite Direction:** Cluster modelers start from physical masses, so the redshift does change their image.
- **Why Clusters Need Physical Units:** With several source planes, no single set of angles describes the lens.
- **PyAutoLens Supports Both:** Angular profiles and physical profiles, and the one rule for using physical ones.
- **Where Next:** Guides that go further.
- **Attribution:** The Slack discussion this guide comes from.

__The Question__

Suppose you set up a lens galaxy at redshift 0.5 and a source galaxy at redshift 1.0, and make an image of the
strong lens. Now move the lens to redshift 0.2 and the source to redshift 2.0, and make the image again.

It is exactly the same, to every pixel. Surely moving galaxies billions of light years around should change
something?

__The Short Answer__

A telescope does not see distances. It sees angles on the sky: where the arcs are, how far apart the multiple images
are, how big the Einstein ring is. All of these are measured in arcseconds.

Think of holding a coin at arm's length and a dinner plate across the room. They can cover exactly the same patch of
your view. From where you stand you cannot tell which one you are looking at; only the angle they cover reaches your
eye.

Lensing works the same way. In PyAutoLens a lens galaxy's strength is set by an angle, for example an
`einstein_radius` of 1.6 arcseconds. Change the redshifts and you change how far away the galaxies are and how much
mass the lens needs to produce that 1.6 arcsecond ring. But the ring still has a radius of 1.6 arcseconds, so the
image does not change.

The figure below shows this. In both setups the light reaches Earth at the same angle, so the telescope sees the same
image, even though the galaxies are at different distances and the lens in setup B has less than a third of the mass
of the lens in setup A.

![Same angle at Earth, identical image](../../../images/guides/redshift_angles_geometry.png)

Conceptual illustration. The side-on sketch is not to scale; the two images on the right are real PyAutoLens images
of the two setups, and they are identical.

__A Little Maths__

The lens equation says where light from the source appears on the sky:

$\\beta = \\theta - \\alpha(\\theta)$

Here $\\theta$ is the angle at which we see a light ray, $\\beta$ is the angle at which we would have seen the source
without the lens, and $\\alpha$ is how far the lens bends the ray. All three are angles on the sky.

The bending that a lens actually does to a ray is the deflection angle $\\hat{\\alpha}$. How much of that bending we
see from Earth depends on the distances involved:

$\\alpha = \\frac{D_{\\rm ls}}{D_{\\rm s}} \\hat{\\alpha}$

where $D_{\\rm s}$ is the distance from Earth to the source and $D_{\\rm ls}$ the distance from the lens to the
source. For a given bending $\\hat{\\alpha}$, the redshifts enter through this distance ratio.

When you give PyAutoLens an `einstein_radius` in arcseconds you are setting $\\alpha$ directly, so the distance ratio
is already absorbed into the number you chose. There is nothing left for the redshifts to change.

The same idea can be written in terms of mass. Lensing depends on the dimensionless convergence

$\\kappa = \\Sigma / \\Sigma_{\\rm crit}(z_{\\rm l}, z_{\\rm s})$

where $\\Sigma$ is the lens's projected surface mass density and $\\Sigma_{\\rm crit}$, the critical surface density,
depends on the lens redshift, the source redshift and the cosmology. Only the ratio matters. A lens with surface
density $\\Sigma$ in a system with one $\\Sigma_{\\rm crit}$ produces exactly the same lensing as a lens with surface
density $2\\Sigma$ in a system whose $\\Sigma_{\\rm crit}$ is twice as large. The image cannot tell them apart.

So the image tells you $\\kappa$, or equivalently the Einstein radius in arcseconds. It does not tell you the mass
until you also say what the redshifts are.

__See It For Yourself__

Let's check this with PyAutoLens. We make the same lens and source twice, once with redshifts (0.5, 1.0) and once
with (0.2, 2.0), and compare the images.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
import autolens as al

grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=0.05)


def tracer_from(redshift_lens, redshift_source):
    lens = al.Galaxy(
        redshift=redshift_lens,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.0),
            einstein_radius=1.6,
        ),
    )

    source = al.Galaxy(
        redshift=redshift_source,
        bulge=al.lp.SersicCore(
            centre=(0.1, 0.1),
            ell_comps=(0.1, 0.0),
            intensity=1.0,
            effective_radius=0.3,
            sersic_index=1.0,
        ),
    )

    return al.Tracer(galaxies=[lens, source])


tracer_a = tracer_from(redshift_lens=0.5, redshift_source=1.0)
tracer_b = tracer_from(redshift_lens=0.2, redshift_source=2.0)

image_a = np.asarray(tracer_a.image_2d_from(grid=grid).native)
image_b = np.asarray(tracer_b.image_2d_from(grid=grid).native)

max_difference = np.max(np.abs(image_a - image_b))

print(f"Largest pixel difference between the two images = {max_difference}")

assert max_difference == 0.0

"""
The difference is exactly zero. PyAutoLens never needed the redshifts to make this image.

The same is true of the source's light: its `effective_radius` is in arcseconds and its `intensity` is what the
telescope records, so PyAutoLens applies no change in apparent size or brightness with redshift. A real galaxy moved
from redshift 1 to 2 would look smaller and fainter; in PyAutoLens you describe that by changing these parameters.

__Converting Afterwards__

The redshifts matter as soon as you ask a physical question: how big is the Einstein radius in kiloparsecs, or how
much mass is inside it?

To answer, we choose a cosmology and convert. The guide `guides/units/cosmology.py` covers these conversions in full;
here we use three of them:

 - `kpc_per_arcsec_from`: how many kiloparsecs one arcsecond spans at the lens redshift.
 - `critical_surface_density_between_redshifts_from`: $\\Sigma_{\\rm crit}$ in solar masses per square arcsecond.
 - `einstein_mass_angular_from`: the Einstein mass in angular units, $\\pi \\theta_{\\rm E}^2$.

Multiplying the angular Einstein mass by $\\Sigma_{\\rm crit}$ gives the mass in solar masses.
"""
cosmology = al.cosmo.Planck15()

# A coarser grid than the one used for the images is enough to locate the critical curve.
grid_calc = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer in [tracer_a, tracer_b]:
    redshift_lens = tracer.planes[0].redshift
    redshift_source = tracer.planes[1].redshift

    lens_calc = al.LensCalc.from_tracer(tracer=tracer)

    einstein_radius = lens_calc.einstein_radius_from(grid=grid_calc)
    einstein_mass_angular = lens_calc.einstein_mass_angular_from(grid=grid_calc)

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_lens)
    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=redshift_lens, redshift_1=redshift_source
        )
    )

    print(f"Lens z = {redshift_lens}, source z = {redshift_source}:")
    print(f"  Einstein radius = {einstein_radius:.2f} arcsec")
    print(f"  Einstein radius = {einstein_radius * kpc_per_arcsec:.2f} kpc")
    print(
        f"  Einstein mass   = {einstein_mass_angular * critical_surface_density:.3e} solar masses"
    )

"""
Same image, same Einstein radius in arcseconds, but a different physical answer: the lens at redshift 0.5 is more
than three and a half times as massive as the lens at redshift 0.2, and its Einstein radius spans nearly twice as many
kiloparsecs. (The Einstein radius prints as 1.59 rather than 1.6 because PyAutoLens measures it as the radius of a
circle with the same area as the critical curve, which for this slightly elliptical lens is a little smaller than the
`einstein_radius` parameter. A round lens gives 1.60.)

This is why PyAutoLens works in angles. The data constrain the angles; physical quantities follow once you adopt
redshifts and a cosmology. It also means you can model a lens whose redshifts you do not know yet, and convert later
when a spectrum arrives. (Time delays are the exception: they depend on the distances themselves, so they need the
redshifts.)

__The Opposite Direction__

Galaxy cluster modelers usually work the other way round. Parametric cluster codes such as Lenstool and glafic
describe each mass component with physical quantities. In Lenstool, for example, these are:

 - a velocity dispersion `v_disp` in km/s, rather than an angular lens strength;
 - core and cut radii that can be given in kiloparsecs;
 - a lens redshift `z_lens`, and a redshift for each multiply imaged source (measured, or fitted when unknown).

The code then uses the redshifts and a cosmology to turn these physical quantities into the angles that the ray
tracing needs. For the dPIE profile Lenstool uses, the angular lens strength is
$b_0 = 6 \\pi (\\sigma / c)^2 D_{\\rm ls} / D_{\\rm s}$ (in radians), and the script
`cluster/lenstool/parameterization_mapping.py` shows this conversion in PyAutoLens. Here $\\sigma$ is Lenstool's
fiducial velocity dispersion, which is $\\sqrt{2/3}$ times the central dispersion $\\sigma_0$; that is why the
prefactor is $6\\pi$ rather than the $4\\pi$ of the isothermal formula below.

So in a cluster code, if you hold the mass fixed and change a redshift, the angles change and the image changes. In
PyAutoLens, if you hold the angle fixed and change a redshift, nothing changes. Both are correct; they hold different
things fixed. The figure below shows the two routes into the same ray-tracing calculation, and what each one does
when the redshifts change.

![Angular units versus physical units](../../../images/guides/angular_vs_physical_units.png)

Conceptual illustration. The flow diagram is schematic; the six images are real PyAutoLens images.

We can reproduce the bottom row ourselves. For a singular isothermal sphere with velocity dispersion $\\sigma$ the
Einstein radius is

$\\theta_{\\rm E} = 4 \\pi \\left(\\frac{\\sigma}{c}\\right)^2 \\frac{D_{\\rm ls}}{D_{\\rm s}}$ (in radians)

We pick the $\\sigma$ that gives a 1.6 arcsecond ring for redshifts (0.5, 1.0), then hold it fixed and move the
galaxies. PyAutoLens's `velocity_dispersion_from` inverts this formula.
"""
speed_of_light_km_s = 299792.458
arcsec_per_radian = 206264.806


def einstein_radius_from_sigma(sigma_km_s, redshift_lens, redshift_source):
    distance_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=redshift_lens, redshift_1=redshift_source
    )
    distance_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_source
    )

    return (
        4.0
        * np.pi
        * (sigma_km_s / speed_of_light_km_s) ** 2
        * (distance_ls / distance_s)
        * arcsec_per_radian
    )


sigma = cosmology.velocity_dispersion_from(
    redshift_0=0.5, redshift_1=1.0, einstein_radius=1.6
)

print(f"Velocity dispersion giving a 1.6 arcsec ring at (0.5, 1.0) = {sigma:.1f} km/s")

for redshift_lens, redshift_source in [(0.5, 1.0), (0.2, 2.0), (0.8, 1.2)]:
    einstein_radius = einstein_radius_from_sigma(
        sigma_km_s=sigma, redshift_lens=redshift_lens, redshift_source=redshift_source
    )
    print(
        f"  Lens z = {redshift_lens}, source z = {redshift_source}: "
        f"Einstein radius = {einstein_radius:.2f} arcsec"
    )

"""
The same galaxy, with the same mass, makes a 1.60, 3.15 or 0.97 arcsecond ring depending on where it and the source
are. Moved to redshift 0.2 with the source at redshift 2.0, it lenses far more strongly.

Two honest caveats:

 - **Not every cluster code is physical first.** Some, such as the light-traces-mass (LTM) method, normalise the
 lensing strength to a reference source redshift and rescale it for each source by $D_{\\rm ls}/D_{\\rm s}$. That is
 closer to the angular approach used by PyAutoLens.

 - **Everyone fits angles in the end.** Every cluster code fits data measured on the sky, such as the positions of
 multiple images in arcseconds. The difference between the two approaches is which parameters you sample, not what
 data you fit.

__Why Clusters Need Physical Units__

If angles are what the data measure, why do cluster modelers bother with physical masses? Because a cluster usually
lenses several background galaxies at different redshifts.

With one source, there is one $\\Sigma_{\\rm crit}$, so one convergence map describes the lens completely. With two
sources at redshifts 1 and 3, the same lens has a different $\\Sigma_{\\rm crit}$, and so a different convergence,
for each of them. A value like "an Einstein radius of 1.6 arcseconds" no longer means anything until you say which
source it refers to.

This rescaling trick also stops working. Earlier, moving the lens changed the mass needed but left the image
unchanged, because only one $\\Sigma_{\\rm crit}$ rescaled. Now moving the lens rescales $\\Sigma_{\\rm crit}$ to each
source by a different factor, so no single set of angles can absorb the change.

Let's put numbers on that. We move a lens from redshift 0.3 to redshift 0.5 and see how much $\\Sigma_{\\rm crit}$
changes for sources at redshift 1 and redshift 3.
"""


def sigma_crit_from(redshift_lens, redshift_source):
    return (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_lens, redshift_1=redshift_source
        )
    )


ratio_source_1 = sigma_crit_from(0.5, 1.0) / sigma_crit_from(0.3, 1.0)
ratio_source_3 = sigma_crit_from(0.5, 3.0) / sigma_crit_from(0.3, 3.0)

print("Moving the lens from z = 0.3 to z = 0.5 rescales Sigma_crit by:")
print(f"  x {ratio_source_1:.3f} for the source at z = 1")
print(f"  x {ratio_source_3:.3f} for the source at z = 3")

assert not np.isclose(ratio_source_1, ratio_source_3, rtol=0.01)

"""
The two factors are different, so one source's arcs cannot be kept fixed by the same rescaling that keeps the
other's fixed. A model with several source planes is therefore most naturally written in terms of the physical mass
and the redshifts, with each source's angles following from them.

The ratio of $\\Sigma_{\\rm crit}$ between the two source planes is the factor PyAutoLens uses to rescale deflection
angles from one plane to another in multi-plane ray-tracing. We can check they are the same number:
"""
for redshift_lens in [0.3, 0.5]:
    sigma_crit_ratio = sigma_crit_from(redshift_lens, 3.0) / sigma_crit_from(
        redshift_lens, 1.0
    )

    scaling_factor = cosmology.scaling_factor_between_redshifts_from(
        redshift_0=redshift_lens, redshift_1=1.0, redshift_final=3.0
    )

    print(
        f"Lens z = {redshift_lens}: Sigma_crit(z_s=3) / Sigma_crit(z_s=1) = {sigma_crit_ratio:.4f}, "
        f"scaling factor = {scaling_factor:.4f}"
    )

    assert np.isclose(sigma_crit_ratio, scaling_factor, rtol=1e-8)

"""
The ratio depends on the lens redshift (0.79 at redshift 0.3, 0.61 at redshift 0.5), which is exactly why the two
sources respond differently when the lens moves.

This has two useful consequences:

 - **The mass-sheet degeneracy is (partly) broken.** With one source, adding a uniform sheet of mass and rescaling
 the lens and the source leaves the image unchanged, which limits what you can learn about the mass. Sources at
 known, different redshifts see different $\\Sigma_{\\rm crit}$ values, so the trick no longer works for all of them
 at once.

 - **Clusters can measure cosmology.** The ratio of lensing strengths between source planes depends only on
 distance ratios, and so on the geometry of the Universe. Clusters with many multiply imaged sources at known
 redshifts constrain cosmological parameters this way (Caminha et al. 2022, https://arxiv.org/abs/2110.06232).

__PyAutoLens Supports Both__

PyAutoLens has mass profiles for both approaches:

 - **Angular profiles** such as `Isothermal`, `NFW` and `dPIEMassB0` take angular or dimensionless parameters (an
 `einstein_radius`, a `kappa_s` and `scale_radius`, a `b0`) and need no redshifts.

 - **Physical profiles** such as `NFWMCRLudlow` (a halo mass `mass_at_200`) and `dPIEMass` (Lenstool's `sigma`,
 with `r_core` and `r_cut` in arcseconds) take physical quantities and the redshifts `redshift_object` and
 `redshift_source`, and convert to angles when you create them.

Below, the same physical halo is created for a source at redshift 1.0 and at redshift 2.0. Its dimensionless lens
strength, `kappa_s`, is different for each, as the sections above predict.
"""
for redshift_source in [1.0, 2.0]:
    halo = al.mp.NFWMCRLudlow(
        centre=(0.0, 0.0),
        mass_at_200=1e13,
        redshift_object=0.5,
        redshift_source=redshift_source,
    )
    print(f"Source z = {redshift_source}: kappa_s = {halo.kappa_s:.4f}")

"""
When a physical profile sits in a system with more than one source plane, there is one rule: set `redshift_source`
to the redshift of the **highest-redshift plane** in the system, for every profile. PyAutoLens then rescales the
deflections correctly for every nearer plane. The guide `guides/advanced/multi_plane.py` explains why and checks it
numerically.

__Where Next__

 - `guides/units/cosmology.py`: every unit conversion, from arcseconds to kiloparsecs and from angular masses to solar
 masses.

 - `guides/advanced/multi_plane.py`: how multi-plane ray-tracing works, the `redshift_source` convention for physical
 profiles, and how to check it.

 - `cluster/lenstool/parameterization_mapping.py`: converting Lenstool's physical `sigma` to the angular `b0` and
 back, for cluster modelers moving to PyAutoLens.

__Attribution__

This guide distills a discussion on the PyAutoLens Slack, where users modeling cluster-scale lenses in physical units
asked why the redshift does not change a PyAutoLens image, and worked through the answer with the development team.
Thanks to all involved: the explanations in that thread, especially of why angular units break down with several
source planes, shaped this guide.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

This guide loads no dataset.

ENV: full_datasets
"""

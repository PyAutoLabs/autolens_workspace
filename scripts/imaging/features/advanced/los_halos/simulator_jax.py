"""
Simulator: Line-of-Sight Halos (JAX)
=====================================

This script simulates the same strong lens configuration as ``simulator.py`` — a galaxy-scale
lens with line-of-sight (LOS) dark matter halos — but uses a JAX-accelerated path that compiles
the full simulation pipeline under ``jax.jit``.

The standard ``Tracer``-based simulator in ``simulator.py`` uses Python loops to iterate over
redshift planes and sum deflections from each halo. This works well for typical lens models with
a few galaxies, but becomes a bottleneck when the number of halos grows into the hundreds or
thousands (as in substructure forward models where the halo population is drawn from a mass
function and varies between realisations).

The JAX path replaces these Python loops with two JAX primitives:

 - ``jax.vmap`` vectorises the deflection computation across all halos on a given plane, so
   that a single GPU kernel evaluates every halo simultaneously.

 - ``jax.lax.scan`` iterates over redshift planes with a fixed-structure loop that compiles
   to a single XLA operation, regardless of how many planes or halos are involved.

Together, these allow the full ``theta -> noisy image`` pipeline to compile once under
``jax.jit`` and then be reused without recompilation for different halo populations. The
compiled function can also be batched via ``jax.vmap`` to produce many realisations per GPU
launch.

__Contents__

- **LOS Configuration:** Redshifts, mass range, light-cone parameters (same as ``simulator.py``).
- **Sample LOS Halos:** Use ``LOSSampler`` to draw a halo population (runs in NumPy).
- **Grid:** Define the 2D grid on which the image is evaluated and simulated.
- **PSF:** A Gaussian PSF kernel for convolution.
- **Lens Galaxy and Source Galaxy:** The main lens (``PowerLaw`` + ``ExternalShear``) and
  the background source (``SersicCore``), identical to ``simulator.py``.
- **Convert to Padded Arrays:** Pack the ``Galaxy`` list from ``LOSSampler`` into fixed-shape
  JAX arrays with boolean masks for unused slots.
- **Scaling Matrix:** Precompute the cosmological scaling factors between all redshift planes.
- **Closure Functions:** Wrap the lens and source galaxies as pure functions of the grid, so
  they can be called inside ``jax.jit``.
- **Single Realisation:** ``simulate_substructure`` produces one noisy lensed image.
- **Batched Realisations:** ``batched_simulate_substructure`` uses ``jax.vmap`` to produce
  many images at once.

__Model__

This script simulates ``Imaging`` of a galaxy-scale strong lens where:

 - The lens galaxy's total mass distribution is a ``PowerLaw`` and ``ExternalShear``.
 - The source galaxy's light is a ``SersicCore``.
 - Line-of-sight halos are ``NFWTruncatedSph`` profiles on multiple redshift planes.
 - Each redshift plane includes a ``MassSheet`` with negative kappa.
"""

from autoconf import jax_wrapper

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
import jax
import jax.numpy as jnp

import autoarray as aa
import autogalaxy as ag
import autolens as al
from autolens.lens import substructure_util
from autolens.lens.los import LOSSampler, los_planes_from

"""
__LOS Configuration__

Parameters controlling the line-of-sight halo population. These are identical to those in
``simulator.py`` and are described in detail there.
"""
z_lens = 0.5
z_source = 1.0

planes_before_lens = 4
planes_after_lens = 4

m_min = 1e7
m_max = 1e10

cone_radius_arcsec = 5.0
c_scatter = 0.15
truncation_factor = 100.0

seed = 42

"""
__Sample LOS Halos__

The ``LOSSampler`` draws halo masses, positions and concentrations from a cosmological mass
function, converts each halo to an ``NFWTruncatedSph`` profile, and adds a compensatory
negative kappa ``MassSheet`` to each plane. See ``simulator.py`` for the full explanation
of the sampling pipeline and the mass function coefficients.
"""
_, plane_centres = los_planes_from(
    z_lens=z_lens,
    z_source=z_source,
    planes_before_lens=planes_before_lens,
    planes_after_lens=planes_after_lens,
)

n_planes = len(plane_centres)

try:
    from autolens.lens.los import mass_function_ab_from, mass_concentration_ab_from
    from astropy.cosmology import Planck15 as astropy_planck15

    mass_function_coefficients = np.zeros((n_planes, 2))
    mass_concentration_coefficients = np.zeros((n_planes, 2))

    for i, z in enumerate(plane_centres):
        mass_function_coefficients[i] = mass_function_ab_from(
            redshift=z, cosmology_astropy=astropy_planck15
        )
        mass_concentration_coefficients[i] = mass_concentration_ab_from(redshift=z)

except ImportError:
    mass_function_coefficients = np.tile([-1.9, 8.0], (n_planes, 1))
    mass_concentration_coefficients = np.tile([-3.0, 40.0], (n_planes, 1))

from autogalaxy.cosmology import Planck15

cosmology = Planck15()

sampler = LOSSampler(
    z_lens=z_lens,
    z_source=z_source,
    planes_before_lens=planes_before_lens,
    planes_after_lens=planes_after_lens,
    m_min=m_min,
    m_max=m_max,
    cone_radius_arcsec=cone_radius_arcsec,
    c_scatter=c_scatter,
    truncation_factor=truncation_factor,
    cosmology=cosmology,
    mass_function_coefficients=mass_function_coefficients,
    mass_concentration_coefficients=mass_concentration_coefficients,
    seed=seed,
)

los_galaxies = sampler.galaxies_from()

n_halos = sum(
    1
    for g in los_galaxies
    if hasattr(g, "mass") and isinstance(g.mass, al.mp.NFWTruncatedSph)
)

print(f"Sampled {n_halos} LOS halos across {n_planes} planes.")

"""
__Grid__

Define the 2D grid on which the image is evaluated and simulated. For the JAX path, we also
extract the raw ``(M, 2)`` coordinate array, because the ``simulate_substructure`` function
operates on plain JAX arrays rather than autoarray grid objects.
"""
grid = al.Grid2D.uniform(
    shape_native=(101, 101),
    pixel_scales=0.05,
)

grid_array = jnp.array(grid.array)
image_shape = grid.shape_native

"""
__PSF__

A Gaussian PSF kernel, normalised to unit sum. For the JAX path this is a plain 2D array
rather than an ``al.Convolver`` object, because ``simulate_substructure`` uses
``jax.scipy.signal.fftconvolve`` directly.
"""
psf_sigma = 0.05
psf_size = 13
half = psf_size // 2
y = np.arange(psf_size) - half
x = np.arange(psf_size) - half
yy, xx = np.meshgrid(y, x, indexing="ij")
psf_kernel_np = np.exp(
    -(yy**2 + xx**2) / (2 * (psf_sigma / grid.pixel_scales[0]) ** 2)
)
psf_kernel_np /= psf_kernel_np.sum()
psf_kernel = jnp.array(psf_kernel_np)

"""
__Lens Galaxy and Source Galaxy__

The lens galaxy and source galaxy are identical to those in ``simulator.py``.

The lens galaxy's total mass distribution is a ``PowerLaw`` (the dominant smooth mass component)
plus an ``ExternalShear`` that accounts for tidal perturbations from the large-scale environment.

The source galaxy's light distribution is a ``SersicCore``, which is a Sersic profile with a
flattened central core.
"""
lens_galaxy = al.Galaxy(
    redshift=z_lens,
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.059, -0.027),
        slope=2.264,
        einstein_radius=1.6,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.0, gamma_2=0.0),
)

source_galaxy = al.Galaxy(
    redshift=z_source,
    bulge=al.lp.SersicCore(
        centre=(0.02, -0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=75.0),
        intensity=1.5,
        effective_radius=0.15,
        sersic_index=3.5,
        radius_break=0.025,
    ),
)

"""
__Convert to Padded Arrays__

The ``LOSSampler`` returns a list of ``Galaxy`` objects. The JAX path needs fixed-shape arrays
so that ``jax.jit`` can compile the simulation once and reuse it for different halo populations
without recompiling.

``galaxies_to_halo_arrays`` extracts the lensing parameters (centre, kappa_s, scale_radius,
truncation_radius) from every ``NFWTruncatedSph`` halo and the negative kappa from every
``MassSheet``, then pads each plane's halo list to a fixed maximum ``max_n``. Unused slots
have their mask set to ``False`` so they contribute zero deflection.

The full list of redshift planes includes the LOS plane centres plus the source redshift. If
the lens redshift coincides with a LOS plane centre it is included automatically; otherwise it
is added so that the lens galaxy's deflections are applied at the correct redshift.
"""
plane_redshifts = sorted(set(list(plane_centres) + [z_lens, z_source]))

max_n = 200

halo_params, halo_mask, sheet_kappas = substructure_util.galaxies_to_halo_arrays(
    galaxies=los_galaxies,
    plane_redshifts=plane_redshifts,
    max_n=max_n,
    profile_cls=al.mp.NFWTruncatedSph,
)

n_active = int(halo_mask.sum())
print(f"Padded to max_n={max_n} per plane ({n_active} active slots across all planes).")

"""
__Scaling Matrix__

The cosmological scaling factors between every pair of redshift planes are precomputed once,
outside ``jax.jit``. This ``(n_planes, n_planes)`` matrix encodes how deflections at one plane
propagate to subsequent planes via the angular diameter distances, and is a constant input to
the compiled simulation function.
"""
scaling_matrix = substructure_util.precompute_scaling_matrix(
    plane_redshifts=plane_redshifts,
    cosmology=cosmology,
)

"""
__Closure Functions__

The lens galaxy and source galaxy are wrapped as pure functions of the grid so they can be
called inside ``jax.jit``. Because the galaxy objects are captured in the closure (rather than
passed as array arguments), JAX traces through their profile methods once during compilation
and bakes the result into the XLA graph. Their parameters are constants — they do not change
between realisations.

The ``lens_plane_mask`` is a float array with ``1.0`` at the lens-galaxy plane and ``0.0``
elsewhere. Inside the scan, the lens galaxy's deflections are computed at every plane step but
multiplied by this mask so they only contribute at the correct redshift.
"""


def lens_deflections_fn(grid_raw):
    g = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    return lens_galaxy.deflections_yx_2d_from(grid=g, xp=jnp).array


def source_image_fn(grid_raw):
    g = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    return source_galaxy.image_2d_from(grid=g, xp=jnp).array


lens_plane_mask = jnp.array(
    [1.0 if abs(z - z_lens) < 1e-6 else 0.0 for z in plane_redshifts]
)

"""
__Single Realisation__

``simulate_substructure`` compiles the full pipeline under ``jax.jit``:

 1. Multi-plane ray tracing via ``jax.lax.scan`` over redshift planes, with halo deflections
    computed by ``jax.vmap`` across all halos on each plane.
 2. Source light evaluation on the final (source-plane) traced grid.
 3. PSF convolution via ``jax.scipy.signal.fftconvolve``.
 4. Poisson noise using a ``jax.random.PRNGKey``.

Passing ``prng_key=None`` skips the noise step and returns the clean lensed-and-convolved image.
"""
key = jax.random.PRNGKey(seed)

image = substructure_util.simulate_substructure(
    grid=grid_array,
    image_shape=image_shape,
    halo_params=halo_params,
    halo_mask=halo_mask,
    scaling_matrix=scaling_matrix,
    macro_deflections_fn=lens_deflections_fn,
    macro_plane_mask=lens_plane_mask,
    sheet_kappas=sheet_kappas,
    source_image_fn=source_image_fn,
    psf_kernel=psf_kernel,
    exposure_time=8000.0,
    background_sky_level=0.1,
    prng_key=key,
    halo_profile_cls=al.mp.NFWTruncatedSph,
)

print(f"Single image shape: {image.shape}, max intensity: {float(jnp.max(image)):.4f}")

"""
__Batched Realisations__

``batched_simulate_substructure`` uses ``jax.vmap`` to produce many images at once. Each
realisation in the batch has a different halo population (drawn by running ``LOSSampler`` with
a different seed) and a different noise realisation (from a different ``PRNGKey``).

The grid, PSF, lens galaxy, source galaxy and scaling matrix are shared across the batch — only
the halo parameters, masks, sheet kappas and noise keys vary.

``los_realizations_to_arrays`` is a convenience helper that runs ``galaxies_to_halo_arrays``
on each realisation and stacks the results into batch-dimensioned arrays.
"""
batch_size = 8

realization_galaxies = []
for i in range(batch_size):
    sampler_i = LOSSampler(
        z_lens=z_lens,
        z_source=z_source,
        planes_before_lens=planes_before_lens,
        planes_after_lens=planes_after_lens,
        m_min=m_min,
        m_max=m_max,
        cone_radius_arcsec=cone_radius_arcsec,
        c_scatter=c_scatter,
        truncation_factor=truncation_factor,
        cosmology=cosmology,
        mass_function_coefficients=mass_function_coefficients,
        mass_concentration_coefficients=mass_concentration_coefficients,
        seed=100 + i,
    )
    realization_galaxies.append(sampler_i.galaxies_from())

hp_batch, hm_batch, sk_batch = substructure_util.los_realizations_to_arrays(
    realization_galaxies=realization_galaxies,
    plane_redshifts=plane_redshifts,
    max_n=max_n,
    profile_cls=al.mp.NFWTruncatedSph,
)

keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

images_batch = substructure_util.batched_simulate_substructure(
    grid=grid_array,
    image_shape=image_shape,
    halo_params_batch=hp_batch,
    halo_mask_batch=hm_batch,
    scaling_matrix=scaling_matrix,
    macro_deflections_fn=lens_deflections_fn,
    macro_plane_mask=lens_plane_mask,
    sheet_kappas_batch=sk_batch,
    source_image_fn=source_image_fn,
    psf_kernel=psf_kernel,
    exposure_time=8000.0,
    background_sky_level=0.1,
    prng_keys=keys,
    halo_profile_cls=al.mp.NFWTruncatedSph,
)

print(f"Batch of {batch_size} images, shape: {images_batch.shape}")

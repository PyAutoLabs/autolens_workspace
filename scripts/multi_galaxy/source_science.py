"""
Source Science: Multi Galaxy
============================

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or
galaxies).

Using a source galaxy model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

This example shows how to perform these calculations for a **multi-galaxy** lens, where two (or more) galaxies of
comparable mass both contribute significantly to the lensing. The API is identical to the galaxy-scale case (see
`imaging/source_science.py`); what changes is the lens itself, and therefore the magnification.

That change is worth being explicit about, because it is the reason multi-galaxy lenses are scientifically
valuable for source science. Two co-dominant deflectors produce a larger and more structured caustic than either
galaxy alone, so a source behind such a system is typically **more highly magnified** than it would be behind a
single galaxy of the same total mass — which is what makes these systems useful for studying faint, high-redshift
sources.

__Contents__

- **Simulated Dataset:** Load and plot the `simple` multi-galaxy dataset (auto-simulating if absent).
- **Mask:** Define the 2D mask applied to the dataset.
- **Source Values:** The true lens and source model used to simulate the dataset.
- **Source Flux:** The total flux of the source galaxy.
- **Source Magnification:** The overall magnification of the source behind two deflectors.
- **Single Deflector Comparison:** How much of the magnification each deflector is responsible for.
- **Tracer:** Computing these quantities from a `max_log_likelihood_tracer`.
- **Parametric Source Models:** What you need from lens modeling to do this on real data.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Simulated Dataset__

We load and plot the `simple` multi-galaxy example dataset, which is simulated imaging of a close pair of
co-dominant deflectors lensing a single background source. We will use it to demonstrate source science
calculations.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We apply a 3.0 arcsecond circular mask and apply it to the `Imaging` object.

Source science calculations are typically performed on masked datasets to ensure only the lensed source is used
in the calculations. For a multi-galaxy lens this mask must enclose the Einstein ring of the *combined* mass
distribution, which wraps around the pair as a whole.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Source Values__

Source science calculations for real lenses are performed using the best-fitting model inferred from a dataset,
and this example demonstrates how to use this below.

However, for simplicity, we demonstrate these calculations using the Sersic source model used to simulate the
dataset, which we refer to as the "true" source model. When analysing real strong lenses, a true underlying model
is not known, but for simulated datasets it is.

This allows us to illustrate the calculations in a way that does not depend on the specific details of the data or
on assumptions about how the lens model is inferred.

The galaxies below correspond to the same tracer used to simulate the `simple` dataset, and therefore represent the
true model — two co-dominant deflectors with comparable Einstein radii (1.0" and 0.8"), an external shear held at
the system centre rather than on either galaxy, and one source.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer = al.Tracer(galaxies=[lens_galaxy_0, lens_galaxy_1, shear_galaxy, source_galaxy])

"""
By plotting the image of the tracer, we confirm it looks like the simulated dataset but does not have CCD imaging
features such as noise or blurring from a PSF.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Source Flux__

A key quantity for a source galaxy is its total flux, which can be used to compute magnitudes (see
`autolens_workspace/*/guides/units/flux` for more details on this).

The most simple way to compute the total flux of a light profile is to create a grid of (y,x) coordinates over which
we compute the image of the light profile, and then sum the image.

The units of the light profile `intensity` are the units of the data the light profile was fitted to. In this example
we will assume everything is in electrons per second (`e- s^-1`), which is typical for Hubble Space Telescope imaging
data.

Note that the source's intrinsic flux is a property of the **source**, so it does not depend on how many deflectors
lens it. It is the magnification below that the multi-galaxy nature of the lens changes.
"""
print(f"Source Galaxy's Intensity {source_galaxy.bulge.intensity} e- s^-1")

"""
The total flux, in units of `e- s^-1`, is computed by summing the image of the light profile over all pixels.

Note that we can use a `grid` of any shape and pixel scale here, the important thing is that it is so large
and high enough resolution that it captures all the light from the light profile.

Note that we are using the source galaxy's true light profile, which corresponds to its emission in the
source-plane. For real datasets, we have to infer this via lens modeling.
"""
grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

print(f"Total Source Flux: {total_flux} e- s^-1")

"""
__Source Magnification__

The overall magnification of the source is estimated as the ratio of total surface brightness in the image-plane and
total surface brightness in the source-plane.

Note that the surface brightness is different to the total flux above, as surface brightness is flux per unit area.
We therefore explicitly mention how area folds into the calculation below.

To ensure the magnification is stable and that we resolve all source emission in both the image-plane and
source-plane we use a very high resolution grid, higher than we used to compute the total flux above.
"""
grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.03)

"""
We repeat our calculation of the source's total flux in the source-plane using this higher resolution grid, note
that we do not take the area into account, the reason for this is explained below.
"""
image = source_galaxy.bulge.image_2d_from(grid=grid)

total_source_plane_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

"""
We now need the total flux of the lensed source in the image-plane, that is how much flux we measure after
gravitational lensing.

To calculate this, we first ray-trace the grid above from the image-plane to the source-plane using the tracer
and then pass it to the source galaxy's light profile to compute the lensed image.

The tracer sums both deflectors' deflection fields internally, so this single line accounts for the lensing of the
pair as a whole.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

source_plane_grid = traced_grid_list[1]

lensed_source_image = source_galaxy.bulge.image_2d_from(grid=source_plane_grid)

total_image_plane_flux = np.sum(
    lensed_source_image
)  # in units e- s^-1 as summed over pixels

"""
We now take the ratio of the total image-plane flux to source-plane flux to estimate the magnification.

Because both fluxes were computed on grids with the same total area and area per pixel, we do not need to
explicitly account for area in this calculation. This is because the area terms cancel out when taking the ratio.
Were the grid areas different, we would need to include area terms in the calculation.
"""
source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Magnification: {source_magnification}")

"""
__Single Deflector Comparison__

For a multi-galaxy lens it is instructive to ask how much of that magnification each deflector is responsible for.

Below we recompute the magnification four times: with each deflector acting alone, with the full pair, and finally
with the pair plus the external shear. This is a calculation you cannot do for a galaxy-scale lens, and it makes
concrete why the multi-galaxy regime is not simply "a slightly bigger lens galaxy".

The magnification of the pair is **not** the sum of the individual magnifications. Magnification depends
non-linearly on the deflection field (it is set by the Jacobian of the lens mapping), so two co-dominant deflectors
reshape the caustic structure rather than just deepening one galaxy's potential. This is the mechanism behind the
strong magnifications these systems deliver.

The first three cases below **exclude** the external shear, so the only thing varying is which deflectors are
present. A fourth case adds the shear back, which reconciles this comparison with the headline magnification
computed above and shows that the shear is not a negligible part of the total.

Each case computes its own source-plane flux on the same grid, so every magnification is a self-consistent ratio.
We use a coarser grid than above (same sky area, fewer pixels), which is ample for comparing ratios and keeps this
illustrative calculation fast — so the fourth number below will be close to, but not exactly equal to, the headline
value.
"""
grid_comparison = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.06)

source_plane_flux_comparison = float(
    np.sum(source_galaxy.bulge.image_2d_from(grid=grid_comparison))
)


def magnification_from(mass_profiles) -> float:
    """
    Compute the total magnification of the source for a given list of lens mass profiles.

    Each mass profile is placed in its own lens galaxy at the lens redshift, the comparison grid is
    ray-traced through them, and the lensed source flux is divided by the unlensed source-plane flux.
    """
    lens_galaxies = [al.Galaxy(redshift=0.5, mass=mass) for mass in mass_profiles]

    tracer_local = al.Tracer(galaxies=lens_galaxies + [source_galaxy])

    traced_grid_local = tracer_local.traced_grid_2d_list_from(grid=grid_comparison)[1]

    lensed_image_local = source_galaxy.bulge.image_2d_from(grid=traced_grid_local)

    return float(np.sum(lensed_image_local) / source_plane_flux_comparison)


mass_0 = lens_galaxy_0.mass
mass_1 = lens_galaxy_1.mass
shear = shear_galaxy.shear

magnification_lens_0 = magnification_from(mass_profiles=[mass_0])
magnification_lens_1 = magnification_from(mass_profiles=[mass_1])
magnification_both = magnification_from(mass_profiles=[mass_0, mass_1])
magnification_both_shear = magnification_from(mass_profiles=[mass_0, mass_1, shear])

print(f"Magnification (lens_0 alone)     : {magnification_lens_0:.3f}")
print(f"Magnification (lens_1 alone)     : {magnification_lens_1:.3f}")
print(f"Magnification (both deflectors)  : {magnification_both:.3f}")
print(f"Magnification (both + shear)     : {magnification_both_shear:.3f}")
print(
    f"Sum of individual magnifications : {magnification_lens_0 + magnification_lens_1:.3f} "
    "(note this does NOT equal the pair's magnification)"
)

"""
The pair magnifies the source several times more strongly than the sum of what each galaxy would achieve alone. Two
co-dominant deflectors are therefore not equivalent to one bigger galaxy, nor to two independent lenses — their
combined potential creates an extended caustic structure that neither produces by itself.

This is the quantitative reason multi-galaxy lenses are prized for studying faint, high-redshift sources.

__Tracer__

Lens modeling returns a `max_log_likelihood_tracer`, which is likely the object you have at hand to compute
source science calculations for real datasets.

The code below shows how using a tracer, composed of any combination of lens and source galaxies, we can
compute the source flux and magnification. It reproduces the calculations above, and works unchanged regardless of
how many deflectors the tracer contains.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[1]

lensed_source_image = tracer.planes[1].image_2d_from(grid=source_plane_grid)
source_plane_image = tracer.planes[1].image_2d_from(grid=image_plane_grid)

total_image_plane_flux = np.sum(lensed_source_image)
total_source_plane_flux = np.sum(source_plane_image)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Plane Total Flux via Tracer: {total_source_plane_flux} e- s^-1")
print(f"Source Magnification via Tracer: {source_magnification}")

"""
Note that `tracer.planes` groups galaxies by **redshift**, not one plane per galaxy. Both deflectors here are at
z=0.5, so `tracer.planes[0]` contains them **both** (plus the shear galaxy) and `tracer.planes[1]` is the source.
This is why the code above is identical to the galaxy-scale version — the source is always the last plane.
"""
print(f"number of planes = {len(tracer.planes)}")
print(f"entries in the lens plane = {len(tracer.planes[0])}")

"""
__Parametric Source Models__

If your lens modeling uses a parametric source model (e.g. Sersic, Multi Gaussian Expansion), the only object
you need to perform source science calculations is the `max_log_likelihood_tracer` returned by lens modeling.

Alternatively, as done above, you can manually set up a tracer using the lens and source galaxies inferred
by lens modeling.

Therefore, you may now wish to go to your results, extract the `max_log_likelihood_tracer`, and use it to compute
the source flux and magnification as shown above.

One caution specific to this regime: the magnification depends on the *total* deflection field, which a
multi-galaxy fit constrains well, but it also depends on how mass is split between the deflectors, which is
constrained less well (see the corner plot discussion in `multi_galaxy/modeling.py`). When quoting a magnification
for a multi-galaxy lens, propagate the posterior rather than using only the maximum likelihood tracer.

Where to go next:

- `autolens_workspace/*/guides/units/flux`: converting fluxes to magnitudes and physical units.
- `autolens_workspace/*/multi_galaxy/modeling`: inferring the lens model these calculations depend on.
- `autolens_workspace/*/imaging/source_science`: the galaxy-scale version of this guide, including pixelized
  source reconstructions.
"""

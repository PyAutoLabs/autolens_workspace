"""
__Log Likelihood Function: Multi Galaxy (Parametric)__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Imaging` data of a
**multi-galaxy** strong lens, where two (or more) galaxies of comparable mass both contribute significantly to the
lensing of a single background source.

The likelihood function is the same one used at galaxy scale (see `imaging/likelihood_function.py`), with one
addition that is the defining feature of this regime: **the deflection field is the sum of every deflector's
field**. This script makes that summation an explicit step so you can see exactly where the extra deflectors enter
the calculation, and confirm that nothing else about the likelihood changes.

This script has the following aims:

 - To provide a resource that authors can include in papers using, so that readers can understand the likelihood
 function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

Accompanying this script is the `contributor_guide.py` which provides URL's to every part of the source-code that
is illustrated in this guide. This gives contributors a sequential run through of what source-code functions,
modules and packages are called when the likelihood is evaluated.

__Contents__

- **JAX:** JAX acceleration of this likelihood function.
- **Dataset:** Load the multi-galaxy dataset (auto-simulating if absent).
- **Extra Galaxies Noise Scaling:** Scale the noise of nearby contaminating galaxies so they do not impact the fit.
- **Mask:** Standard set up of the mask that is fitted.
- **Over Sampling:** Set up the over-sampling used for light profile evaluation.
- **Masked Image Grid:** To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.
- **Main Lens Galaxy Light (Setup):** The light profiles which represent each deflector's light.
- **Main Lens Galaxy Mass:** The mass profiles which represent each deflector's mass.
- **Summed Deflection Field:** The multi-galaxy step — every deflector's deflection field is added together.
- **Main Lens Galaxies:** Combine each deflector's light and mass into `Galaxy` objects.
- **Source Galaxy Light Profile:** The source galaxy is fitted using another analytic light profile.
- **Lens Light:** Compute a 2D image of the combined light of every deflector.
- **Ray Tracing:** Ray-trace every image-plane coordinate to the source-plane using the summed deflections.
- **Source Image:** Evaluate the source galaxy's 2D image on the traced grid.
- **Convolution:** Convolve the 2D image of the lenses and source with the PSF.
- **Likelihood Function:** Quantify the goodness-of-fit of the lens and source model.
- **Chi Squared:** The chi-squared term of the likelihood function.
- **Noise Normalization Term:** The noise normalization term of the likelihood function.
- **Calculate The Log Likelihood:** Combine the two terms to compute the `log_likelihood`.
- **Fit:** Confirm the step-by-step calculation matches the `FitImaging` object.
- **Lens Modeling:** How this likelihood function is sampled by a non-linear search.
- **Wrap Up:** Summary of the script and next steps.

__JAX__

The step-by-step likelihood you walk through below can be JAX-accelerated, which is how the modeling scripts run it
in practice. For the JAX patterns — wrapping a likelihood in `@jax.jit`, going via the `Analysis` object, and
validating the JAX path against the NumPy result computed here — see `autolens_workspace/*/guides/using_jax.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autoarray as aa
import autolens.plot as aplt

"""
__Dataset__

In order to perform a likelihood evaluation, we first load a dataset.

This example fits the simulated multi-galaxy lens `simple`, which is HST-resolution (0.05 arcsecond-per-pixel)
imaging of a close pair of co-dominant deflectors.
"""
dataset_path = Path("dataset", "multi_galaxy", "simple")

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

"""
This guide uses in-built visualization tools for plotting.

For example, using the `aplt.subplot_imaging_dataset` the imaging dataset we perform a likelihood evaluation on is
plotted.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxies Noise Scaling__

Before masking, we must deal with any extra galaxies in the data: nearby galaxies (or foreground stars, or
data-reduction artefacts) whose emission is not associated with the strong lens but blends into the field. If
their light is left in the data it will contaminate the likelihood evaluation and bias the inferred lens model.

For a multi-galaxy lens this carries a judgement the galaxy-scale case does not: a galaxy that contributes
significantly to the lensing is a co-dominant *deflector* and belongs in the model below, with its own mass
profile. A galaxy whose light merely blends into the field is a *contaminant* and belongs here.

To prevent extra galaxies from impacting the fit, we do not mask them entirely from the fit. Instead, the pixels
are kept in the fit but their data values are scaled to zero and their noise-map values increased to very large
values, so they contribute negligibly to the likelihood. This is preferable to removing the pixels entirely
(e.g. for a pixelized source reconstruction, removing pixels can produce discontinuities in the pixelization).

The `simple` dataset includes a faint extra galaxy, and a `mask_extra_galaxies.fits` covering it is shipped with
the dataset (created by the simulator). If you are modeling your own data with an extra galaxy, you must either
create such a mask using the data-preparation tools (the GUI at the end of `multi_galaxy/start_here.py`, or
`imaging/data_preparation/gui/mask_extra_galaxies.py`), or shrink the circular mask below so the extra galaxy lies
outside it and is removed from the fit entirely.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,  # `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

The likelihood is only evaluated using image pixels contained within a 2D mask, which we choose before performing
lens modeling.

Below, we define a 2D circular mask with a 3.0" radius. For a multi-galaxy lens this must enclose the Einstein ring
of the *combined* mass distribution (~1.8" here), which wraps around the pair as a whole, rather than being sized
from either galaxy's own light.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
When we plot the masked imaging, only the circular masked region is shown.
"""
aplt.subplot_imaging_dataset(dataset=masked_dataset)

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, we disable over sampling in this guide by setting `sub_size=1`.

A full description of over sampling and how to use it is given in
`autolens_workspace/*/guides/advanced/over_sampling.py`.
"""
masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

These are given by `masked_dataset.grids.lp`, which we can plot and see is a uniform grid of (y,x) Cartesian
coordinates which have had the 3.0" circular mask applied.

Each (y,x) coordinate corresponds to the centre of each image-pixel in the dataset, meaning that when this grid is
used to perform ray-tracing and evaluate a light profile, the intensity of the profile at the centre of each
image-pixel is computed, making it straight forward to compare the light profile's image to the image data.
"""
aplt.plot_grid(grid=masked_dataset.grids.lp, title="")

print(
    f"(y,x) coordinates of first ten unmasked image-pixels {masked_dataset.grid[0:9]}"
)

"""
To perform lensing calculations we convert this 2D (y,x) grid of coordinates to elliptical coordinates:

 $\\eta = \\sqrt{(x - x_c)^2 + (y - y_c)^2/q^2}$

Where:

 - $y$ and $x$ are the (y,x) arc-second coordinates of each unmasked image-pixel, given by `masked_dataset.grids.lp`.
 - $y_c$ and $x_c$ are the (y,x) arc-second `centre` of the light or mass profile used to perform lensing calculations.
 - $q$ is the axis-ratio of the elliptical light or mass profile (`axis_ratio=1.0` for spherical profiles).
 - The elliptical coordinates is rotated by position angle $\\phi$, defined counter-clockwise from the positive
 x-axis.

This is the first place the multi-galaxy regime differs in practice: each deflector has its **own** centre, axis
ratio and position angle, so this transformation is performed separately per galaxy. Nothing is shared between
them.

$q$ and $\\phi$ are not used to parameterize a light profile but expresses these as "elliptical components",
or `ell_comps` for short:

$\\epsilon_{1} =\\frac{1-q}{1+q} \\sin 2\\phi, \\,\\,$
$\\epsilon_{2} =\\frac{1-q}{1+q} \\cos 2\\phi.$

Note that `Ell` is used as shorthand for elliptical and `Sph` for spherical.
"""
profile = al.EllProfile(centre=(0.35, 0.25), ell_comps=(0.1, 0.2))

"""
Transform `masked_dataset.grids.lp` to the centre of profile and rotate it using its angle `phi`.
"""
transformed_grid = profile.transformed_to_reference_frame_grid_from(
    grid=masked_dataset.grids.lp
)

aplt.plot_grid(grid=transformed_grid, title="")
print(
    f"transformed coordinates of first ten unmasked image-pixels {transformed_grid[0:9]}"
)

"""
Using these transformed (y',x') values we compute the elliptical coordinates
$\\eta = \\sqrt{(x')^2 + (y')^2/q^2}$
"""
elliptical_radii = profile.elliptical_radii_grid_from(grid=transformed_grid)

print(
    f"elliptical coordinates of first ten unmasked image-pixels {elliptical_radii[0:9]}"
)

"""
__Main Lens Galaxy Light (Setup)__

To perform a likelihood evaluation we now compose our lens model.

We first define the light profiles which represent each deflector's light, which will be used to fit the lens
light. In the multi-galaxy regime there is one such profile **per deflector** — every galaxy is a main lens galaxy
with its own free light model, and none of the parameters are shared.

A light profile is defined by its intensity $I (\\eta_{\\rm l}) $, for example the Sersic profile:

$I_{\\rm  Ser} (\\eta_{\\rm l}) = I \\exp \\bigg\\{ -k \\bigg[ \\bigg( \\frac{\\eta}{R} \\bigg)^{\\frac{1}{n}} - 1 \\bigg] \\bigg\\}$

Where:

 - $\\eta$ are the elliptical coordinates (see above) of the masked image-grid.
 - $I$ is the `intensity`, which controls the overall brightness of the Sersic profile.
 - $n$ is the ``sersic_index``, which via $k$ controls the steepness of the inner profile.
 - $R$ is the `effective_radius`, which defines the arc-second radius of a circle containing half the light.

In this example each of our two deflectors is composed of one light profile, an elliptical Sersic representing its
bulge. The values below are those used to simulate the dataset (see `multi_galaxy/simulator.py`), so this
evaluation sits at the likelihood's maximum.
"""
bulge_0 = al.lp.Sersic(
    centre=(0.35, 0.25),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    intensity=1.2,
    effective_radius=0.6,
    sersic_index=4.0,
)

bulge_1 = al.lp.Sersic(
    centre=(-0.35, -0.25),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
    intensity=1.0,
    effective_radius=0.5,
    sersic_index=4.0,
)

"""
Using the masked 2D grid defined above, we can calculate and plot images of each light profile component.

(The transformation to elliptical coordinates above is built into the `image_2d_from` function and performed
implicitly).
"""
aplt.plot_array(
    array=bulge_0.image_2d_from(grid=masked_dataset.grid), title="Lens 0 Bulge Image"
)
aplt.plot_array(
    array=bulge_1.image_2d_from(grid=masked_dataset.grid), title="Lens 1 Bulge Image"
)

"""
__Main Lens Galaxy Mass__

We next define the mass profiles which represent each deflector's mass, which will be used to ray-trace the
image-plane 2D grid of (y,x) coordinates to the source-plane so that the source model can be evaluated.

In this example each deflector is an elliptical isothermal mass distribution. The system also has a single overall
`ExternalShear`, defined below alongside them, which describes the tidal field of everything outside the system
being modeled. Because it is a property of the system as a whole rather than of an individual galaxy, it is kept as
a standalone profile here and later placed in its own galaxy at the system centre.

A mass profile is defined by its convergence $\\kappa (\\eta)$, which is related to the surface density of the mass
distribution as

$\\kappa(\\eta)=\\frac{\\Sigma(\\eta)}{\\Sigma_\\mathrm{crit}},$

where

$\\Sigma_\\mathrm{crit}=\\frac{{\\rm c}^2}{4{\\rm \\pi} {\\rm G}}\\frac{D_{\\rm s}}{D_{\\rm l} D_{\\rm ls}},$

and

 - `c` is the speed of light.
 - $D_{\\rm l}$, $D_{\\rm s}$, and $D_{\\rm ls}$ are respectively the angular diameter distances to the lens, to the
 source, and from the lens to the source.

For readers less familiar with lensing, we can think of $\\kappa(\\eta)$ as a convenient and dimensionless way to
describe how light is gravitationally lensed after assuming a cosmology.

For the isothermal profile:

$\\kappa(\\eta) = \\frac{1.0}{1 + q} \\bigg( \\frac{\\theta_{\\rm E}}{\\eta} \\bigg)$

Where:

 - $\\theta_{\\rm E}$ is the `einstein_radius` (which is rescaled compared to other einstein radius definitions).

Note that the two Einstein radii below (1.0" and 0.8") are comparable. That is what makes this a multi-galaxy lens
rather than a single lens with a minor perturber, and it is why neither mass profile can be neglected.
"""
mass_0 = al.mp.Isothermal(
    centre=(0.30, 0.28),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
    einstein_radius=1.0,
)

mass_1 = al.mp.Isothermal(
    centre=(-0.31, -0.22),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
    einstein_radius=0.8,
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

"""
The convergence of the two mass profiles adds, because convergence is linear in the surface density. Plotting the
summed convergence shows the pair as a single, elongated mass distribution — which is the reason the lensed arcs
wrap around both galaxies rather than forming two separate rings.
"""
convergence_0 = mass_0.convergence_2d_from(grid=masked_dataset.grid)
convergence_1 = mass_1.convergence_2d_from(grid=masked_dataset.grid)

aplt.plot_array(array=convergence_0 + convergence_1, title="Summed Convergence")

"""
__Summed Deflection Field__

From each mass profile we can compute its deflection angles, which describe how due to gravitational lensing
image-pixels are ray-traced to the source plane.

The deflection angles are computed by integrating $\\kappa$:

$\\vec{{\\alpha}}_{\\rm x,y} (\\vec{x}) = \\frac{1}{\\pi} \\int \\frac{\\vec{x} - \\vec{x'}}{\\left | \\vec{x} - \\vec{x'} \\right |^2} \\kappa(\\vec{x'}) d\\vec{x'} \\, ,$

**This is the step that defines the multi-galaxy regime.** Deflection angles add linearly, so the total deflection
field of the system is simply the sum of every deflector's field:

$\\vec{\\alpha}_{\\rm total}(\\vec{x}) = \\sum_{i} \\vec{\\alpha}_{i}(\\vec{x})$

This is valid because both deflectors are at the **same redshift**, so a light ray is deflected by both of them at
the same point along its path. (If the deflectors were at different redshifts the system would be a compound,
multi-plane lens, and the deflections would instead be applied sequentially — see the cluster package.)

Everything downstream of this summation — ray-tracing, evaluating the source, convolution, the chi-squared — is
completely unchanged from the galaxy-scale likelihood function. Below we compute each field and add them.

The `ExternalShear` enters this sum in exactly the same way as a galaxy's mass profile: it contributes its own
deflection field, which is added to the others. It is held in its own galaxy below, at the system centre, because it
acts on the system as a whole rather than on any one deflector.
"""
deflections_0 = mass_0.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_1 = mass_1.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_shear = shear.deflections_yx_2d_from(grid=masked_dataset.grid)

deflections_total = deflections_0 + deflections_1 + deflections_shear

"""
We plot the y and x components of the summed deflection field.
"""
deflections_y = aa.Array2D(
    values=deflections_total.slim[:, 0], mask=masked_dataset.grid.mask
)
aplt.plot_array(array=deflections_y, title="Summed Deflections Y")

deflections_x = aa.Array2D(
    values=deflections_total.slim[:, 1], mask=masked_dataset.grid.mask
)
aplt.plot_array(array=deflections_x, title="Summed Deflections X")

"""
We can confirm numerically that neither deflector dominates — the defining property of co-dominance. If one field
were much smaller than the other, this would be a galaxy-scale lens with a perturber and belong in `imaging/`.
"""
magnitude_0 = float(np.mean(np.linalg.norm(np.asarray(deflections_0.array), axis=-1)))
magnitude_1 = float(np.mean(np.linalg.norm(np.asarray(deflections_1.array), axis=-1)))

print(f'mean |deflection| lens_0 = {magnitude_0:.3f}"  lens_1 = {magnitude_1:.3f}"')
print(f"ratio = {magnitude_1 / magnitude_0:.2f}  (co-dominant: neither is negligible)")

"""
__Main Lens Galaxies__

We now combine each deflector's light and mass profiles into `Galaxy` objects.

When computing quantities for the light and mass profiles from these objects, each individual quantity is computed
and added together. The `Tracer` built further down does the same across galaxies, which is how the deflection
summation shown explicitly above is performed internally.

Neither galaxy carries the `ExternalShear`. It describes the tidal field of structure *outside* the system and is
therefore a property of the system as a whole, so we hold it in its own galaxy at the system centre (0.0", 0.0")
rather than attaching it to a deflector. Its deflection field is added into the sum exactly like a galaxy's, which is
why this is numerically identical to attaching it to `lens_0` — it is simply an honest description of what it is.
"""
lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=bulge_0,
    mass=mass_0,
)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=bulge_1,
    mass=mass_1,
)

"""
__Source Galaxy Light Profile__

The source galaxy is fitted using another analytic light profile, in this example a cored elliptical Sersic.
"""
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

"""
__Lens Light__

Compute a 2D image of the lens light as the sum of every deflector's light profiles.

At galaxy scale this is the image of one galaxy's profiles added together. Here it is the image of **both**
galaxies' profiles added together — the light of a multi-galaxy lens is a single summed image, which is why the
deflectors' blended light cannot be separated by the data alone.
"""
lens_image_2d = lens_galaxy_0.image_2d_from(
    grid=masked_dataset.grid
) + lens_galaxy_1.image_2d_from(grid=masked_dataset.grid)

aplt.plot_array(array=lens_image_2d, title="Summed Lens Light")

"""
To convolve the lens light's 2D image with the imaging data's PSF, we need its `blurring_image`. This represents
all flux values not within the mask, which are close enough to it that their flux blurs into the mask after PSF
convolution.

To compute this, a `blurring_mask` and `blurring_grid` are used, corresponding to these pixels near the edge of the
actual mask whose light blurs into the image:
"""
lens_blurring_image_2d = lens_galaxy_0.image_2d_from(
    grid=masked_dataset.grids.blurring
) + lens_galaxy_1.image_2d_from(grid=masked_dataset.grids.blurring)

"""
__Ray Tracing__

To perform lensing calculations we ray-trace every 2d (y,x) coordinate $\\theta$ from the image-plane to its (y,x)
source-plane coordinate $\\beta$ using the **summed** deflection angles $\\alpha$ of every lens galaxy:

 $\\beta = \\theta - \\alpha_{\\rm total}(\\theta)$

The likelihood function of a source light profile ray-traces two grids from the image-plane to the source-plane:

 1) A 2D grid of (y,x) coordinates aligned with the imaging data's image-pixels.

 2) The 2D blurring grid (used for the lens light above) which accounts for pixels at the edge of the mask whose
 light blurs into the mask.

The `Tracer` below computes the 2D deflection angles of all of its lens galaxies, sums them, and subtracts them
from the image-plane 2D (y,x) coordinates $\\theta$ of each grid, thus ray-tracing their coordinates to the source
plane to compute their $\\beta$ values.
"""
shear_galaxy = al.Galaxy(redshift=0.5, shear=shear)

tracer = al.Tracer(
    galaxies=[lens_galaxy_0, lens_galaxy_1, shear_galaxy, source_galaxy]
)

# A list of every grid (e.g. image-plane, source-plane) however we only need the source plane grid with index -1.
traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

aplt.plot_grid(grid=traced_grid, title="")

traced_blurring_grid = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.blurring
)[-1]

aplt.plot_grid(grid=traced_blurring_grid, title="")

"""
We can confirm that the tracer's internal summation matches the explicit sum we computed above, verifying that
`Tracer` really is just adding the deflection fields of its lens galaxies.
"""
traced_grid_manual = masked_dataset.grid - deflections_total

print(
    "tracer traced grid matches manual summation: "
    f"{np.allclose(np.asarray(traced_grid.slim), np.asarray(traced_grid_manual.slim))}"
)

"""
__Source Image__

We pass the traced grid and blurring grid of coordinates to the source galaxy to evaluate its 2D image.
"""
source_image_2d = source_galaxy.image_2d_from(grid=traced_grid)

source_blurring_image_2d = source_galaxy.image_2d_from(grid=traced_blurring_grid)

aplt.plot_array(array=source_image_2d, title="Lensed Source Image")

"""
__Lens + Source Light Addition__

We add the lens and source galaxy images and blurring images together, to create an overall image of the strong
lens.
"""
image = lens_image_2d + source_image_2d

aplt.plot_array(array=image, title="")

blurring_image_2d = lens_blurring_image_2d + source_blurring_image_2d

aplt.plot_array(array=blurring_image_2d, title="")

"""
__Convolution__

Convolve the 2D image of the lenses and source above with the PSF in real-space (as opposed to via an FFT) using
a `Kernal2D`.
"""
convolved_image_2d = masked_dataset.psf.convolved_image_from(
    image=image, blurring_image=blurring_image_2d
)

aplt.plot_array(array=convolved_image_2d, title="")

"""
__Likelihood Function__

We now quantify the goodness-of-fit of our lens and source model.

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for parametric lens modeling consists of two terms:

 $-2 \\mathrm{ln} \\, \\epsilon = \\chi^2 + \\sum_{\\rm  j=1}^{J} { \\mathrm{ln}} \\left [2 \\pi (\\sigma_j)^2 \\right]  \\, .$

We now explain what each of these terms mean. Note that these terms are entirely independent of how many
deflectors the lens has — the multi-galaxy regime is fully accounted for by the deflection summation above.

__Chi Squared__

The first term is a $\\chi^2$ statistic, which is computed as follows:

 - `model_data` = `convolved_image_2d`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not.

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the
image for, corresponding to a fit with a lower likelihood.
"""
model_image = convolved_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = al.Array2D(values=chi_squared_map, mask=mask)

aplt.plot_array(array=chi_squared_map, title="")

"""
__Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term in the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared.

Given the `noise_map` is fixed, this term does not change during the lens modeling process and has no impact on the
model we infer.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the lens model, by combining the two terms computed above
using the likelihood function defined above.
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(figure_of_merit)

"""
__Fit__

This step-by-step process to perform a likelihood function evaluation is what is performed in the `FitImaging`
object.

We confirm below that the hand-computed figure of merit matches it exactly.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)
fit_figure_of_merit = fit.figure_of_merit
print(fit_figure_of_merit)

print(
    "hand-computed figure of merit matches FitImaging: "
    f"{np.isclose(figure_of_merit, fit_figure_of_merit)}"
)

aplt.subplot_fit_imaging(fit=fit)

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `Nautilus` (https://github.com/johannesulf/nautilus), though
multiple MCMC and optimization algorithms are supported. `multi_galaxy/start_here.py` instead uses a multi-start
gradient optimizer, `af.MultiStartProdigy`.

__Wrap Up__

We have presented a visual step-by-step guide to the multi-galaxy parametric likelihood function, which uses
analytic light profiles to fit the light of every deflector and of the source.

The essential point is how little changes relative to the galaxy-scale likelihood function. Every deflector gets
its own light and mass profiles, their convergences and deflection fields are summed, and from the ray-tracing step
onwards the calculation is identical. The cost of extra deflectors is in the dimensionality of parameter space, not
in the likelihood evaluation.

There are a number of other input features which slightly change the behaviour of this likelihood function, which
are described in additional notebooks found in this package. In brief, these describe:

 - **Sub-gridding**: Oversampling the image grid into a finer grid of sub-pixels, which are all individually
 ray-traced to the source-plane and used to evaluate the light profile more accurately.

 - **Pixelizations**: Reconstructing the source on a mesh of pixels, instead of an analytic light profile.

Where to go next:

- `autolens_workspace/*/multi_galaxy/fit`: the `FitImaging` API and every quantity it computes.
- `autolens_workspace/*/multi_galaxy/modeling`: fitting this model to the data with a non-linear search.
- `autolens_workspace/*/guides/using_jax`: JAX-accelerating a likelihood function.
"""

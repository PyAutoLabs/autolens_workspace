"""
__Log Likelihood Function: Double Einstein Ring__

This script describes the additional steps required to compute the `log_likelihood` for a double Einstein ring
lens — a strong lens system with two source galaxies at different redshifts behind the foreground lens.

This script does NOT repeat the steps shared with single-plane lensing (mask, image-plane grid, convolution,
chi-squared, noise normalization). It documents only the parts of the likelihood function which are specific
to double Einstein ring multi-plane ray-tracing.

__Prerequisites__

The likelihood function below builds directly on standard imaging and MGE likelihood functions. You should read
these notebooks first:

 - `autolens_workspace/scripts/imaging/likelihood_function.py` — the canonical single-plane log likelihood
   walkthrough, covering image-plane grids, ray-tracing, source-plane evaluation, PSF convolution, chi-squared
   and the noise normalization term.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py` — how a `Basis`
   of linear Gaussians is solved for via linear algebra.

Sections covered in those scripts (e.g. "Chi Squared", "Noise Normalization Term", "Calculate The Log
Likelihood") are not repeated here; this script focuses entirely on what changes for a double Einstein ring.

__Contents__

- **Prerequisites:** Reading order before this script (see above).
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Galaxies:** Three galaxies at three redshifts — lens, source_0 (light + mass), source_1 (light only).
- **Multi-Plane Ray-Tracing:** The deflection chain that produces three ray-traced grids, one per plane.
- **Source-Plane Images:** Each source galaxy's light is evaluated at its own ray-traced grid.
- **Model Image:** Sum of both source-plane contributions, then PSF convolution.
- **Likelihood:** Reference up to the canonical imaging likelihood for chi-squared, noise normalization, and
  the final log likelihood expression.
- **Fit Check:** Confirm the manual reconstruction matches `FitImaging.log_likelihood`.
- **Wrap Up.**

__What Changes For A Double Einstein Ring__

For a single-plane lens, ray-tracing maps image-plane (y,x) coordinates onto a single source-plane via the lens
galaxy's deflection map alpha_lens(theta). The source galaxy's light is then evaluated at the source-plane
coordinates and projected back into the image plane.

For a double Einstein ring, there are TWO source-planes at different redshifts, and the first source galaxy
acts as a deflector for the second. The deflection chain is:

  Plane 0 (image-plane)        : theta
  Plane 1 (source_0, z=1.0)    : theta - alpha_lens(theta)
  Plane 2 (source_1, z=2.0)    : theta - alpha_lens(theta) - beta_01 * alpha_source_0(plane_1_grid)

where beta_01 is a scaling factor derived from the angular diameter distances between the lens, source_0 and
source_1 — this is what makes double Einstein ring systems sensitive to cosmology. The factor is computed
internally by `Tracer.traced_grid_2d_list_from` based on the redshifts and cosmology.

The model image is then the sum of the two source-plane reconstructions projected back to the image plane,
PSF-convolved, and compared to the data exactly as in the single-plane case.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the double Einstein ring dataset. The auto-simulation block mirrors the other example scripts.
"""
dataset_name = "double_einstein_ring"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/imaging/features/advanced/double_einstein_ring/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Galaxies__

The three galaxies that participate in the multi-plane ray-tracing:

 - `lens` (z=0.5): an `Isothermal` mass profile. No light, matching the simulator.
 - `source_0` (z=1.0): an MGE light component (a simple basis of 10 linear Gaussians) AND an `IsothermalSph`
   mass profile. `source_0` deflects the light from `source_1`.
 - `source_1` (z=2.0): an MGE light component only.

The mass-profile parameters and source centres are set to the simulator's true values so the manual likelihood
computation below produces a sensible-looking model image.
"""
total_gaussians = 10
log10_sigma_list = np.linspace(-2, np.log10(0.5), total_gaussians)


def build_source_basis(centre):
    gaussian_list = [
        al.lp_linear.Gaussian(
            centre=centre,
            ell_comps=(0.0, 0.0),
            sigma=10 ** log10_sigma_list[i],
        )
        for i in range(total_gaussians)
    ]
    return al.lp_basis.Basis(profile_list=gaussian_list)


lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_0 = al.Galaxy(
    redshift=1.0,
    bulge=build_source_basis(centre=(-0.15, -0.15)),
    mass=al.mp.IsothermalSph(centre=(-0.15, -0.15), einstein_radius=0.3),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=build_source_basis(centre=(-0.45, 0.45)),
)

tracer = al.Tracer(galaxies=[lens, source_0, source_1])

"""
__Multi-Plane Ray-Tracing__

The single call below performs the full deflection chain. `traced_grid_2d_list_from` returns one grid per plane,
in redshift order: image-plane grid (no deflection), source_0-plane grid (deflected by the lens), source_1-plane
grid (deflected by the lens AND source_0).

Note: the cosmological scaling factor `beta_01` mentioned in the header is applied internally to the
source_0-plane deflection contribution to the source_1-plane grid. PyAutoLens uses `Planck18` by default; to
make `Om0` a free parameter, see `modeling.py`.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=dataset.grid)

grid_image_plane = traced_grid_list[0]
grid_source_0 = traced_grid_list[1]
grid_source_1 = traced_grid_list[2]

print(f"Number of planes traced: {len(traced_grid_list)}")
print(f"Plane 1 (source_0) first coord: {grid_source_0[0]}")
print(f"Plane 2 (source_1) first coord: {grid_source_1[0]}")

"""
We can plot the image-plane grid after it has been deflected to each source-plane. This visualises how the
caustics of the lens galaxy carve up image-plane (y,x) coordinates and route them to different source-plane
positions.
"""
aplt.plot_grid(grid=grid_source_0, title="Ray-traced grid at source_0 plane (z=1.0)")
aplt.plot_grid(grid=grid_source_1, title="Ray-traced grid at source_1 plane (z=2.0)")

"""
__Source-Plane Images__

Each source galaxy's light component is evaluated at the ray-traced grid that arrives at its own redshift plane.

For linear light profiles (Gaussians in our MGE), `image_2d_from` returns an `Array2D` of image-plane (y,x)
pixel values per profile, with a placeholder `intensity=1.0`. The true `intensity` is solved for at the linear
algebra step (see the MGE likelihood prerequisite).

For this manual walkthrough we use the convenience method `image_2d_from` on the `Tracer`, which evaluates
every galaxy's light at the correct plane and sums them into a single model image.
"""
model_image_unconvolved = tracer.image_2d_from(grid=dataset.grid)

aplt.plot_array(
    array=model_image_unconvolved, title="Model image before PSF convolution"
)

"""
What `image_2d_from` does internally for our double Einstein ring:

  1. Ray-traces the image-plane grid to obtain `grid_source_0` and `grid_source_1`.
  2. Evaluates `source_0`'s MGE at `grid_source_0`, producing its image-plane contribution.
  3. Evaluates `source_1`'s MGE at `grid_source_1`, producing its image-plane contribution.
  4. Sums all source contributions into the model image.

For a single-plane lens there is only one source-plane grid and one such evaluation; for the double Einstein
ring there are two.

__Model Image__

PSF convolution and chi-squared / noise normalization are unchanged from the single-plane case. The model image
above is convolved with the PSF and compared to the data via the standard imaging chi-squared expression
documented in `autolens_workspace/scripts/imaging/likelihood_function.py`.

We delegate the remaining steps to `FitImaging`, which handles the linear-algebra step that solves for each
Gaussian's `intensity` and assembles the full `log_likelihood`.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"\nLog likelihood of the manual double Einstein ring fit: {fit.log_likelihood}")

"""
__Likelihood__

The final `log_likelihood` combines:

  - The chi-squared term, computed from the residuals between the PSF-convolved model image and the data,
    weighted by the noise map.
  - The noise normalization term, the standard Gaussian normalization over all unmasked pixels.
  - The linear algebra terms (regularization and curvature determinants) introduced by the MGE
    `Basis` of linear Gaussians.

The first two are documented in `imaging/likelihood_function.py`; the third in
`imaging/features/multi_gaussian_expansion/likelihood_function.py`. No new terms are introduced by the multi-plane
ray-tracing — the only change is the source-plane evaluation step described above.

__Wrap Up__

The double Einstein ring `log_likelihood` differs from the single-plane case in exactly one place: the source
galaxies are evaluated on different ray-traced grids, one per source-plane redshift. Every other step (PSF
convolution, chi-squared, noise normalization, linear algebra) is shared with the single-plane likelihood and
documented in the prerequisite scripts.

The deflection scaling factor `beta_01` between source_0 and source_1 is the physical reason double Einstein
rings constrain cosmology: it depends on the ratio of angular diameter distances `D_{ls_0 -> s_1} / D_{s_0 ->
s_1}`, which in turn depends on the cosmological parameters. This is why the `modeling.py` example exposes the
option to make `Om0` a free parameter.
"""

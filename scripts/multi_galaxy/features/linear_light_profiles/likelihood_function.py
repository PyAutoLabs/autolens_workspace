"""
__Log Likelihood Function: Linear Light Profiles (Multi Galaxy)__

This script walks through the log likelihood function of a multi-galaxy lens fitted with **linear light profiles**,
step by step, with an emphasis on the one step that behaves differently when there are two co-dominant deflectors.

__Prerequisites__

Read these first; this script does not repeat them:

 - `multi_galaxy/likelihood_function.py` — the multi-galaxy likelihood with standard light profiles, including the
   summed deflection field and ray tracing. Everything up to the light evaluation is identical here.
 - `imaging/features/linear_light_profiles/likelihood_function.py` — the full derivation of the linear inversion
   at galaxy scale: the mapping matrix, the data vector $D$, the curvature matrix $F$, and the reconstruction.
   The equations are not re-derived here.

__Contents__

- **What Changes:** The single step of the likelihood that the linear solve replaces.
- **Dataset, Mask & Over Sampling:** Set up, identical to `multi_galaxy/likelihood_function.py`.
- **Tracer:** The two co-dominant deflectors, shear and source, with linear light profiles.
- **Linear Object Lists:** One for the image plane holding *both* deflectors, one for the source plane.
- **Blurred Mapping Matrix (f):** Three columns, not two.
- **Data Vector (D):** One entry per linear light profile.
- **Curvature Matrix (F):** The multi-galaxy step — and the number worth taking away.
- **Reconstruction:** Solving for the intensities.
- **Log Likelihood:** Assembling the final value.
- **Wrap Up:** Where to go next.

__What Changes__

The likelihood of a multi-galaxy lens with standard light profiles goes: evaluate each deflector's light, sum the
deflectors' deflection fields, ray-trace, evaluate the source, add everything, convolve, compare to the data.

With linear light profiles, one step is replaced. Instead of *evaluating* each light profile at a given
`intensity`, the profiles' unit-intensity images are assembled into a matrix and the intensities that maximize the
likelihood are solved for in closed form.

At galaxy scale that linear system has two unknowns: the lens's intensity and the source's. In a multi-galaxy lens
it has one per deflector plus the source — three here. That is the whole difference, and the interesting part of
it is not the extra column but what the extra column does to the matrix that couples them.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset, set up exactly as in `multi_galaxy/likelihood_function.py`.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

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
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit before anything else, as `multi_galaxy/modeling.py` explains.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Mask & Over Sampling__

The standard 3.0" mask, with adaptive over-sampling centred on both deflectors.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

masked_dataset = dataset.apply_mask(mask=mask)

masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=masked_dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.35, 0.25), (-0.35, -0.25)],
    )
)

aplt.subplot_imaging_dataset(dataset=masked_dataset)

"""
__Tracer__

The two co-dominant deflectors, the shear galaxy and the source, with `lp_linear` light profiles carrying no
`intensity`. Every other parameter is set to the value the simulator used.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source])

"""
__Linear Object Lists__

Linear light profiles are not evaluated and summed like standard ones. They are handed to a
`LightProfileLinearObjFuncList`, which is the interface between the profiles and the linear algebra: it computes
each profile's unit-intensity image, convolves it, and exposes the result as a matrix column.

A separate list is needed **per plane**, because each plane has its own ray-traced grid. Both deflectors are in
the image plane, so — and this is the multi-galaxy difference — they go into a *single* list with a
`light_profile_list` of length two, rather than needing one list each.
"""
lens_func = al.LightProfileLinearObjFuncList(
    grid=masked_dataset.grids.lp,
    blurring_grid=masked_dataset.grids.blurring,
    psf=masked_dataset.psf,
    light_profile_list=[lens_0.bulge, lens_1.bulge],
    regularization=None,
)

"""
`params` is the number of intensity values this list contributes to the linear system. It is 2, one per deflector,
and grows with the number of co-dominant galaxies.
"""
print("Intensity values solved for in the image plane:")
print(lens_func.params)

"""
The source plane's list, using the ray-traced grid.
"""
traced_grids = tracer.traced_grid_2d_list_from(grid=masked_dataset.grids.lp)
traced_blurring_grids = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.blurring
)

source_func = al.LightProfileLinearObjFuncList(
    grid=traced_grids[-1],
    blurring_grid=traced_blurring_grids[-1],
    psf=masked_dataset.psf,
    light_profile_list=[source.bulge],
    regularization=None,
)

"""
__Blurred Mapping Matrix (f)__

Each column is one light profile's PSF-convolved unit-intensity image. The image-plane and source-plane matrices
are stacked horizontally into the single matrix the inversion solves.

Its shape is `(total_image_pixels, total_linear_light_profiles)` — `(11304, 3)` for this dataset and mask. The
galaxy-scale equivalent has 2 columns; each extra co-dominant deflector adds one.
"""
blurred_mapping_matrix = np.hstack(
    [
        lens_func.operated_mapping_matrix_override,
        source_func.operated_mapping_matrix_override,
    ],
)

print("\nBlurred mapping matrix shape (image pixels, linear light profiles):")
print(blurred_mapping_matrix.shape)

plt.imshow(
    blurred_mapping_matrix,
    aspect=(blurred_mapping_matrix.shape[1] / blurred_mapping_matrix.shape[0]),
)
plt.show()
plt.close()

"""
__Data Vector (D)__

$D$ weights each linear light profile by how well its image maps onto the data. It has one entry per profile —
three here, in the order `lens_0`, `lens_1`, `source`, which is the order the matrices were stacked.

The equation is given in full in `imaging/features/linear_light_profiles/likelihood_function.py`.
"""
data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(masked_dataset.data),
    noise_map=np.array(masked_dataset.noise_map),
)

print("\nData vector D (lens_0, lens_1, source):")
print(data_vector)

"""
__Curvature Matrix (F)__

This is the step that carries the multi-galaxy content of this script.

$F_{ik}$ is the noise-weighted sum, over every image pixel, of profile $i$'s blurred image multiplied by profile
$k$'s. In words: **how much profile $i$'s light overlaps profile $k$'s**. It is a
`(total_linear_light_profiles, total_linear_light_profiles)` matrix — 3 x 3 here, 2 x 2 at galaxy scale.

The off-diagonal entries are where the regime shows up. At galaxy scale there is exactly one, $F_{0,1}$, coupling
the lens to the source — and their overlap is limited, because the lens sits inside the ring and the source's light
arrives as arcs around it. A multi-galaxy lens introduces an off-diagonal of a different kind: $F_{0,1}$ now
couples the two **deflectors**, whose light does not merely overlap at the edges but sits on top of each other.
"""
curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

print("\nCurvature matrix F:")
print(curvature_matrix)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
Raw $F$ entries are hard to compare because the diagonal terms differ by orders of magnitude. Normalizing by the
diagonal turns $F$ into a correlation-like matrix in which every entry is directly comparable.
"""
diagonal = np.sqrt(np.diag(curvature_matrix))
normalized_curvature_matrix = curvature_matrix / np.outer(diagonal, diagonal)

print("\nNormalized curvature matrix (lens_0, lens_1, source):")
print(np.round(normalized_curvature_matrix, 4))

"""
__The Number Worth Taking Away__

Measured on this dataset at full resolution:

    normalized F = [[1.0000  0.2955  0.1353]
                    [0.2955  1.0000  0.1130]
                    [0.1353  0.1130  1.0000]]

The deflector-deflector coupling is **0.296**. The deflector-source couplings are **0.135** and **0.113**.

So the strongest off-diagonal term in the entire linear system is the one between the two lens galaxies — more
than twice either coupling to the source. The linear solve is not mostly separating lens light from source light,
as the galaxy-scale intuition suggests. It is mostly separating the two deflectors from *each other*.

That single number explains the behaviour measured in `fit.py` in this folder, where mis-specifying `lens_0`'s
`effective_radius` moved `lens_1`'s solved intensity by 5.2% and the flux ratio by 33%. A large off-diagonal in $F$
*is* the mechanism: the solver cannot determine the two intensities independently, so an error in one profile's
shape is partly absorbed by the other's intensity.

It also tells you when to worry. The size of $F_{0,1}$ scales with how much the two galaxies' light overlaps, so a
close, blended pair like this one (0.86" separation) couples strongly, while two well-separated deflectors barely
couple at all. If you are fitting a wide pair, the flux ratio is far safer than it is here.
"""

"""
__Reconstruction__

Solving $F s = D$ gives the intensities $s$ — one per linear light profile.

The reconstruction below is the positive-negative solution, which places no constraint on the sign. Negative
values are unphysical and, in a multi-galaxy fit, usually mean one deflector's profile is over-subtracting into
its neighbour — the failure mode the large $F_{0,1}$ above makes possible.
"""
reconstruction = np.linalg.solve(curvature_matrix, data_vector)

print("\nReconstructed intensities (lens_0, lens_1, source):")
print(reconstruction)

"""
__Log Likelihood__

From here the likelihood is assembled exactly as in `multi_galaxy/likelihood_function.py`: map the reconstruction
back to an image, compute the residuals, chi-squared and noise normalization, and combine them.

Rather than repeat that arithmetic, we let `FitImaging` do it and confirm it agrees with the intensities solved
above.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

print(f"\nLog likelihood from FitImaging = {fit.log_likelihood}")

tracer_solved = fit.tracer_linear_light_profiles_to_light_profiles

print("\nIntensities from FitImaging (lens_0, lens_1):")
print(tracer_solved.galaxies[0].bulge.intensity)
print(tracer_solved.galaxies[1].bulge.intensity)

aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

The linear inversion replaces one step of the multi-galaxy likelihood, and adds one column to its matrices per
co-dominant deflector. The consequence is in the curvature matrix: the two deflectors are the most strongly
coupled pair in the system, so their intensities — and therefore their flux ratio — are solved jointly and are
sensitive to each other's shape parameters.

Where to go next:

 - `multi_galaxy/features/linear_light_profiles/fit.py` — the measured consequence of that coupling.
 - `multi_galaxy/likelihood_function.py` — the same likelihood with standard light profiles.
 - `imaging/features/linear_light_profiles/likelihood_function.py` — the full derivation of every matrix here.
 - `imaging/features/multi_gaussian_expansion` — an MGE is many linear profiles per galaxy, so $F$ becomes
   much larger and this coupling structure becomes richer.
"""

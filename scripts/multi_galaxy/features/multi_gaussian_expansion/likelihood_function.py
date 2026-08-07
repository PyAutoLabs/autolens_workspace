"""
__Log Likelihood Function: Multi Gaussian Expansion (Multi Galaxy)__

This script walks through the log likelihood function of a multi-galaxy lens fitted with **an MGE per co-dominant
deflector**, focusing on the one structure that has no galaxy-scale equivalent: the block structure of the
curvature matrix when two nearby galaxies each carry their own basis.

__Prerequisites__

Read these first; they are not repeated here:

 - `multi_galaxy/features/linear_light_profiles/likelihood_function.py` — the same walkthrough with **one** linear
   profile per galaxy. It derives the mapping matrix, the data vector `D` and the curvature matrix `F`, and
   measures the two deflectors' coupling at **0.296**. This script is that script with 20 profiles per galaxy
   instead of 1.
 - `imaging/features/multi_gaussian_expansion/likelihood_function.py` — the galaxy-scale MGE derivation.

__Contents__

- **What Changes:** Why 20 profiles per galaxy is a different situation, not just a bigger one.
- **Dataset, Mask & Over Sampling:** Set up.
- **Tracer:** Two MGE bases plus mass, shear and source.
- **Linear Object Lists:** Both bases go in one image-plane list.
- **Blurred Mapping Matrix:** 41 columns, not 3.
- **Curvature Matrix (F):** The block structure, and the number that matters.
- **Condition Number:** Why the positive-only solver is load-bearing here.
- **Reconstruction:** Solving for 41 intensities.
- **Wrap Up:** Where to go next.

__What Changes__

With one linear `Sersic` per galaxy, the linear system has 3 unknowns and one off-diagonal that couples the two
deflectors. With a 20-Gaussian MGE per galaxy it has 41, arranged in blocks:

    columns  0-19 : lens_0's basis
    columns 20-39 : lens_1's basis
    column     40 : the source

`F` is then 41 x 41 with three distinct kinds of off-diagonal: *within* a galaxy's basis, *between* the two
galaxies' bases, and to the source. They behave very differently, and only one of them is a problem.
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

The `mge` dataset — the co-dominant pair with twisted two-component light.
"""
dataset_name = "mge"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/multi_gaussian_expansion/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

mask_radius = 3.0
main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

masked_dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=masked_dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=main_lens_centres,
    )
)

aplt.subplot_imaging_dataset(dataset=masked_dataset)

"""
__Tracer__

Two MGE bases — one per deflector — plus the true mass profiles, shear and source.
"""
total_gaussians = 20

log10_sigma_list = np.linspace(
    np.log10(dataset.pixel_scales[0] / 10.0), np.log10(mask_radius), total_gaussians
)


def mge_basis_from(centre):
    return al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Gaussian(
                centre=centre, ell_comps=(0.0, 0.0), sigma=10**log10_sigma
            )
            for log10_sigma in log10_sigma_list
        ]
    )


lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=mge_basis_from(centre=main_lens_centres[0]),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=mge_basis_from(centre=main_lens_centres[1]),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
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

A `LightProfileLinearObjFuncList` is needed per **plane**, not per galaxy. Both deflectors are in the image plane,
so both bases go into one list — its `light_profile_list` is the two galaxies' Gaussians concatenated.

This is worth seeing explicitly, because it is what makes the coupling below possible: the solver is handed 40
image-plane profiles with no record of which galaxy each belongs to. The block structure exists in *our* reading
of the matrix, not in the linear algebra.
"""
lens_func = al.LightProfileLinearObjFuncList(
    grid=masked_dataset.grids.lp,
    blurring_grid=masked_dataset.grids.blurring,
    psf=masked_dataset.psf,
    light_profile_list=list(lens_0.bulge.profile_list)
    + list(lens_1.bulge.profile_list),
    regularization=None,
)

print(f"Image-plane intensity values solved for: {lens_func.params}")

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
__Blurred Mapping Matrix__

Each column is one profile's PSF-convolved unit-intensity image; 41 columns for this model.
"""
blurred_mapping_matrix = np.hstack(
    [
        lens_func.operated_mapping_matrix_override,
        source_func.operated_mapping_matrix_override,
    ],
)

print(f"Blurred mapping matrix shape: {blurred_mapping_matrix.shape}")

"""
__Data Vector (D)__

One entry per profile, in the stacking order: `lens_0`'s 20 Gaussians, `lens_1`'s 20, then the source.
"""
data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(masked_dataset.data),
    noise_map=np.array(masked_dataset.noise_map),
)

"""
__Curvature Matrix (F)__

`F_ik` is how much profile `i`'s light overlaps profile `k`'s, noise-weighted. Normalizing by the diagonal makes
every entry directly comparable.
"""
curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

diagonal = np.sqrt(np.diag(curvature_matrix))
normalized = curvature_matrix / np.outer(diagonal, diagonal)

plt.imshow(np.abs(normalized))
plt.colorbar()
plt.title("|normalized F| — block structure")
plt.show()
plt.close()

"""
The image above is the point of this script. Three regions are visible:

 - Two bright squares on the diagonal — each galaxy's basis correlating with itself.
 - A block off the diagonal — the two galaxies correlating with **each other**.
 - A thin final row and column — everything correlating with the source.

Let us put numbers on them.
"""
block_0 = slice(0, total_gaussians)
block_1 = slice(total_gaussians, 2 * total_gaussians)

cross = np.abs(normalized[block_0, block_1])
within = np.abs(normalized[block_0, block_0][np.triu_indices(total_gaussians, 1)])
to_source = np.abs(normalized[0 : 2 * total_gaussians, 2 * total_gaussians])

print(f"\ndeflector-deflector |C| : mean={cross.mean():.4f}  max={cross.max():.4f}")
print(f"within-lens_0      |C| : mean={within.mean():.4f}  max={within.max():.4f}")
print(
    f"deflector-source   |C| : mean={to_source.mean():.4f}  max={to_source.max():.4f}"
)

"""
__The Number That Matters__

Measured on this dataset:

    deflector-deflector |C| : mean 0.1193, max 0.9877
    within-lens_0       |C| : mean 0.4592, max 1.0000
    deflector-source    |C| : mean 0.0977, max 0.3841

These are stable to four decimal places across re-simulations, because `F` depends on the model geometry and the
noise map rather than on the particular noise draw. (The log likelihoods quoted in `fit.py` are not — they move a
percent or two.)

Read them in order:

**within-lens_0, max 1.0000.** Adjacent Gaussians in one basis are perfectly correlated. This is expected and
harmless: it is what a basis *is*. Nobody interprets an individual Gaussian's intensity — only their sum, which is
well constrained. The regularization-free MGE relies on the positive-only solver to keep this from turning into
ringing.

**deflector-source, max 0.3841.** Modest, and the quantity a galaxy-scale intuition expects to be the hard one.
It is not.

**deflector-deflector, max 0.9877.** Some Gaussian in `lens_0`'s basis is 99% degenerate with some Gaussian in
`lens_1`'s. Unlike the within-basis case, this one *is* interpreted: the sum over `lens_0`'s basis is that
galaxy's luminosity, and the ratio of the two galaxies' luminosities is frequently the measurement. A 0.99
correlation between columns belonging to different galaxies means the data cannot fully separate whose light is
whose.

Compare the single-Sersic case in `multi_galaxy/features/linear_light_profiles/likelihood_function.py`: one
profile per galaxy, coupling 0.296. Giving each galaxy 20 profiles to describe itself with also gave the pair 400
ways to trade light between them.

__Condition Number__

The practical consequence.
"""
print(f"\ncondition number of F: {np.linalg.cond(curvature_matrix):.3e}")

"""
Measured at roughly **1e24**. A matrix this ill-conditioned cannot be inverted naively — the solution is
enormously sensitive to the data, and a positive-negative solver will find a "better" fit consisting of large
positive Gaussians in one galaxy cancelled by large negative ones in its neighbour.

That is why PyAutoLens uses a **positive-only** solver, and why it matters more in this regime than at galaxy
scale. At galaxy scale, positive-negative ringing is unphysical but confined to one galaxy's basis. Here it can
move flux *between* galaxies, corrupting the one measurement the multi-galaxy regime exists to make.

__Reconstruction__

Solving the system gives 41 intensities.
"""
reconstruction = np.linalg.solve(curvature_matrix, data_vector)

print(f"\nlens_0 basis intensities (first 5): {reconstruction[block_0][:5]}")
print(f"lens_1 basis intensities (first 5): {reconstruction[block_1][:5]}")
print(f"source intensity: {reconstruction[-1]}")

"""
Note this uses `np.linalg.solve`, the positive-negative solver, precisely to show what PyAutoLens avoids.

Measured on this dataset: **21 of the 41 intensities come back negative.** Half the basis is describing light that
is being subtracted rather than emitted — the "ringing" the positive-only solver exists to prevent, and here it is
happening across two galaxies that each have a real luminosity someone will want to quote. `FitImaging` below uses
the positive-only solver, and its intensities are the ones to trust.
"""
n_negative = int((reconstruction < 0.0).sum())
print(
    f"\nnegative intensities from the naive solve: {n_negative} of {len(reconstruction)}"
)

fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

print(f"\nLog likelihood from FitImaging: {fit.log_likelihood}")

aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

The MGE changes the likelihood's linear step from a 3-unknown system to a 41-unknown one with block structure. The
blocks within each galaxy are near-singular and harmless; the block *between* the galaxies reaches 0.9877 and is
not, because it acts on a quantity people interpret.

Where to go next:

 - `multi_galaxy/features/multi_gaussian_expansion/fit.py` — the measured consequences for the fit.
 - `multi_galaxy/features/linear_light_profiles/likelihood_function.py` — the 3-unknown version, coupling 0.296.
 - `multi_galaxy/features/multi_gaussian_expansion/source_science.py` — integrating a basis to a luminosity,
   which is the quantity all of this affects.
"""

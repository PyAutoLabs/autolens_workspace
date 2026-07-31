"""
Modeling Features: Pixelization (Multi Galaxy)
==============================================

This script fits a multi-galaxy strong lens with a **pixelized source reconstruction**: the source is
reconstructed on a mesh of pixels whose fluxes are solved by linear algebra, rather than assumed to follow an
analytic profile.

__Contents__

- **Advantages:** Why a pixelized source, at any scale.
- **The Multi-Galaxy Cost:** What it does to the mass split, measured.
- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Positions:** The constraint that compensates, and why it is not optional here.
- **Model Composition:** One `lens_i` per deflector, plus a pixelized source.
- **Search & Analysis:** Configure the fit.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Advantages__

Real source galaxies are irregular: clumpy star formation, asymmetric disks, multiple components. An analytic
profile cannot represent that, and the residuals it leaves are absorbed by the mass model, biasing it. A pixelized
source has one free parameter per source pixel (solved linearly, so none of them enter the non-linear search) and
can reconstruct essentially any morphology.

This is why `multi_galaxy/slam.py` moves to a pixelized source for its final stages, and why the galaxy-scale
guidance is to adopt one as soon as your mass model is good enough to support it.

__The Multi-Galaxy Cost__

There is a cost specific to this regime, and it is large enough to change how you use the feature.

`multi_galaxy/modeling.py` establishes the regime's central degeneracy: the data constrains the **total**
deflection well and the **split** between the deflectors much less well. A pixelized source weakens the
constraint on that split further, because a free-form mesh can rearrange itself to keep reproducing the arcs when
the split is wrong — where a parametric source, having a fixed functional form, simply fails.

`fit.py` in this folder measures the size of it. Holding the total Einstein radius at its true 1.8" and varying
only the split between the two deflectors:

     split    r_0   r_1 |   parametric    pixelized
     0.556   1.00  0.80 |     12,379.2     11,466.6      <- true split
     0.600   1.08  0.72 |     -7,955.3      5,907.2
     0.700   1.26  0.54 |   -110,272.7    -32,605.9
     0.800   1.44  0.36 |   -182,813.4    -76,774.0

As a penalty relative to each model's own best:

     split    parametric    pixelized    ratio
     0.600       20,334        5,560      3.7x
     0.700      122,652       44,073      2.8x
     0.800      195,193       88,241      2.2x

A 26% error in `lens_0`'s Einstein radius (the 0.600 row) costs a parametric fit 20,334 in log likelihood and a
pixelized fit only 5,560. The mesh absorbs roughly three quarters of the evidence against a wrong mass split.

This is not an argument against pixelized sources — a Sersic fitted to a genuinely irregular source biases the
mass model in its own way. It is an argument for the two mitigations below.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is an MGE, its mass an `Isothermal` with its centre fixed.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source is reconstructed on a `RectangularAdaptDensity` mesh with `Constant` regularization.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/modeling.py` and `imaging/features/pixelization/modeling.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset — the same co-dominant pair fitted by `multi_galaxy/modeling.py`, so the two
fits are directly comparable.
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

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains.

This matters more for a pixelized source than a parametric one. Unmodelled flux inside the mask is exactly what a
free-form mesh will reconstruct as source structure — a contaminant left in the data does not merely add
residuals, it can appear in your source reconstruction.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Positions__

Multiple-image positions constrain the mass model **directly** — they depend on where the deflection field sends
rays, not on how well the source is reconstructed. That makes them the natural compensation for the effect
measured above: the constraint a free-form mesh weakens is precisely the one positions restore.

At galaxy scale the positions likelihood is presented as a guard against unphysical source reconstructions. Here
it is doing a second job, and skipping it costs more.

The positions were solved by the simulator and saved alongside the dataset.
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "positions.json")
)

print(f"Multiple image positions used to constrain the mass model:\n{positions}")

"""
__Model Composition__

The standard multi-galaxy composition — one `lens_i` per deflector in a loop, the shear in its own
`shear_galaxy` — with the source's light replaced by a `Pixelization`.

A `Pixelization` has two parts:

 - a **mesh**, which decides where the source pixels are. `RectangularAdaptDensity` places more pixels where the
   source is brighter, so resolution follows the signal.
 - a **regularization**, which penalizes unsmooth reconstructions. `Constant` applies one smoothing strength
   everywhere, and its coefficient is the single sampled parameter the source contributes.

Two constraints on that pairing are worth knowing before you vary it, because both fail loudly rather than
silently:

 - **`AdaptSplit` will not pair with a rectangular mesh.** It regularizes using a cross of four points around each
   pixel centre and needs the mesh to supply split-cross mappings, which the rectangular meshes do not. Pairing
   them raises a `PixelizationException` naming the incompatibility. The Delaunay meshes do support it.
 - **The `Adapt` schemes need adapt-images**, an estimate of the source's surface brightness used to decide where
   to smooth harder. A standalone script has no earlier fit to derive one from, so `Constant` is the right choice
   here. `multi_galaxy/slam.py` uses `Adapt` because by that point `source_pix[1]` has produced one.

Neither contributes non-linear parameters for the pixel fluxes; those are solved linearly. The regularization
coefficient(s) are sampled.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

# External Shear:

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source (pixelized):

pixelization = af.Model(
    al.Pixelization,
    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=(28, 28)),
    regularization=af.Model(al.reg.Constant),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source)
)

"""
The `info` attribute shows the model in a readable format.

Note the source contributes only the regularization parameters — its ~784 pixel fluxes are solved, not sampled.
"""
print(model.info)

"""
__Search__

The lens model is fitted using the nested sampling algorithm Nautilus, with 200 live points as in
`multi_galaxy/modeling.py`.

The mesh shape is fixed rather than fitted. `imaging/features/pixelization/modeling.py` explains why: the number
of source pixels changes the model's dimensionality, so it cannot be a sampled parameter.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="pixelization",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

The `positions_likelihood` is passed here. Given the measurement above, treat it as part of the model rather than
an optional safeguard when fitting a multi-galaxy lens with a pixelized source.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[
        al.PositionsLH(positions=positions, threshold=0.3)
    ],
)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

Beyond the usual checks (`multi_galaxy/modeling.py`, `guides/results`), two are specific to a pixelized source at
multi-galaxy scale:

 - **Inspect the mass split and its errors.** If the two Einstein radii are individually poorly constrained while
   their sum is tight, that is the degeneracy this feature widens, showing up exactly where expected.
 - **Look at the reconstructed source for structure that tracks the deflectors.** A source with features aligned
   to the lens galaxies' positions rather than to its own morphology is the visible signature of the mesh
   absorbing a mass-model error.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/fit.py` — the measurement quoted above, run directly.
 - `multi_galaxy/slam.py` — the pipeline whose stage ordering (parametric source first, pixelized second, mass
   held while the source is freed) exists to manage this.
 - `imaging/features/pixelization` — the galaxy-scale walkthrough, with the full mesh and regularization API,
   adaptive meshes, Delaunay meshes and CPU-fast modeling. Those variants are not yet written for this package;
   they apply with the single lens galaxy swapped for this package's `lens_0`, `lens_1`, ... loop.
"""

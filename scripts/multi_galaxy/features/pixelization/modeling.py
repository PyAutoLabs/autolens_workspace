"""
Modeling Features: Pixelization (Multi Galaxy)
==============================================

This script fits a multi-galaxy strong lens with a **pixelized source reconstruction**: the source is
reconstructed on a mesh of pixels whose fluxes are solved by linear algebra, rather than assumed to follow an
analytic profile.

__Contents__

- **Advantages:** Why a pixelized source is used.
- **Disadvantages:** The costs of a pixelized source.
- **Positive Only Solver:** Why source pixel fluxes are constrained to be positive.
- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Positions:** The constraint that compensates, and why it is not optional here.
- **Model Composition:** One `lens_i` per deflector, plus a pixelized source.
- **Search & Analysis:** Configure the fit.
- **Run Time:** Profiling the expected run time of the model-fit.
- **VRAM:** GPU memory used by a pixelized model-fit.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Advantages__

Real source galaxies are irregular: clumpy star formation, asymmetric disks, multiple components. An analytic
profile cannot represent that, and the residuals it leaves are absorbed by the mass model, biasing it. A pixelized
source has one free parameter per source pixel (solved linearly, so none of them enter the non-linear search) and
can reconstruct essentially any morphology.

This is why `multi_galaxy/slam.py` moves to a pixelized source for its final stages, and why the galaxy-scale
guidance is to adopt one as soon as your mass model is good enough to support it.

__Disadvantages__

A pixelized source is slower to evaluate than an analytic light profile, because every likelihood evaluation
performs a linear inversion whose size scales with the number of source pixels. It also uses considerably more
GPU memory (see __VRAM__ below).

The reconstruction is regularized, and the regularization coefficient is a free parameter which must be fitted.
Too little regularization over-fits the noise, too much over-smooths the source.

__Positive Only Solver__

Many codes which use linear algebra rely on a solver which allows positive and negative values in the solution
(e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means negative surface brightness values can be used to represent a galaxy's light,
which is unphysical. For a pixelization this often produces negative source pixels which over-fit the data.

All pixelized source reconstructions use a positive-only solver, so every source pixel reconstructs positive flux
only. This ensures the source reconstruction is physical.

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

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
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
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Positions__

The multiple images of the source, used to constrain the mass model via a `PositionsLH` likelihood penalty. This
is described in full in `imaging/features/pixelization/modeling.py`.

The positions were solved by the simulator and saved alongside the dataset.
"""
positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))

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
        sigma_min=dataset.pixel_scales[0] / 10.0,
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
__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

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

The `positions_likelihood_list` is passed to the analysis, which applies the likelihood penalty to every lens
mass model that is fitted.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`.

A pixelized source is slower per likelihood evaluation than an analytic source, because each evaluation performs a
linear inversion over the source mesh. The cost scales with the number of source pixels, so the `mesh_shape` set
above is the main run-time dial in this script.

__VRAM__

The `multi_galaxy/modeling.py` example explains how VRAM is used during GPU-based fitting and how to print the
estimated VRAM required by a model.

Pixelizations use considerably more VRAM than light-profile-only models, with the amount depending on the size of
the dataset and the number of source pixels in the mesh. Reducing the search's batch size lowers VRAM use at the
cost of run time.

The method below prints the VRAM estimate for this analysis and model. It takes 20-30 seconds, so comment it out
once you are familiar with your GPU's limits.
"""
# analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, described in `multi_galaxy/modeling.py` and in full in
`autolens_workspace/*/guides/results`.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/fit.py` — the same pixelization without a search, where the inversion can
   be inspected directly.
 - `multi_galaxy/features/pixelization/adaptive.py` — the adaptive mesh and regularization schemes.
 - `multi_galaxy/features/pixelization/delaunay.py` — the Delaunay meshes.
 - `multi_galaxy/slam.py` — the SLaM pipeline, whose later stages use a pixelized source.
 - `imaging/features/pixelization` — the galaxy-scale walkthrough, with the full mesh and regularization API.
"""

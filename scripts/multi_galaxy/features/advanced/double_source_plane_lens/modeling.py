"""
Modeling Features: DSPL (Multi Galaxy)
======================================

A double source-plane lens (DSPL) is a strong lens with two source galaxies at different redshifts behind the
same deflector. They appear as two distinct Einstein rings in the image plane, and can constrain cosmological
parameters in a way single-ring lenses cannot.

This script fits one whose deflector is a pair of co-dominant galaxies. The mass of both deflectors **and** of
the first source must be modelled simultaneously, and the light of both sources must be modelled simultaneously.

__Practical Use: Read This First__

This script is a tutorial. It produces a working fit by "cheating" — key priors are initialised at the true
simulator values, narrowed by small Gaussians. On real data that is impossible, and a single Nautilus search on a
model this size would almost certainly converge to a local maximum.

The scripts you would actually use on real data are `chaining.py` (two chained searches) and `slam.py` (the full
pipeline), both in this folder.

__Contents__

- **Why Two Source Planes Help Here:** The regime-specific point, and what it is not.
- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Model Composition:** One `lens_i` per deflector, plus two source planes.
- **Model Cookbook:** Where the full model-composition API is documented.
- **Cheating:** The priors that make a single search tractable, and why they are not honest.
- **Cosmology:** What the second source plane is usually wanted for.
- **Search & Analysis:** Configure the fit.
- **Run Time:** Profiling the expected run time of the model-fit.
- **VRAM:** GPU memory used by the model-fit.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Why Two Source Planes Help Here__

The multi-galaxy regime's standing problem is that the data constrains the *total* deflection of the pair well
and the *split* between the two deflectors much less well (`multi_galaxy/modeling.py`). A second source plane
helps with that split — but not for the reason it is usually assumed to.

It is **not** the extra redshift. Both sources sit behind the same pair of deflectors, so the deflection field
the second source sees is the first source's field scaled by a geometric factor. A scaling carries no
information about how the mass is divided between the two galaxies, and the degeneracy scales straight through
it.

What helps is that the second source sits at a **different sky position**, so its ring's images land somewhere
else in the image plane. The mass split is a statement about the *spatial structure* of the deflection field,
and a second set of images measures that field where the first source's images do not reach. Two rings at the
same sky position would add almost nothing; two rings in different places add a great deal.

This is why `simulator.py` offsets `source_1` from `source_0` rather than placing it directly behind.

The extra redshift is still worth having — it is what constrains cosmology, below. It just is not what constrains
the mass split.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' DSPL where:

 - Each co-dominant deflector's light is an MGE, its mass an `Isothermal` with its centre fixed.
 - The system has a single overall `ExternalShear` at the system centre.
 - `source_0` at z=1.0 has a linear light profile **and** an `IsothermalSph` mass, since it deflects `source_1`.
 - `source_1` at z=2.0 has a linear light profile only.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/modeling.py` for the multi-galaxy composition and
`imaging/features/advanced/double_source_plane_lens/modeling.py` for the single-deflector DSPL and the
multi-plane ray-tracing API.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `dspl` multi-galaxy dataset — the `simple` co-dominant pair with a second source plane behind the first.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "dspl"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/double_source_plane_lens/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre.

The mask has to contain both rings, not just the brighter one. A mask sized for the first ring alone would cut
away exactly the images that carry the extra information described above.
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
__Model Composition__

The standard multi-galaxy composition from `multi_galaxy/modeling.py` — one `lens_i` per deflector in a loop, the
shear in its own `shear_galaxy` — with two source galaxies instead of one.

`source_0` carries mass as well as light. That is not optional: its mass deflects `source_1`, and omitting it
would leave the second ring's position to be explained by the deflectors, biasing exactly the split this dataset
is meant to constrain.

Redshifts are what make the tracing multi-plane. PyAutoLens orders the galaxies by redshift internally, so
setting `redshift=1.0` and `redshift=2.0` is the whole of the configuration.
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

# Source 0 (light + mass, at z=1.0):

source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=af.Model(al.lp_linear.SersicCore),
    mass=af.Model(al.mp.IsothermalSph),
)

# Source 1 (light only, at z=2.0):

source_1 = af.Model(
    al.Galaxy,
    redshift=2.0,
    bulge=af.Model(al.lp_linear.ExponentialCoreSph),
)

"""
__Cheating__

The priors below are narrowed around the simulator's true values so that one search can find the solution.

This is a tutorial device, not a method. A multi-galaxy DSPL has both of the hard initialisation problems at
once: the mass split between two co-dominant deflectors, and a second source plane whose position depends on the
first source's mass. Neither is solvable by one search from broad priors.

The deflector mass centres are kept fixed to the loaded values, which is the package convention and not a cheat —
they come from the observed light. What is cheated is `source_0`'s mass and both sources' positions.
"""
source_0.bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
source_0.bulge.centre_1 = af.GaussianPrior(mean=0.03, sigma=0.1)
source_0.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
source_0.mass.centre_1 = af.GaussianPrior(mean=0.03, sigma=0.1)
source_0.mass.einstein_radius = af.GaussianPrior(mean=0.15, sigma=0.05)

source_1.bulge.centre_0 = af.GaussianPrior(mean=-0.25, sigma=0.1)
source_1.bulge.centre_1 = af.GaussianPrior(mean=0.28, sigma=0.1)

"""
__Cosmology__

This is what the second redshift is actually for.

The ratio of angular diameter distances between the deflector, `source_0` and `source_1` depends on the
cosmological model, and a DSPL measures that ratio directly. It is the reason DSPLs are sought out, and it is
independent of the mass-split argument at the top of this script — one comes from the extra redshift, the other
from the extra sky position.

This script keeps a fixed Planck18 cosmology to hold the parameter count down. To free `Om0`, uncomment the two
lines below and pass `cosmology` into the `af.Collection`. A real cosmological constraint needs the chained
workflow in `chaining.py` and far more than one system.
"""
# cosmology = af.Model(al.cosmo.FlatLambdaCDM)
# cosmology.Om0 = af.GaussianPrior(mean=0.3, sigma=0.1)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(
        **lens_dict,
        shear_galaxy=shear_galaxy,
        source_0=source_0,
        source_1=source_1,
    ),
    # cosmology=cosmology,
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Search__

The lens model is fitted using the nested sampling algorithm Nautilus.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features" / "advanced",
    name="double_source_plane_lens",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`.

Multi-plane ray tracing is more expensive than single-plane: the grid is traced through each plane in turn
rather than once. Combined with two deflectors and two sources, this is the slowest model in the package — which
is another reason `chaining.py` is the script to use on real data.

__VRAM__

The `multi_galaxy/modeling.py` example explains how VRAM is used during GPU-based fitting.

A multi-galaxy DSPL is the most VRAM-hungry model here: multi-plane tracing, two deflectors and batched search
samples all multiply. The method below prints the estimate; it takes 20-30 seconds.
"""
# analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result carries both source planes. Beyond the usual multi-galaxy checks in `multi_galaxy/modeling.py`, two
are specific to a DSPL:

 - **`source_0`'s mass.** It is constrained almost entirely by the second ring's position. A posterior that has
   not moved from its prior means the second ring is not informing the fit.
 - **Both rings' residuals.** A model that fits the bright first ring and leaves the fainter second one poorly
   fitted has bought none of the extra constraint this dataset exists to provide.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/double_source_plane_lens/chaining.py` — the same lens fitted properly, in two
   searches, without cheating.
 - `multi_galaxy/features/advanced/double_source_plane_lens/fit.py` — the multi-plane tracer inspected directly.
 - `multi_galaxy/features/advanced/double_source_plane_lens/slam.py` — the full pipeline.
 - `imaging/features/advanced/double_source_plane_lens` — the single-deflector DSPL walkthrough.
"""

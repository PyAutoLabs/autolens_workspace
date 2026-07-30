"""
Modeling Features: Group DSPL
=============================

A group-scale double source-plane lens (DSPL), where two source galaxies at different redshifts are lensed by multiple
main lens galaxies at the lens-plane redshift.

This script illustrates the PyAutoLens API for modeling such a system in a single non-linear search. The lens
plane is composed via the group `lens_dict` convention (loaded from a JSON file of main lens galaxy centres),
and the source plane has two galaxies — `source_0` at z=1.0 with light + mass, and `source_1` at z=2.0 with
light only.

__Practical Use: Read This First__

This script is a tutorial. It produces a working fit by "cheating" — every prior is initialised at the true
simulator value, narrowed by a small Gaussian. On real data this is impossible, and a single Nautilus search on
this many-parameter group DSPL model would almost certainly converge to a local maximum.

The script you will actually use to fit a group-scale DSPL on real data is
`autolens_workspace/scripts/group/features/advanced/double_source_plane_lens/chaining.py`, which runs two chained
non-linear searches: the first initialises the main lens galaxies' mass and `source_0` using a smaller mask
that excludes `source_1`, the second introduces `source_1` and frees `source_0`'s mass.

For production-quality modeling, see `slam.py` in the same directory.

Read this script to understand the model composition API, then jump to `chaining.py`.

__Contents__

- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask, Over Sampling:** Standard set up.
- **Main Lens Centres:** Load the two main lens galaxy centres from JSON.
- **Model Composition:** Build `lens_dict` via the group `lens_dict` convention, plus `source_0` and `source_1`.
- **Cheating:** Override priors with narrow Gaussians around the true simulator values.
- **Cosmology:** Fixed at Planck18 by default — a commented-out snippet shows how to make `Om0` free.
- **Search, Analysis, Run-Time, Result, Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the group DSPL `Imaging` dataset.
"""
dataset_name = "double_source_plane_lens"
dataset_path = Path("dataset") / "group" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/double_source_plane_lens/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Centres__

The two main lens galaxy centres are loaded from the JSON file saved by the simulator. They are used to define
the over-sampling pattern and as centres for each lens galaxy's MGE bulge in the model composition below.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

A 4.0" circular mask that encloses both Einstein rings around the centroid of the two main lens galaxies.
"""
mask_radius = 4.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

Adaptive over-sampling at each main lens galaxy centre.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__

We compose a lens model where:

 - Each main lens galaxy has an MGE bulge (20 Gaussians) and an `Isothermal` mass profile. Centres start fixed at
   the simulator values for over-sampling and are then given narrow Gaussian priors in the "Cheating" section.

 - `source_0` (z=1.0) has an MGE bulge and an `IsothermalSph` mass, since it acts as a secondary deflector for
   `source_1`.

 - `source_1` (z=2.0) has an MGE bulge only.

Total non-linear parameter count: ~16 per main lens galaxy mass + bulge centre (~5 each), plus ~6 for source_0
and ~3 for source_1 — comparable to the imaging DSPL example.

__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Main Lens Galaxies via lens_dict:

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

# Source 0:

bulge_source_0 = af.Model(al.lp_linear.ExponentialCoreSph)
mass_source_0 = af.Model(al.mp.IsothermalSph)

source_0 = af.Model(
    al.Galaxy, redshift=1.0, bulge=bulge_source_0, mass=mass_source_0, centre=(0.0, 0.0)
)

# Source 1:

bulge_source_1 = af.Model(al.lp_linear.ExponentialCoreSph)

source_1 = af.Model(al.Galaxy, redshift=2.0, bulge=bulge_source_1, centre=(0.0, 0.0))

"""
__Cheating__

Initializing a group DSPL model is even harder than the single-lens case, due to the larger
parameter space. To make the single-search example tractable, we override key priors with narrow Gaussians
centred at the simulator values. On real data this is impossible.

The main lens galaxy centres are kept fixed at their loaded positions (this is the standard group convention —
centres are determined from independent photometry). We "cheat" by narrowing `source_0`'s mass priors around
its true position.
"""
source_0.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.2)
source_0.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.2)
source_0.mass.einstein_radius = af.GaussianPrior(mean=0.25, sigma=0.1)

"""
__Cosmology__

DSPLs constrain cosmology via the angular diameter distance ratios between the lens, source_0,
and source_1. This script uses a fixed Planck18 cosmology to keep the parameter count manageable for the
single-search "cheating" workflow. A realistic cosmological constraint requires the chained-search workflow in
`chaining.py` and significantly more data than one simulated system.

To make `Om0` (Omega_m) a free parameter, uncomment the three lines below.
"""
# cosmology = af.Model(al.cosmo.FlatLambdaCDM)
# cosmology.Om0 = af.GaussianPrior(mean=0.3, sigma=0.1)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source_0=source_0, source_1=source_1),
    # cosmology=cosmology,
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

We use the nested sampling algorithm Nautilus.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="double_source_plane_lens",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=2000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisImaging` object.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__VRAM__

Group-scale DSPLs are VRAM-intensive on GPU: multi-plane ray-tracing, multiple main lens
galaxies and batched non-linear search samples all multiply VRAM use.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

Run times for this model are quite long, because (a) multi-plane ray-tracing is expensive, (b) the model has
many more parameters than a single-plane lens, and (c) the "cheating" workflow is only viable when the priors
are narrow enough to keep the search away from local maxima.

For real data, use `chaining.py` instead — the chained-search approach is significantly more efficient AND more
robust.

__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__
"""
print(result.info)
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

Group-scale DSPLs can be fit in PyAutoLens, but this script "cheats" by initialising
priors at their true values. For real data, use `chaining.py` (two chained searches) or `slam.py` (the full
SLaM pipeline), and consult the imaging DSPL example for an introduction to the multi-plane
ray-tracing API.
"""

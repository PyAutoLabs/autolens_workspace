"""
Modeling Features: Pixelization (Group)
=======================================

This script fits a group-scale strong lens where the source galaxy is reconstructed using a pixelized mesh
(Delaunay triangulation) with adaptive regularization, rather than parametric light profiles.

For group-scale lenses, pixelized source reconstructions are especially valuable because the lensed source
morphology is often complex, with multiple extended arcs and counter-images produced by the combined mass of
several galaxies. A parametric source model (e.g. Sersic or MGE) may struggle to capture these features,
whereas a pixelized mesh can reconstruct arbitrarily complex source-plane emission.

The main lens galaxies and extra galaxies are modeled with MGE light profiles (via ``al.model_util.mge_model_from``)
and Isothermal mass profiles, following the standard group modeling pattern. The source galaxy uses a
Delaunay pixelization with AdaptSplit regularization.

__Contents__

**Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
**Galaxy Centres:** Load centres for main lens and extra galaxies from JSON files.
**Model:** Compose the group lens model with a pixelized source.
**Over Sampling:** Adaptive over-sampling at all galaxy centres.
**Search:** Configure the non-linear search.
**Analysis:** Create the Analysis object with pixelization settings.
**Result:** Overview of the results of the model-fit.

__Example__

This script fits an `Imaging` dataset of a 'group-scale' strong lens where:

 - There is a main lens galaxy whose light is an MGE and total mass is `Isothermal` with `ExternalShear`.
 - There are two extra lens galaxies with MGE light and `IsothermalSph` mass, centres fixed.
 - The source galaxy's light is reconstructed using a `Delaunay` mesh with `AdaptSplit` regularization.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset `simple`, which is the dataset we will use to perform lens modeling.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "group", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
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
__Mask__

We use a 7.5 arcsecond circular mask, which is larger than galaxy-scale lenses because group-scale systems
have lensed emission spread over a wider area.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

Load the centres of the main lens galaxies and extra galaxies from JSON files.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Model__

We compose a group lens model where:

 - Each main lens galaxy has MGE light and Isothermal mass. Only lens_0 carries ExternalShear.
 - Each extra galaxy has MGE light and IsothermalSph mass with fixed centres and bounded Einstein radii.
 - The source galaxy uses a Delaunay pixelization with AdaptSplit regularization.

The pixelized source captures complex lensed morphologies far better than parametric profiles, which is
especially important for group lenses where the extended mass distribution creates intricate arc structures.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_dict[f"lens_{i}"] = lens

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    # Extra Galaxy Light

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=centre
    )

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source:

pixelization = af.Model(
    al.Pixelization,
    mesh=af.Model(al.mesh.Delaunay),
    regularization=af.Model(al.reg.AdaptSplit),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    extra_galaxies=extra_galaxies,
)

"""
The `info` attribute shows the model in a readable format, confirming the pixelized source.
"""
print(model.info)

"""
__Over Sampling__

Over sampling at each galaxy centre (both main lens galaxies and extra galaxies) ensures the lens light
profiles are accurately evaluated across the full field of the group.

For the pixelization, a separate uniform over-sampling is applied via `over_sample_size_pixelization`.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Search__

We use Nautilus with `n_live=100` and `n_batch=20`, suitable for a group-scale pixelization model.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="pixelization",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=20,
    iterations_per_quick_update=10000,
)

"""
__Analysis__

The `AnalysisImaging` object defines how the model is fitted to the data.

For pixelized source fits, we enable mixed precision to speed up GPU run times on consumer hardware.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    settings=al.Settings(use_mixed_precision=True),
)

"""
__Run Times__

Group-scale pixelization fits are more computationally expensive than galaxy-scale fits because the larger
7.5" mask contains many more image pixels, all of which must be mapped to source pixels. GPU acceleration
is strongly recommended.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The result contains the best-fit pixelized source reconstruction alongside the group lens model.
"""
print(result.info)

print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script demonstrated how to model a group-scale strong lens with a pixelized source reconstruction.

The pixelized source is particularly powerful for group lenses because:

 - The combined mass of multiple galaxies creates complex arc structures that parametric profiles cannot capture.
 - The Delaunay mesh adapts to the source morphology, placing more triangles where the source is brightest.
 - AdaptSplit regularization allows different smoothing in bright vs. faint source regions.

For automated modeling of large samples, the SLaM pipeline (see `group/features/pixelization/slam.py`) provides
a robust framework that chains parametric and pixelized source fits.
"""

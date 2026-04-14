"""
Modeling Features: Multi Gaussian Expansion (Group)
===================================================

A Multi Gaussian Expansion (MGE) decomposes the light of each galaxy into ~10-30+ Gaussians, where the `intensity`
of every Gaussian is solved for via linear algebra using a process called an "inversion" (see the
`linear_light_profiles` feature for a full description of this).

This script performs lens modeling of a group-scale strong lens using MGE light profiles for all galaxies:
the main lens galaxies, the extra galaxies, and the source galaxy. MGE models are constructed using the
convenience function `al.model_util.mge_model_from`, which handles the setup of Gaussian basis functions
with appropriate sigma ranges and linked parameters.

__Contents__

**MGE Advantages for Group Lenses:** The MGE is especially important for group-scale lenses because adding an extra.
**Model:** Compose the lens model fitted to the data.
**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Search:** Configure the non-linear search used to fit the model.
**Analysis:** Create the Analysis object that defines how the model is fitted to the data.
**Run Times:** Profiling the expected run time of the model-fit.
**Result:** Overview of the results of the model-fit.

__MGE Advantages for Group Lenses__

The MGE is especially important for group-scale lenses because adding an extra galaxy with a traditional Sersic
light profile introduces 5 non-linear parameters per galaxy. For a group with many extra galaxies, this makes the
model prohibitively complex and slow to fit.

In contrast, an MGE uses **linear light profiles** whose intensities are solved via linear algebra. This means that
adding an extra galaxy with an MGE light profile adds **zero** additional non-linear parameters to the model. The
only non-linear parameters for each galaxy are the centre and elliptical components, which for extra galaxies are
typically fixed to their observed values.

Furthermore, MGE models capture irregular and asymmetric galaxy morphologies (e.g. isophotal twists, radially
varying ellipticity) far more effectively than symmetric Sersic profiles, leading to more accurate lens models.

The combination of fewer non-linear parameters and better morphological accuracy makes MGE the recommended
approach for group-scale lens modeling.

__Model__

This script fits an `Imaging` dataset of a 'group-scale' strong lens where:

 - Each main lens galaxy's light is an MGE with 20 Gaussians [~4 non-linear parameters per galaxy].
 - Each main lens galaxy's total mass distribution is an `Isothermal`, with `ExternalShear` on `lens_0` [7 parameters].
 - Each extra galaxy's light is an MGE with 10 Gaussians with fixed centres [0 non-linear parameters per galaxy].
 - Each extra galaxy's total mass distribution is an `IsothermalSph` with bounded Einstein radius [1 parameter per galaxy].
 - The source galaxy's light is an MGE with 20 Gaussians [~4 non-linear parameters].

__Simulation__

This script fits a simulated `Imaging` dataset of a strong lens, which is produced in the
script `autolens_workspace/*/group/simulator.py`.
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

The model-fit requires a 2D mask defining the regions of the image we fit the lens model to the data.

We create a 7.5 arcsecond circular mask and apply it to the `Imaging` object that the lens model fits. This is
larger than a typical galaxy-scale lens mask because the group-scale lens has emission spread over a wider area
due to the multiple lens galaxies.
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

The centres of the main lens galaxies and extra galaxies are loaded from JSON files in the dataset directory.
These centres are used to set up the MGE models and mass models for each galaxy.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Model__

We compose a lens model where every galaxy uses an MGE for its light profile, constructed via the
`al.model_util.mge_model_from` convenience function.

For **main lens galaxies**, we use 20 Gaussians with uniform centre priors, allowing the MGE to capture
the full morphology of the main lens light. Only `lens_0` carries an `ExternalShear`.

For **extra galaxies**, we use 10 Gaussians with centres fixed to the observed positions. This is crucial:
because the MGE intensities are linear parameters, adding extra galaxies with fixed centres introduces
**zero** additional non-linear parameters. A Sersic model would add 5 non-linear parameters per galaxy.

For the **source galaxy**, we use 20 Gaussians with a single basis group and Gaussian centre priors.
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

    # Extra Galaxy Light (MGE with fixed centre -- zero non-linear parameters)

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

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    extra_galaxies=extra_galaxies,
)

"""
The `info` attribute shows the model in a readable format.

This shows the group scale MGE model, with separate entries for each main lens galaxy (e.g. `lens_0`),
the source galaxy and the extra galaxies collection. Note how the extra galaxies have many Gaussian light
profiles but no non-linear light parameters -- their intensities are all solved via linear algebra.
"""
print(model.info)

"""
__Over Sampling__

Over sampling at each galaxy centre (both main lens galaxies and extra galaxies) is performed to ensure the lens
calculations are accurate across the full field of the group.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Search__

The lens model is fitted to the data using the nested sampling algorithm Nautilus.

We use 150 live points, which is sufficient for the MGE model despite the group-scale complexity, because
the MGE parameterization has a much simpler non-linear parameter space than Sersic-based models.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="multi_gaussian_expansion",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
)

"""
__Analysis__

We next create an `AnalysisImaging` object with JAX acceleration enabled.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

"""
__VRAM Use__

For MGE models, VRAM use scales with the total number of Gaussians across all galaxies. With 20 Gaussians
for the main lens, 10 per extra galaxy, and 20 for the source, the total is moderate but larger than a
simple Sersic-based model. Check VRAM usage before running on GPU.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Times__

The MGE model has a slower per-evaluation time than Sersic models because the image of every Gaussian must
be computed. However, the much simpler non-linear parameter space means Nautilus converges in far fewer
iterations, so the overall run time is typically shorter.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format.

The result contains entries for each main lens galaxy (e.g. `lens_0`), the source galaxy and the extra galaxies,
all with their MGE light profiles and inferred intensities.
"""
print(result.info)

print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script has demonstrated how to use MGE light profiles for group-scale lens modeling. The key advantages are:

 - **No additional non-linear parameters per extra galaxy**: MGE intensities are linear, so adding extra galaxies
   does not increase the dimensionality of the non-linear parameter space.

 - **Better morphological accuracy**: MGE captures irregular features like isophotal twists and radially varying
   ellipticity that symmetric Sersic profiles cannot.

 - **Simpler parameter space**: The MGE parameterization removes degeneracies related to galaxy size parameters,
   making the non-linear search converge faster.

For group-scale lenses with many extra galaxies, MGE is strongly recommended over Sersic-based models.

__Features__

We recommend you also checkout:

- ``scaling_relation``: Model the mass of extra galaxies using a luminosity-to-mass scaling relation.
- ``pixelization``: Reconstruct the source using an adaptive mesh for even more accurate source modeling.
"""

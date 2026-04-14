"""
Modeling Features: Operated Light Profiles (Group)
==================================================

Operated light profiles are light profiles which are assumed to have already been convolved with the PSF. This means
that during the model-fitting process, these profiles are NOT convolved again with the PSF, unlike standard light
profiles.

This is useful when modeling data where the lens light subtraction was performed using PSF-convolved models, or
when a galaxy has compact point-source emission (e.g. an AGN) that has already been blurred by the telescope optics.
The operated profile is fitted directly to this already-convolved emission.

For a group-scale lens, this feature can be applied to both the main lens galaxies and the extra galaxies. Each
galaxy may have a compact nuclear component that is best represented as an operated light profile.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Centres:** The centres of the main lens galaxies and extra galaxies are loaded from JSON files.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Model:** Compose the lens model fitted to the data.
**Search:** Configure the non-linear search used to fit the model.
**Analysis:** Create the Analysis object that defines how the model is fitted to the data.
**Result:** Overview of the results of the model-fit.

__Model__

This script fits an ``Imaging`` dataset of a 'group-scale' strong lens where:

 - Each main lens galaxy's light is a linear ``Sersic`` bulge plus an operated linear ``Gaussian`` PSF component.
 - The first main lens galaxy's total mass distribution is an ``Isothermal`` and ``ExternalShear``.
 - There are two extra lens galaxies with linear operated ``Sersic`` light and ``IsothermalSph`` total mass
   distributions, with centres fixed to the observed centres of light.
 - The source galaxy's light is a linear ``SersicCore`` (which IS convolved with the PSF as normal).

__Start Here Notebook__

If any code in this script is unclear, refer to the ``group/modeling`` and
``imaging/features/advanced/operated_light_profile/modeling`` notebooks.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset `operated`, which includes lens light that has already been convolved with
the PSF.
"""
dataset_name = "operated"
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
        [
            sys.executable,
            "scripts/group/features/advanced/operated_light_profile/simulator.py",
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
__Mask__

We use a 7.5 arcsecond circular mask for group-scale lenses, which is larger than galaxy-scale because the
group has emission spread over a wider area.
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
__Centres__

Load the centres of the main lens galaxies and extra galaxies from JSON files.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

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
__Model__

We compose a group lens model where:

 - The main lens galaxy's light is a linear ``Sersic`` bulge plus an operated linear ``Gaussian`` PSF component.
   The operated ``Gaussian`` represents compact point-source emission (e.g. AGN) that has already been convolved
   with the telescope PSF. It is NOT convolved again during fitting.

 - The main lens galaxy's total mass distribution is an ``Isothermal`` and ``ExternalShear``.

 - The extra galaxies use linear operated ``Sersic`` light profiles. Because the data was simulated with
   operated profiles for these galaxies, we model them with the same type. Their centres are fixed to the
   observed centres of light.

 - The source galaxy's light is a linear ``SersicCore``, which IS convolved with the PSF as it represents
   genuine unconvolved source emission.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = af.Model(al.lp_linear.Sersic)
    psf = af.Model(al.lp_linear_operated.Gaussian)
    bulge.centre = psf.centre

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        psf=psf,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_dict[f"lens_{i}"] = lens

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    # Extra Galaxy Light (operated -- not convolved again with PSF)

    bulge = af.Model(al.lp_linear_operated.Gaussian)
    bulge.centre = centre

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    extra_galaxies=extra_galaxies,
)

"""
The ``info`` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Improved Lens Model__

The model above uses simple parametric profiles. For better performance, we replace the main lens galaxy's light
with an MGE model and the extra galaxies with MGE models. However, here the operated Gaussian PSF component
remains to demonstrate the operated light profile feature.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    psf = af.Model(al.lp_linear_operated.Gaussian)

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        psf=psf,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_dict[f"lens_{i}"] = lens

# Extra Galaxies:

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    # Extra Galaxy Light (MGE)

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

print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="operated_light_profiles",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
)

"""
__Analysis__

Create the ``AnalysisImaging`` object defining how the model is fitted to the data.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

"""
__Run Time__

The likelihood evaluation time for operated light profiles is faster than standard light profiles because the
PSF convolution step is omitted for those components. The overall run-time may be slightly slower due to the
additional ``psf`` parameter on the main lens galaxy.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result contains entries for each main lens galaxy (with its operated ``psf`` component), the source galaxy
and the extra galaxies.
"""
print(result.info)

print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script shows how to fit a group-scale lens model where compact emission is modeled using operated light
profiles. The operated profiles bypass PSF convolution, making them ideal for point-source emission or
pre-convolved light components.
"""

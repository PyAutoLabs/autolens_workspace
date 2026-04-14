"""
Modeling Features: Sky Background (Group)
=========================================

The background of an image is the light that is not associated with the strong lens we are interested in. This is
due to light from the sky, zodiacal light, and light from other galaxies in the field of view.

The background sky is often subtracted from image data during data reduction. If this subtraction is perfect,
there is no need to include the sky in the model. However, achieving a perfect subtraction is difficult, and
residuals can leave a signal degenerate with the lens galaxy light, especially for low surface brightness features.

For group-scale lenses, this is particularly important because the larger field of view means more area is
affected by sky background uncertainties, and the faint outskirts of multiple group galaxies can be affected.

This example illustrates how to include the sky background in the model-fitting of a group-scale ``Imaging``
dataset as a non-linear free parameter.

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

 - The sky background is included as a ``DatasetModel`` with a free ``background_sky_level`` parameter.
 - Each main lens galaxy's light is an MGE bulge.
 - The first main lens galaxy's total mass distribution is an ``Isothermal`` and ``ExternalShear``.
 - There are two extra lens galaxies with MGE light and ``IsothermalSph`` total mass distributions.
 - The source galaxy's light is an MGE.

__Start Here Notebook__

If any code in this script is unclear, refer to the ``group/modeling`` and
``imaging/features/advanced/sky_background/modeling`` notebooks.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset ``sky_background``, which has not had the sky background subtracted.
"""
dataset_name = "sky_background"
dataset_path = Path("dataset", "group", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/sky_background/simulator.py",
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

We use a 7.5 arcsecond circular mask for group-scale lenses.
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

Over sampling at each galaxy centre (both main lens galaxies and extra galaxies).
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

We compose a group lens model that includes a sky background component:

 - The sky background is modeled as a ``DatasetModel`` with a free ``background_sky_level`` parameter.
   This is not part of the ``galaxies`` collection but is a separate model component.

 - The main lens galaxies use MGE light profiles and isothermal mass profiles. Only the first main lens
   galaxy carries an ``ExternalShear``.

 - The extra galaxies use MGE light profiles with fixed centres and isothermal mass profiles.

 - The source galaxy uses an MGE light profile.

The prior on ``background_sky_level`` must be set manually based on the expected sky level in the data.
In this example, the true sky level is 5.0 electrons per second.
"""
# Main Lens Galaxies:

lens_models = []

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

    lens_models.append(lens)

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

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Sky Background:

dataset_model = af.Model(al.DatasetModel)
dataset_model.background_sky_level = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

# Overall Lens Model:

lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
lens_dict["source"] = source

model = af.Collection(
    dataset_model=dataset_model,
    galaxies=af.Collection(**lens_dict),
    extra_galaxies=extra_galaxies,
)

"""
The ``info`` attribute shows the model in a readable format. This confirms that the sky is a model component
that is not part of the ``galaxies`` collection.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="sky_background",
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

Adding the background sky model to the analysis has a negligible impact on the run time, as it simply adds a
constant value to the data. The run time is dominated by the group galaxy light and mass model evaluation.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result contains the inferred ``background_sky_level`` alongside the group galaxy model parameters.
"""
print(result.info)

print(result.instance.dataset_model.background_sky_level)

print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Wrap Up__

This script shows how to include the sky background as part of a group-scale lens model using a ``DatasetModel``
object. This ensures uncertainties on galaxy light profile parameters fully account for sky background
subtraction errors, which is especially important for the extended faint emission of group-scale lenses.
"""

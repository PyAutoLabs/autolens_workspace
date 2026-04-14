"""
Modeling Features: No Lens Light (Group)
========================================

This script models a group-scale strong lens where none of the lens galaxies have visible light emission. In the
group context, "no lens light" means that **all** main lens galaxies **and** all extra galaxies are modeled with
mass profiles only — no light profiles at all. Only the source galaxy has light.

This is the group-scale analogue of `imaging/features/no_lens_light/modeling.py`. The key difference is that
removing light from a group has a much larger impact on the model dimensionality: every main lens galaxy and
every extra galaxy that would normally require an MGE light model (adding non-linear parameters for centre,
ellipticity, etc.) now has only mass parameters. For a group with one main lens and two extra galaxies, this
removes all light-related non-linear parameters from the lens side of the model.

__Contents__

**Advantages:** The main advantage of fitting group data without lens light is the dramatic reduction in model.
**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Centres:** The centres of both the main lens galaxies and the extra galaxies are loaded from JSON files.
**Model:** Compose the lens model fitted to the data — all galaxies have mass only.
**Over Sampling:** Not needed for lens light when there is none.
**Search:** Configure the non-linear search used to fit the model.
**Analysis:** Create the Analysis object that defines how the model is fitted to the data.
**Result:** Overview of the results of the model-fit.

__Advantages__

The main advantage of fitting group data without lens light is the dramatic reduction in model dimensionality.
In a standard group fit, every main lens galaxy and every extra galaxy requires an MGE light model, each
contributing non-linear parameters (centre, ellipticity, etc.). By omitting all lens light:

 - The number of non-linear parameters drops substantially.
 - The non-linear search converges much faster.
 - The parameter space is simpler and less prone to local maxima.

This is especially powerful for groups with many galaxies, where the light model would otherwise dominate
the parameter budget.

__Model__

This script fits an `Imaging` dataset of a 'group-scale' strong lens where:

 - There is a main lens galaxy whose total mass distribution is an `Isothermal` and `ExternalShear` — no light.
 - There are two extra lens galaxies whose total mass distributions are `IsothermalSph` models — no light.
 - The source galaxy's light is a Multi Gaussian Expansion.

__Start Here Notebook__

If any code in this script is unclear, refer to the `group/modeling.py` script for the full group modeling API.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset `simple__no_lens_light`, which is the dataset we will use to perform
lens modeling. This dataset contains only lensed source emission — no lens galaxy light.
"""
dataset_name = "simple__no_lens_light"
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
        [sys.executable, "scripts/group/features/no_lens_light/simulator.py"],
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

We create a 7.5 arcsecond circular mask and apply it to the `Imaging` object that the lens model fits.
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

The centres of both the main lens galaxies and the extra galaxies are loaded from JSON files in the dataset
directory. This makes the script reusable across different datasets without hardcoding centre values.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Model__

We compose a lens model where all galaxies have mass only — no light profiles:

 - The main lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - There are two extra lens galaxies with `IsothermalSph` total mass distributions, with centres fixed to
   the observed centres and bounded Einstein radii [2 parameters].

 - The source galaxy's light is a Multi Gaussian Expansion [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.

Compare this to the standard group model (with MGE light for all galaxies) which would have N=28 or more.
The removal of all lens light profiles is what makes this so efficient.

__Model Composition (List-Based API)__

The API below uses the same list-based approach as `group/modeling.py`, but every galaxy is created without
a `bulge` parameter — mass only.
"""
# Main Lens Galaxies (mass only):

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    mass = af.Model(al.mp.Isothermal)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

    lens_dict[f"lens_{i}"] = lens

# Extra Galaxies (mass only):

extra_galaxies_list = []

for centre in extra_galaxies_centres:

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source (MGE light):

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

This confirms that no lens galaxy (main or extra) has a light profile — only mass profiles are present.
"""
print(model.info)

"""
__Over Sampling__

When there is no lens light, we do not need adaptive over-sampling for the lens galaxies. Over-sampling is
normally applied at the centres of lens galaxies to ensure their light profiles are evaluated accurately on a
higher-resolution grid. Since no galaxy has a light profile in this model, this step is unnecessary.

The source galaxy uses a cored light profile (`SersicCore`) which changes gradually in its central regions,
so it does not require over-sampling either.
"""

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus.

Given the reduced model complexity (no lens light parameters), we use `n_live=100` which is sufficient for
this relatively simple model.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="no_lens_light",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

"""
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

The `info` attribute shows the model in a readable format.

This confirms there is no lens galaxy light in the model-fit — only mass profiles for the main lens and
extra galaxies, plus the source light.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

__Wrap Up__

This script shows how to fit a group-scale lens model to data where no galaxy has visible light.

The key advantage in the group context is the dramatic reduction in model dimensionality. For a group with
one main lens and two extra galaxies, removing the MGE light model from all three galaxies eliminates all
light-related non-linear parameters, leaving only the mass and source parameters. This makes the model-fit
much faster and more robust.
"""

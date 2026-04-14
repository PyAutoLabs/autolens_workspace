"""
Modeling Features: Linear Light Profiles (Group)
================================================

This script fits a group-scale strong lens using **linear light profiles**, where the ``intensity`` of every
light profile is solved analytically via linear algebra rather than being a free parameter in the non-linear
search.

For a group-scale lens this is especially beneficial: the group model contains many galaxies (main lenses and
extra galaxies), and each galaxy's light profile would normally add an ``intensity`` parameter. By using linear
light profiles, none of these contribute to the non-linear parameter space, significantly reducing dimensionality
and improving sampling efficiency.

__Contents__

**Advantages:** Linear light profiles remove `intensity` from the non-linear parameter space.
**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Centres:** The centres of the main lens galaxies and extra galaxies are loaded from JSON files.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Model:** Compose the lens model fitted to the data.
**Search:** Configure the non-linear search used to fit the model.
**Analysis:** Create the Analysis object that defines how the model is fitted to the data.
**Result:** Overview of the results of the model-fit.
**Intensities:** How to extract the solved-for intensity values from the result.

__Advantages__

Each light profile's ``intensity`` parameter is solved via a linear inversion, reducing the dimensionality of
non-linear parameter space by the number of light profiles. This also removes the degeneracies between
``intensity`` and other light profile parameters (e.g. ``effective_radius``, ``sersic_index``), producing
more reliable lens model results that converge in fewer iterations.

For group-scale lenses with many galaxies, this reduction is particularly impactful: every main lens galaxy
and every extra galaxy has its intensity removed from the search.

__Model__

This script fits an ``Imaging`` dataset of a 'group-scale' strong lens where:

 - Each main lens galaxy's light is a linear ``Sersic`` bulge [6 parameters].
 - The first main lens galaxy's total mass distribution is an ``Isothermal`` and ``ExternalShear`` [7 parameters].
 - There are two extra lens galaxies with linear ``SersicSph`` light and ``IsothermalSph`` total mass
   distributions, with centres fixed to the observed centres of light [8 parameters].
 - The source galaxy's light is a linear ``SersicCore`` [5 parameters].

__Start Here Notebook__

If any code in this script is unclear, refer to the ``group/modeling`` and
``imaging/features/linear_light_profiles/modeling`` notebooks.
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

We use a 7.5 arcsecond circular mask, which is larger than a typical galaxy-scale mask because the group-scale
lens has emission spread over a wider area.
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

We compose a lens model using linear light profiles for all galaxies. The key difference from the standard
group modeling script is that we use ``al.lp_linear.Sersic``, ``al.lp_linear.SersicSph`` and
``al.lp_linear.SersicCore`` instead of their standard ``al.lp`` counterparts. These linear profiles have no
``intensity`` parameter -- it is solved via linear algebra.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = af.Model(al.lp_linear.Sersic)

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

    # Extra Galaxy Light (linear -- no intensity parameter)

    bulge = af.Model(al.lp_linear.SersicSph)

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

This confirms that light profiles of all galaxies do not include an ``intensity`` parameter.
"""
print(model.info)

"""
__Over Sampling__

Over sampling at each galaxy centre (both main lens galaxies and extra galaxies) is performed to ensure
the lens calculations are accurate across the full field of the group.
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

The lens model is fitted to the data using a non-linear search. Because the linear light profiles reduce the
dimensionality of the parameter space, the fit is more efficient than using standard light profiles.
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="linear_light_profiles",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
)

"""
__Analysis__

We create an ``AnalysisImaging`` object. The linear light profiles are handled automatically by the analysis
object -- when it detects linear light profiles in the model, it performs the linear inversion to solve for
their intensities during every likelihood evaluation.
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

The result ``info`` confirms that ``intensity`` parameters are not inferred by the model-fit.
"""
print(result.info)

print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result.samples)

"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not
displayed in the ``model.results`` file.

To extract the ``intensity`` values of a specific component in the model, we use the
``max_log_likelihood_tracer``, which has already performed the inversion and therefore the galaxy light
profiles have their solved-for ``intensity`` values associated with them.
"""
tracer = result.max_log_likelihood_tracer

print(f"Main lens galaxy bulge intensity: {tracer.galaxies[0].bulge.intensity}")
print(f"Source galaxy bulge intensity: {tracer.galaxies[-1].bulge.intensity}")

"""
The ``Tracer`` contained in the ``max_log_likelihood_fit`` also has the solved-for ``intensity`` values:
"""
fit = result.max_log_likelihood_fit

tracer = fit.tracer

print(f"Main lens galaxy bulge intensity (via fit): {tracer.galaxies[0].bulge.intensity}")

"""
Fin.
"""

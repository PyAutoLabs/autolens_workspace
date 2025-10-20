"""
Start Here: Cluster
===================

Cluster scale lenses are composed of:

 - Brightest Cluster Galaxies (BCG) which are modeled individually.
 - One or more large scale dark matter halos (typically > 1e14 MSun) which are modeled individually.
 - 50 - 100 member galaxies, whose collective mass contributes to ray tracing significantly and therefore all are modeled.
 - 5-50 source galaxies, all at different redshifts, which are all modeled individually.

This script shows you how to model cluster lens system using **PyAutoLens** with as little setup
as possible. In about 15 minutes you’ll be able to point the code at your own cluster catalogue and FITS files and
fit your first cluster-scale lens.

We focus on a *cluster-scale* lens (20 + lenses, many sources). If you have a single lens galaxy responsible for
most th lensing, lensing a single source, you should instead checlout the `start_here_group.ipynb` example.

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (a few minutes instead of an hour). If you don’t have
a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.

__Beta Feature__

Modeling strong lens clusters with PyAutoLens is a feature in beta testing, and there are many deficiencies with
the current implementation:

- Visualization is not optimal for cluster models with many lens and sources.
- Documentation on the workspace is limited compared to other features.

However, the PyAutoLens cluster implementation has a key feature which means you may still want to use it over
more established software. For lens modeling, the JAX GPU likelihood evaluation (which for those familiar with cluster
modeling uses an image plane chi squared) is over 50 times faster than existing established cluster modeling software.
It also fully supports multi-plane ray tracing of any complexity.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code below state you are not in a Colab environment and skip the setup.
"""

from autoconf import setup_colab

setup_colab.for_autolens(
    raise_error_if_not_gpu=True  # Switch to False for CPU Google Colab
)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We begin by loading CCD imaging of the cluster dataset. 

The `pixel_scales` value converts pixel units into arcseconds. It is critical you set this
correctly for your data.

The image itself is not used for cluster modeling, but plotting it shows the cluster configuration
and where the lens and source galaxies are.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "cluster" / dataset_name

data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)

array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()


"""
__Main Galaxies and Extra Galaxies__

For a group-scale lens, we designate there to be two types of lens galaxies in the system:

 - `main_galaxy`: The main lens galaxy which likely make up the majority of light and mass in the lens system.
 These are modeled individually with a unique name for each, with their light and mass distributions modeled using 
 parametric models.
 
 - `extra_galaxies`: The extra galaxies which are nearby the group lens system, whose mass contribute to the lensing 
 of the source galaxy. These are modeled with a more restrictive model, for example with their are centres fixed to the 
 observed centre of light. These are grouped into a single `extra_galaxies` collection.
 
__Centres__

For group-scale lenses we must manually specify the centres of the extra galaxies, which are fixed to the observed
centres of light of the galaxies. This is integral to ensuring the lens model can be fitted accurately, without these
centres being input there is a high chance the model will not converge to the correct solution.

In this example, we simply load the centres from a .json file contained in the dataset folder. After modeling the
data, this example will provide a GUI for you to determine the centres of the extra galaxies in your own data,
if they are not already known.
"""
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Masking__

Lens modeling does not need to fit the entire image, only the region containing lens and
source light, and the light of extra galaxies in the group. We therefore define a circular mask around all galaxies.

- Make sure the mask fully encloses the lensed arcs, lens galaxy and extra galaxies.
- Avoid masking too much empty sky, as this slows fitting without adding information.

We’ll also oversample the central pixels, which improves modeling accuracy without adding
unnecessary cost far from the lens. Over sampling is also applied to the extra galaxies.
"""
mask_radius = 3.7

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

# Over sampling is important for accurate lens modeling, but details are omitted
# for simplicity here, so don't worry about what this code is doing yet!

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)] + extra_galaxies_centres.in_list,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

To perform lens modeling we must define a lens model, describing the light profiles of the lens and source galaxies,
and the mass profile of the lens galaxy. This includes the mass of the groups extra galaxies.

A brilliant lens model to start with is one which uses a Multi Gaussian Expansion (MGE) to model the lens and source
light, and a Singular Isothermal Ellipsoid (SIE) plus shear to model the lens mass. 

Full details of why this models is so good are provided in the main workspace docs, but in a nutshell it 
provides an excellent balance of being fast to fit, flexible enough to capture complex galaxy morphologies and 
providing accurate fits to the vast majority of strong lenses. For group scale lenses, the MGE allows us to fit
the light of extra galaxies without increasing the number of free parameters in the model.

The MGE model composition API is quite long and technical, so we simply load the MGE models for the lens and source 
below via a utility function `mge_model_from` which hides the API to make the code in this introduction example ready 
to read. We then use the PyAutoLens Model API to compose the over lens model.
 
Note how we also loop over the extra galaxy centres, creating an MGE light model and SIE mass model for each extra 
galaxy fixed to the input centre.
"""
# Main Lens:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Extra Galaxies

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Light

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_fixed=extra_galaxy_centre,
        use_spherical=True,
    )

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
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
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

"""
We can print the model to show the parameters that the model is composed of, which shows many of the MGE's fixed
parameter values the API above hided the composition of.
"""
print(model.info)

"""
__Model Fit__

We now fit the data with the lens model using the non-linear fitting method and nested sampling algorithm Nautilus.

This requires an `AnalysisImaging` object, which defines the `log_likelihood_function` used by Nautilus to fit
the model to the imaigng data.
"""
search = af.Nautilus(
    path_prefix=Path("group"),  # The path where results and output are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # For fast GPU fitting lens model fits are batched and run simultaneously.
    iterations_per_quick_update=2500,  # Every N iterations the max likelihood model is visualized and written to output folder.
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

Now this is running you should checkout the `autolens_workspace/output` folder, where many results of the fit
are written in a human readable format (e.g. .json files) and .fits and .png images of the fit are stored.

When the fit is complex, we can print the results by printing `result.info`.
"""
print(result.info)

"""
The result also contains the maximum likelihood lens model which can be used to plot the best-fit lensing information
and fit to the data.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
The result object contains pretty much everything you need to do science with your own strong lens, but details
of all the information it contains are beyond the scope of this introductory script. The `guides` and `result` 
packages of the workspace contains all the information you need to analyze your results yourself.

__Centre Input GUI__

__Model Your Own Lens__

If you have your own strong lens imaging data, you are now ready to model it yourself by adapting the code above
and simply inputting the path to your own .fits files into the `Imaging.from_fits()` function.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- Supply your own CCD image, PSF, and RMS noise-map.
- Ensure the lens galaxy is roughly centered in the image.
- Ensure you input the centres of the extra galaxies in the group correctly.
- Double-check `pixel_scales` for your telescope/detector.
- Adjust the mask radius to include all relevant light.
- Start with the default model — it works very well for pretty much all group with < 5 extra galaxies!

__Simulator__

In the galaxy-scale examples (`start_here_imaging.ipynb`, `start_here_interferometer.ipynb`, `start_here_point_source.ipynb`)
we illustrate how to simulate strong lens images. 

For group scale lenses, we omit this, as it is quite techinical and long. The `autolens_workspace/*/group/simulator` 
package has examples of how to simulate group scale lenses if you are interested.

__Scaling Relations__

This example models the mass of each galaxy individually, which means the number of dimensions of the model increases
as we model group scale lenses with more galaxies. This can lead to a model that is slow to fit and poorly constrained.
There may also not be enough information in the data to constrain every galaxy's mass.

A common approach to overcome this is to put many of the extra galaxies a scaling relation, where the mass of the 
galaxies are related to their light via a observationally motivated scaling relation. This means that as more 
galaxies are included in the lens model, the dimensionality of the model does not increase. Furthermore, their 
luminosities act as priors on their masses, which helps ensure the model is well constrained.

Lens modeling using scaling relations is fully support and described in the `features/scaling_relation.ipynb` example.
If your group has many extra galaxies (e.g. more than 5) you probably want to read this example once you are confident
with this one.

In the near future (Novembver 2026) we will provide more extensive group scale lens modeling examples which ensure
that complex groups with 10+ extra galaxies can be fitted efficiently and robustly using scaling relations. PyAutoLens
can do a good jbo now, but big improvements are coming!

__Wrap Up__

This script has shown how to model CCD imaging data of group-scale strong lenses.

Details of the **PyAutoLens** API and how lens modeling works were omitted for simplicity, but everything you need to 
know is described throughout the main workspace documentation. You should check it out, but maybe you want to try and 
model your own lens first!

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/*/modeling/group`: A full description of the lens modeling API and how to customize your model-fits.
- `autolens_workspace/*/simulators/group`: A full description of the lens simulation API and how to customize your simulations.
- `autolens_workspace/*/data_preparation/group`: How to load and prepare your own imaging data for lens modeling.
- `autolens_workspace/results`: How to load and analyze the results of your lens model fits, including tools for large samples.
- `autolens_workspace/guides`: A complete description of the API and information on lensing calculations and units.
- `autolens_workspace/feature`: A description of advanced features for lens modeling, for example pixelized source reconstructions, read this once you're confident with the basics!
"""

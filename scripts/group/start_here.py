"""
Start Here: Group
=================

Group scale lenses typically have a single "main" lens galaxy and 2-10 smaller extra galaxies nearby, whose light
may blur with the source light and whose mass contributes significantly to the ray-tracing, meaning both are
therefore included in the analysis, Groups typically also have just one lensed source.

This script shows you how to model a group lens system using **PyAutoLens** with as little setup
as possible. In about 15 minutes you’ll be able to point the code at your own FITS files and
fit your first group-scale lens.

We focus on a *group-scale* lens (a single lens galaxy with some extra galaxies nearby). If you have a single
lens galaxy, see the `start_here_imaging.ipynb` example, if your system has many lens and sources galaxies
see `start_here_cluster.ipynb` example.

This example uses Euclid CCD imaging data, but the workflow for interferometer data on group scale lenses is similar.
The lens has only 2 extra galaxies, so the model is not too complex, meaning this example runs in about 10 minutes on a
good GPU. More complex groups with more extra galaxies will take longer to fit, but the workflow is identical and
PyAutoLens can efficient scale to these more complex systems.

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (a few minutes instead of an hour). If you don’t have
a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.

Finally, we also show how to simulate strong lens groups. This is useful for practice, for
building training datasets, or for investigating lensing effects in a controlled way.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
"""

import subprocess
import sys

try:
    import google.colab

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autoconf", "--no-deps"]
    )
except ImportError:
    pass

from autoconf import setup_colab

setup_colab.for_autolens(
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
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

We begin by loading the dataset. Three ingredients are needed for lens modeling:

1. The image itself (CCD counts).
2. A noise-map (per-pixel RMS noise).
3. The PSF (Point Spread Function).

Here we use HST imaging of a Euclid group-scale strong lens. Replace these FITS paths with your own to
immediately try modeling your data.

The `pixel_scales` value converts pixel units into arcseconds. It is critical you set this
correctly for your data.
"""
dataset_name = "102021990_NEG650312660474055399"
dataset_path = Path("dataset") / "group" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

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
    n_live=150,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # For fast GPU fitting lens model fits are batched and run simultaneously.
    iterations_per_full_update=100000,  # Every N iterations the results are written to hard-disk for inspection.
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

The centres of the extra galaxies above were loaded from a .json file, which was create using a GUI where one simply
clicks the centres of the extra galaxies on the image. 

For your own group lens, if you do not know the centres of the extra galaxies already, you can use the GUI below
to do this yourself. It will output a .json file in the dataset folder you can then load and use in the model above.
"""
search_box_size = (
    3  # Size of the search box to find the brightest pixel around your click
)

try:
    clicker = al.Clicker(
        image=dataset.data,
        pixel_scales=dataset.pixel_scales,
        search_box_size=search_box_size,
    )

    extra_galaxies_centres = clicker.start(
        data=dataset.data,
        pixel_scales=dataset.pixel_scales,
    )

    al.output_to_json(
        file_path=dataset_path / "extra_galaxies_centres.json",
        obj=extra_galaxies_centres,
    )
except Exception as e:
    print(
        """
        Problem loading GUI, probably an issue with TKinter or your matplotlib TKAgg backend.
        
        You will likely need to try and fix or reinstall various GUI / visualization libraries, or try
        running this example not via a Jupyter notebook.
        
        There are also manual tools for performing this task in the workspace.
        """
    )
    print()
    print(e)

"""
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

__Scaling Relations__

This example above models the mass of each galaxy individually, which means the number of dimensions of the model 
increases as we model group scale lenses with more galaxies. This can lead to a model that is slow to fit and poorly 
constrained. There may also not be enough information in the data to constrain every galaxy's mass.

A common approach to overcome this is to put many of the extra galaxies a scaling relation, where the mass of the 
galaxies are related to their light via a observationally motivated scaling relation. This means that as more 
galaxies are included in the lens model, the dimensionality of the model does not increase. Furthermore, their 
luminosities act as priors on their masses, which helps ensure the model is well constrained.

We now perform a fit using this scaling relation approach. Instead of the SIE model used for extra galaxies above,
we instead model the mass of ech extra galaxy using the dual Pseudo-Isothermal Elliptical (dPIE)
mass distribution introduced in Eliasdottir 2007: https://arxiv.org/abs/0710.5636.

It relates the luminosity of every galaxy to a cut radius (r_cut), a core radius (r_core) and a mass normaliaton b0:

$r_cut = r_cut^* (L/L^*)^{0.5}$

$r_core = r_core^* (L/L^*)^{0.5}$

$b0 = b0^* (L/L^*)^{0.25}$

The free parameters are therefore L^*, r_cut^*, r_core^* and b0^*.

We use this model because it is commonly used in studies of lensing groups and clusters to put member galaxies on a
scaling relation, thus it is more consistent with previous literature!

To perform scaling relation lens modeling, the luminosity of every member galaxy must have been measured and is input
below. If you want to put your own lens into this example, you'll need to have the luminosities measured yourself
already. Note also that the code below could easily be adapted to use stellar masses or velocity dispersions,
if you have those measurements instead.
"""
extra_galaxies_luminosity_list = [1e9, 1e9]

ra_star = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e11)
rs_star = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
b0_star = af.LogUniformPrior(lower_limit=1e5, upper_limit=1e7)
luminosity_star = 1e9

extra_galaxies_list = []

for extra_galaxy_centre, extra_galaxy_luminosity in zip(
    extra_galaxies_centres.in_list, extra_galaxies_luminosity_list
):

    mass = af.Model(al.mp.dPIEMassSph)

    mass.centre = extra_galaxy_centre

    mass.ra = ra_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.rs = rs_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.b0 = b0_star * (extra_galaxy_luminosity / luminosity_star) ** 0.25

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

"""
__Model Fit__

We now compose the model using the same API as before.
"""
# Main Lens:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

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
the `model.info` shows that this scaling relation has been used to setup the galaxy parameters.

Note how, although there are only 2 extra galaxies, adding extra galaxies now *no longer adds any free parameters** to
the model complexity!
"""
print(model.info)

"""
We now fit the model using the scaling relation.
"""
search = af.Nautilus(
    path_prefix=Path("group"),  # The path where results and output are stored.
    name="start_here_scaling_relation",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # For fast GPU fitting lens model fits are batched and run simultaneously.
    iterations_per_full_update=100000,  # Every N iterations the results are written to hard-disk for inspection.
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Simulator__

In the galaxy-scale examples (`start_here_imaging.ipynb`, `start_here_interferometer.ipynb`, `start_here_point_source.ipynb`)
we illustrate how to simulate strong lens images. 

For group scale lenses, we omit this, as it is quite techinical and long. The `autolens_workspace/*/group/simulator` 
package has examples of how to simulate group scale lenses if you are interested.

__Wrap Up__

This script has shown how to model CCD imaging data of group-scale strong lenses.

Details of the **PyAutoLens** API and how lens modeling works were omitted for simplicity, but everything you need to 
know is described throughout the main workspace documentation. You should check it out, but maybe you want to try and 
model your own lens first!

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/*/group/modeling`: A full description of the lens modeling API and how to customize your model-fits.
- `autolens_workspace/*/group/simulators`: A full description of the lens simulation API and how to customize your simulations.
- `autolens_workspace/*/group/data_preparation`: How to load and prepare your own imaging data for lens modeling.
- `autolens_workspace/guides/results`: How to load and analyze the results of your lens model fits, including tools for large samples.
- `autolens_workspace/guides`: A complete description of the API and information on lensing calculations and units.
- `autolens_workspace/group/features`: A description of advanced features for lens modeling, for example pixelized source reconstructions, read this once you're confident with the basics!
"""

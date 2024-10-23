"""
Modeling Features: Operated Light Profiles
==========================================

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or a compact knot of star formation.

This point-source emission is subject to blurring during data acquiziton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a linear parametric `Sersic` bulge.
 - The lens galaxy includes a linear parametric `Gaussian` psf.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `simple` via .fits files.
"""
dataset_name = "light_operated"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose a lens model where:

 - The lens galaxy's light is a linear parametric `Sersic` bulge [6 parameters].
 
 - The lens galaxy's point source emission is a linear parametric operated `Gaussian` centred on the bulge [3 parameters].
 
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=25.

This lens galaxy includes a `Gaussian` operated light profile (`lp_operated`) PSF, which accounts for the PSF
convolution of the point source emission.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
bulge = af.Model(al.lp_linear.Sersic)
psf = af.Model(al.lp_operated.Gaussian)
bulge.centre = psf.centre

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    psf=psf,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
There is also a linear variant of every operated light profile (see `light_parametric_linear.py`).

We will use this, as it simplifies parameter space, which is particularly important for operated light profiles 
which can prove quite difficult to sample robustly.

The number of free parameters and therefore the dimensionality of non-linear parameter space for this model is N=22.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)
psf = af.Model(al.lp_linear_operated.Gaussian)
bulge.centre = psf.centre

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, psf=psf, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp_linear.SersicCore)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the lens galaxy's light has a `Gaussian` PSF component.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="operated_light_profiles",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Time__

The likelihood evaluation time for operated light profiles are faster than all other light profiles. This is because
the computationally expensive PSF convolution step is omitted.

The overall run-time may be a little slower than other models though, because the `psf` component adds a few
extra parameters.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this does 
not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix this).

This confirms that the lens galaxy's light has a `Gaussian` PSF component.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

__Wrap Up__

This script shows how fit a lens model where bright point source emission is modeled using a `Gaussian` PSF.

It is uncommon for this feature to be necessary, as there are very few lens galaxies which have such bright point
source emission in their centre. Nevertheless, it has been observed in a select few systems, and is therefore 
something you may need to model.
"""

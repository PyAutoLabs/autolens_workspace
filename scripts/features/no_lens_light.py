"""
Modeling Features: No Lens Light
================================

CCD imaging data of a strong lens may not have lens galaxy light emission present, for example if the lens galaxy light
has already been subtracted from the image. 

This example illustrates how to fit a lens model to data where the lens galaxy's light is not present.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of fitting data without lens light.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Fit:** Perform a fit to a dataset without lens light.
**Model:** Composing a model without lens light and how it changes the number of free parameters.
**Search & Analysis:** Standard set up of non-linear search and analysis.
**Run Time:** Profiling of run times without lens light.
**Model-Fit:** Performs the model fit using standard API.
**Result:** Accessing results of model fit without lens light.

__Advantages__

The main advantage of fitting data without lens light is the reduction in the number of free parameters in the
model-fit. In the `start_here.py` example, the lens galaxy's light profile was composed of a `bulge` and `disk` which
were both Sersic profiles, which contributed an extra N=12 free parameters. Without the lens light we therefore
reduce the dimensionality of non-linear parameter space by 12, which makes the model-fit much faster and more efficient.

__Disadvantages__

If the lens light was simply not present in the observed data (e.g. it was too faint at the wavelength of the
observation) then there is no option but to fit the data without lens light, and therefore there is also no disadvantage.

If you are fitting data where the lens light was subtracted before lens modeling (e.g. using a Sersic
subtraction or GUI / b-splines based fitting), there are disadvantages.

The lens light subtraction process may not be clean. The lens and source light are blended with one another in the
data (e.g. due to PSF convlution). The process may over subtract source light in certain regions of the data and
fail to subtract lens light in other regions. This will distort the lensed source emission input into the lens
modeling and therefore potentially bias the inferred mass model.

Performing lens and source modeling simultanoeusly means that the model-fit can find the optimal deblending of the two
components whilst fully propagating the errors on the inferred model parameters forward.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
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

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files
"""
dataset_name = "simple__no_lens_light"
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
__Fit__

This is to illustrate the API for performing a fit without lens light using standard autolens objects like 
the `Galaxy`, `Tracer` and `FitImaging`.

We simply do not input a `bulge` with a light profile into the `lens`.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
By plotting the fit, we see that the lens light is not included in the model fit.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.

The lens galaxy does not include a light profile `bulge` or `disk` component, and thus its emission is not fitted for.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the lens galaxy's light is omitted from the model-fit.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="no_lens_light",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Time__

The likelihood evaluation time for fits to data without lens light are only small bit faster than fits to data with
lens light. This is because the most computationally expensive steps (e.g. computing the deflection angles, blurring
the image with the PSF) are performed for both model-fits.

However, the overall run-time will be faster than before, as the removal of the lens light reduces the dimensionality
of non-linear parameter space by 7 or more parameters. This means that the non-linear search will more efficiently
converge on the highest likelihood regions of parameter space.
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

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms there is no lens galaxy light in the model-fit.
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

This script shows how to fit a lens model to data where the lens galaxy's light is not present.

It was a straightforward extension to the modeling API illustrated in `start_here.ipynb`, where one simply removed
the light profiles from the lens galaxy's model.

Models where the source has no light, or other components of the model are omitted can also be easily composed using
the same API manipulation.
"""

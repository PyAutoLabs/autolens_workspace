"""
Modeling: Mass Total + Source Inversion
=======================================

This script fits `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `DelaunayMagnification` `Pixelization` and `Constant`
   regularization.
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
import numpy as np

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=3.0, sub_size=1
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files , which we will fit 
with the lens model.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
__Inversion Settings (Run Times)__

The run times of an interferometer `Inversion` depend significantly on the following settings:

 - `transformer_class`: whether a discrete Fourier transform (`TransformerDFT`) or non-uniform fast Fourier Transform
 (`TransformerNUFFT) is used to map the inversion's image from real-space to Fourier space.

 - `use_linear_operators`: whether the linear operator formalism or matrix formalism is used for the linear algebra.

The optimal settings depend on the number of visibilities in the dataset:

 - For N_visibilities < 1000: `transformer_class=TransformerDFT` and `use_linear_operators=False` gives the fastest
 run-times.
 - For  N_visibilities > ~10000: use `transformer_class=TransformerNUFFT`  and `use_linear_operators=True`.

The dataset modeled by default in this script has just 200 visibilties, therefore `transformer_class=TransformerDFT`
and `use_linear_operators=False`. If you are using this script to model your own dataset with a different number of
visibilities, you should update the options below accordingly.

The script `autolens_workspace/*/interferometer/profiling.py` allows you to compute the run-time of an inversion
for your interferometer dataset. It does this for all possible combinations of settings and therefore can tell you
which settings give the fastest run times for your dataset.
"""
settings_dataset = al.SettingsInterferometer(transformer_class=al.TransformerDFT)
settings_inversion = al.SettingsInversion(use_linear_operators=False)

"""
We now create the `Interferometer` object which is used to fit the lens model.

This includes a `SettingsInterferometer`, which includes the method used to Fourier transform the real-space 
image of the strong lens to the uv-plane and compare directly to the visiblities. We use a non-uniform fast Fourier 
transform, which is the most efficient method for interferometer datasets containing ~1-10 million visibilities.
"""
dataset = dataset.apply_settings(settings=settings_dataset)
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source-galaxy's light uses a `Rectangular` mesh with fixed resolution 30 x 30 pixels (0 parameters).
 
 - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6. 
 
It is worth noting the `Pixelization`  use significantly fewer parameters (1 parameter) than 
fitting the source using `LightProfile`'s (7+ parameters). 

__Coordinates__

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/imaging/modeling/customize/priors.py`).
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.DelaunayMagnification(shape=(30, 30)),
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Dynesty (see `start.here.py` for a 
full description).
"""
search = af.DynestyStatic(
    path_prefix=path.join("interferometer"),
    name="mass[sie]_source[pix]",
    unique_tag=dataset_name,
    nlive=50,
    number_of_cores=1,
)

"""
__Position Likelihood__

We add a penalty term ot the likelihood function, which penalizes models where the brightest multiple images of
the lensed source galaxy do not trace close to one another in the source plane. This removes "demagnified source
solutions" from the source pixelization, which one is likely to infer without this penalty.

A comprehensive description of why we do this is given at the following readthedocs page. I strongly recommend you 
read this page in full if you are not familiar with the positions likelihood penalty and demagnified source reconstructions:

 https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

__Brief Description__

Unlike other example scripts, we also pass the `AnalysisInterferometer` object below a `PositionsLHPenalty` object, 
whichincludes the positions we loaded above, alongside a `threshold`.

This is because `Inversion`'s suffer a bias whereby they fit unphysical lens models where the source galaxy is 
reconstructed as a demagnified version of the lensed source. These are covered in more detail in chapter 4 
of **HowToLens**. 

To prevent these solutions biasing the model-fit we specify a `position_threshold` of 0.5", which requires that a 
mass model traces the four (y,x) coordinates specified by our positions (that correspond to the brightest regions of the 
lensed source) within 0.5" of one another in the source-plane. If this criteria is not met, a large penalty term is
added to likelihood that massively reduces the overall likelihood. This penalty is larger if the ``positions``
trace further from one another.

This ensures the unphysical solutions that bias an `Inversion` have much lower likelihood that the physical solutions
we desire. Furthermore, the penalty term reduces as the image-plane multiple image positions trace closer in the 
source-plane, ensuring Dynesty converges towards an accurate mass model. It does this very fast, as 
ray-tracing just a few multiple image positions is computationally cheap. 

The threshold of 0.3" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. However, we only want the threshold to aid the non-linear with the choice of mass model in the intiial fit.

Position thresholding is described in more detail in the 
script `autolens_workspace/*/imaging/modeling/customize/positions.py`
"""
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)


positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=0.3)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` used by the non-linear search to fit the 
model to the `Interferometer`dataset.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood=positions_likelihood,
    settings_inversion=settings_inversion,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format:
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via dynesty.

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer,
    grid=real_space_mask.derive_grid.unmasked_sub_1,
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""

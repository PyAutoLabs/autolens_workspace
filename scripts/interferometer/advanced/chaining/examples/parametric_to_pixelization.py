"""
Chaining: Parametric To Pixelization
====================================

This script chains two searches to fit `Interferometer` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Exponential`.

The two searches break down as follows:

 1) Model the source galaxy using a parametric `Sersic` and lens galaxy mass as an `Isothermal`.
 2) Models the source galaxy using an `Inversion` and lens galaxy mass as an `Isothermal`.

__Why Chain?__

There are a number of benefits of chaining a parametric source model and `Inversion`, as opposed to fitting the
`Inversion` in one search:

 - Parametric sources are computationally faster to fit. Therefore, even though the `Sersic` has more
 parameters for the search to fit than an `Inversion`, the model-fit is faster overall.

 - `Inversion`'s often go to unphysical solutions where the mass model goes to high / low normalization_list and the source
 is reconstructed as a demagnified version of the image. (see Chapter 4, tutorial 6 for a complete description of
 this effect). This does not occur for a parametric source, therefore the mass model can be initialized using a
 parametric source, which sets up the search which fits an `Inversion` so as to not sample these unphysical solutions.

 - The `auto_positions` feature can be used to further ensure these unphysical solutions do not bias the model-fit.
 Auto-positions use the maximum log likelihood mass model of the first search to determine the image-plane position of
 the lensed source. In the second search, we then require that a mass model must trace these positions within a threshold
 arc-secoond value of one another in the source-plane, removing these unphysical solutions.

__Run Times and Settings__

The run times of an interferometer `Inversion` depend significantly on the following settings:

 - `transformer_class`: whether a discrete Fourier transform (`TransformerDFT`) or non-uniform fast Fourier Transform
 (`TransformerNUFFT) is used to map the inversion's image from real-space to Fourier space.

 - `use_linear_operators`: whether the linear operator formalism or matrix formalism is used for the linear algebra.

The optimal settings depend on the number of visibilities in the dataset:

 - For N_visibilities < 1000: `transformer_class=TransformerDFT` and `use_linear_operators=False` gives the fastest
 run-times.
 - For  N_visibilities > ~10000: use `transformer_class=TransformerNUFFT`  and `use_linear_operators=True`.

The dataset modeled by default in this script has just 200 visibilties, therefore `transformer_class=TransformerDFT`
and `use_linear_operators=False`.

The script `autolens_workspace/*/interferometer/run_times.py` allows you to compute the run-time of an inversion
for your interferometer dataset. It does this for all possible combinations of settings and therefore can tell you
which settings give the fastest run times for your dataset.
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
__Dataset + Masking__ 

Load, plot and mask the `Interferometer` data.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(151, 151), pixel_scales=0.05, radius=3.0
)

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

The script `autolens_workspace/*/interferometer/run_times.py` allows you to compute the run-time of an inversion
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
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 - The source galaxy's light is a parametric `Sersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search_1 = af.Nautilus(
    path_prefix=path.join("interferometer", "chaining", "parametric_to_pixelization"),
    name="search[1]__parametric",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisInterferometer(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_1.info)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is again an `Isothermal` and `ExternalShear` [7 parameters].
 - The source galaxy's light uses an `Overlay` image-mesh [2 parameters].
 
 - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].
 - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

The priors of the majority of the `Isothermal` and `ExternalShear` initialized using the results of search 1.

The term `model` below passes the source model as model-components that are to be fitted for by the 
non-linear search. We pass the `lens` as a `model`, so that we can use the mass model inferred by search 1. The source
does not use any priors from the result of search 1.
"""
lens = result_1.model.galaxies.lens

pixelization = af.Model(
    al.Pixelization,
    image_mesh=al.image_mesh.KMeans,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.ConstantSplit,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_2.info)

"""
__Analysis + Position Likelihood__

We add a penalty term ot the likelihood function, which penalizes models where the brightest multiple images of
the lensed source galaxy do not trace close to one another in the source plane. This removes "demagnified source
solutions" from the source pixelization, which one is likely to infer without this penalty.

A comprehensive description of why we do this is given at the following readthedocs page. I strongly recommend you 
read this page in full if you are not familiar with the positions likelihood penalty and demagnified source reconstructions:

 https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

__Brief Description__

In this example we update the positions between searches, where the positions correspond to the (y,x) locations of the 
lensed source's multiple images. When a model-fit uses positions, it requires them to trace within a threshold value of 
one another for every mass model sampled by the non-linear search. If they do not, a penalty term is added to the
likelihood penalizing that solution 

Below, we use the results of the first search to compute the lensed source positions that are input into search 2. The
code below uses the maximum log likelihood model mass model and source galaxy centre, to determine where the source
positions are located in the image-plane. 

We also use this result to set the `threshold`, whereby the threshold value is based on how close these positions 
trace to one another in the source-plane (using the best-fit mass model again). This threshold is multiplied by 
a `factor` to ensure it is not too small (and thus does not remove plausible mass  models). If, after this 
multiplication, the threshold is below the `minimum_threshold`, it is rounded up to this minimum value.
"""
analysis_2 = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood=result_1.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=settings_inversion,
)

"""
__Search + Model-Fit__

We now create the non-linear search and perform the model-fit using this model.
"""
search_2 = af.Nautilus(
    path_prefix=path.join("interferometer", "chaining", "parametric_to_pixelization"),
    name="search[2]__pixelization",
    unique_tag=dataset_name,
    n_live=80,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

The final results can be summarized via printing `info`.
"""
print(result_2.info)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize an `Isothermal` + `ExternalShear` lens mass model 
using a parametric source and pass this model to a second search which modeled the source using an `Inversion`. 

This was more computationally efficient than just fitting the `Inversion` by itself and helped to ensure that the 
`Inversion` did not go to an unphysical mass model solution which reconstructs the source as a demagnified version
of the lensed image.

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex lens modeling 
in a robust and efficient way. 

The following example pipelines fits an inversion, using the same approach demonstrated in this script of first fitting 
a parametric source:

 `autolens_workspace/interferometer/chaining/pipelines/mass_total__source_pixelization.py`

 __SLaM (Source, Light and Mass)__
 
An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling 
processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
mass model. 

The SLaM pipelines begin with a parametric Source pipeline, which then switches to an inversion Source pipeline, 
exploiting the chaining technique demonstrated in this example.
"""

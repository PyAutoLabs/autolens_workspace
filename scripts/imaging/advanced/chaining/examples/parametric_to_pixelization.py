"""
Chaining: Parametric To Pixelization
====================================

This script chains two searches to fit `Imaging` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Exponential`.

The two searches break down as follows:

 1) Model the source galaxy using a linear parametric `Sersic` and lens galaxy mass as an `Isothermal`.
 2) Models the source galaxy using an `Inversion` and lens galaxy mass as an `Isothermal`.

__Why Chain?__

There are a number of benefits of chaining a linear parametric source model and `Inversion`, as opposed to fitting the
`Inversion` in one search:

 - Parametric sources are computationally faster to fit. Therefore, even though the `Sersic` has more
 parameters for the search to fit than an `Inversion`, the model-fit is faster overall.

 - `Inversion`'s often go to unphysical solutions where the mass model goes to high / low normalization_list and the source
 is reconstructed as a demagnified version of the image. (see Chapter 4, tutorial 6 for a complete description of
 this effect). This does not occur for a linear parametric source, therefore the mass model can be initialized using a
 parametric source, which sets up the search which fits an `Inversion` so as to not sample these unphysical solutions.
      
 - The positions and positions threshold can be updated to further ensure these unphysical solutions do not bias the
 model-fit. The updated positions use the maximum log likelihood mass model of the first search to determine the
 image-plane position of the lensed source. In the second search, we then require that a mass model must trace these
 positions within a threshold arc-secoond value of one another in the source-plane, removing these unphysical solutions.

__Preloading__

When certain components of a model are fixed its associated quantities do not change during a model-fit. For
example, for a lens model where all light profiles are fixed, the PSF blurred model-image of those light profiles
is also fixed.

**PyAutoLens** uses _implicit preloading_ to inspect the model and determine what quantities are fixed. It then stores
these in memory before the non-linear search begins such that they are not recomputed for every likelihood evaluation.

In this example no preloading occurs.

__Start Here Notebook__

If any code in this script is unclear, refer to the `chaining/start_here.ipynb` notebook.
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
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)

dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "parametric_to_pixelization")

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__parametric",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_1.info)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is again an `Isothermal` and `ExternalShear` [7 parameters: 
 priors initialized from search 1].
 - The source galaxy's light uses an `Overlay` image-mesh [2 parameters].
 
 - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].
 - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel equally [1 parameter]. 
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

The term `model` below passes the source model as model-components that are to be fitted for by the 
non-linear search. We pass the `lens` as a `model`, so that we can use the mass model inferred by search 1. The source
does not use any priors from the result of search 1.
"""
lens = result_1.model.galaxies.lens

image_mesh = af.Model(al.image_mesh.Overlay)
image_mesh.shape = (30, 30)

pixelization = af.Model(
    al.Pixelization,
    image_mesh=image_mesh,
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
analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=result_1.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
)

"""
__Search + Model-Fit__

We now create the non-linear search and perform the model-fit using this model.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
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

 `autolens_workspace/imaging/chaining/pipelines/mass_total__source_pixelization.py`

 __SLaM (Source, Light and Mass)__
 
An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling 
processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
mass model. 

The SLaM pipelines begin with a parametric Source pipeline, which then switches to an inversion Source pipeline, 
exploiting the chaining technique demonstrated in this example.
"""

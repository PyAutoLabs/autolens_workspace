"""
Modeling: Mass Total + Source Inversion
=======================================

A pixelization reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a source galaxy model which uses a pixelization to reconstruct the source's light. A Delaunay
mesh and constant regularization scheme are used, which are the simplest forms of mesh and regularization
with provide computationally fast and accurate solutions in **PyAutoLens**.

For simplicity, the lens galaxy's light is omitted from the model and is not present in the simulated data. It is
straightforward to include the lens galaxy's light in the model.

Pixelizations are covered in detail in chapter 4 of the **HowToLens** lectures.

__Advantages__

Many strongly lensed source galaxies are complex, and have asymmetric and irregular morphologies. These morphologies
cannot be well approximated by a parametric light profiles like a Sersic, or many Sersics, and thus a pixelization
is required to reconstruct the source's irregular light.

Even basis functions like shapelets or a multi-Gaussian expansion cannot reconstruct a source-plane accurately
if there are multiple source galaxies, or if the source galaxy has a very complex morphology.

To infer detailed components of a lens mass model (e.g. its density slope, whether there's a dark matter subhalo, etc.)
then pixelized source models are required, to ensure the mass model is fitting all of the lensed source light.

There are also many science cases where one wants to study the highly magnified light of the source galaxy in detail,
to learnt about distant and faint galaxies. A pixelization reconstructs the source's unlensed emission and thus
enables this.

__Disadvantages__

Pixelizations are computationally slow, and thus the run times will be much longer than a parametric source model.
It is not uncommon for a pixelization to take hours or even days to fit high resolution imaging data (e.g. Hubble Space
Telescope imaging).

Lens modeling with pixelizations is also more complex than parametric source models, with there being more things
that can go wrong. For example, there are solutions where a demagnified version of the lensed source galaxy is
reconstructed, using a mass model which effectively has no mass or too much mass. These are described in detail below,
the point for now is that it may take you a longer time to learn how to fit lens models with a pixelization successfully!

__Positive Only Solver__

All pixelized source reconstructions use a positive-only solver, meaning that every source-pixel is only allowed
to reconstruct positive flux values. This ensures that the source reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real source galaxy (a common systematic solution in lens
analysis).

It may be surprising to hear that this is a feature worth pointing out, but it turns out setting up the linear algebra
to enforce positive reconstructions is difficult to make efficient. A lot of development time went into making this
possible, where a bespoke fast non-negative linear solver was developed to achieve this.

Other methods in the literature often do not use a positive only solver, and therefore suffer from these
unphysical solutions, which can degrade the results of lens model in general.

__Chaining__

Due to the complexity of fitting with a pixelization, it is often best to use **PyAutoLens**'s non-linear chaining
feature to compose a pipeline which begins by fitting a simpler model using a parametric source.

More information on chaining is provided in the `autolens_workspace/notebooks/imaging/advanced/chaining` folder,
chapter 3 of the **HowToLens** lectures.

The script `autolens_workspace/scripts/imaging/advanced/chaining/parametric_to_pixelization.py` explitly uses chaining
to link a lens model using a light profile source to one which then uses a pixelization.

__Model__

This script fits `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Overlay` image-mesh, `Delaunay` mesh and `Constant` regularization.

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
import numpy as np

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800),
    pixel_scales=0.05,
    radius=3.0,
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
    transformer_class=al.TransformerDFT,
)

"""
__Run Times (Inversion Settings)__

The run times of an interferometer pixelization reconstruction (called an `Inversion`) depend significantly 
on how the reconstruction is performed, specifically the transformer used and way the linear algebra is performed.

The transformer maps the inversion's image from real-space to Fourier space, with two options available that have
optimal run-times depending on the number of visibilities in the dataset:

- `TransformerDFT`: A discrete Fourier transform which is most efficient for < ~100 000 visibilities.

- `TransformerNUFFT`: A non-uniform fast Fourier transform which is most efficient for > ~100 000 visibilities.

This dataset fitted in this example has just ~200 visibilities, so we will input the 
setting `transformer_cls=TransformerDFT`.

The linear algebra describes how the linear system of equations used to reconstruct a source via a pixelization is
solved. 

There are with three options available that again have run-times that are optimal for datasets of different sizes 
(do not worry if you do not understand how the linear algebra works, all you need to do is ensure you choose the
setting most appropriate for the size of your dataset):

- `use_w_tilde`: If `False`, the matrices in the linear system are computed via a `mapping_matrix`, which is optimal
  for datasets with < ~10 000 visibilities.

- `use_w_tilde`: If `True`, the matrices are computed via a `w_tilde` matrix instead, which is optimal for datasets 
  with between ~10 000 and 1 000 000 visibilities.

- `use_linear_operators`: A different formalism is used entirely where matrices are not computed and linear operators
   are used instead. This is optimal for datasets with > ~1 000 000 visibilities.

The dataset fitted in this example has ~200 visibilities, so we will input the settings `use_w_tilde=False` and
`use_linear_operators=False`.

The script `autolens_workspace/*/interferometer/data_preparation/examples/run_times.py` compares the run-time of an inversion for your 
interferometer dataset for different settings. 

I recommend you use this script to choose the optimal settings for your dataset, as the difference in run-time can be
huge!
"""
settings_inversion = al.SettingsInversion(use_linear_operators=False, use_w_tilde=False)

"""
We now plot the `Interferometer` object which is used to fit the lens model.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source-galaxy's light uses a `Rectangular` mesh with fixed resolution 30 x 30 pixels (0 parameters).
 
 - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8. 
 
It is worth noting the `Pixelization`  use significantly fewer parameters (1 parameter) than 
fitting the source using `LightProfile`'s (7+ parameters). 
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

pixelization = af.Model(
    al.Pixelization,
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
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

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("interferometer", "modeling"),
    name="mass[sie]_source[pix]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=4,
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
reconstructed as a demagnified version of the lensed source. 

To prevent these solutions biasing the model-fit we specify a `position_threshold` of 0.5", which requires that a 
mass model traces the four (y,x) coordinates specified by our positions (that correspond to the brightest regions of the 
lensed source) within 0.5" of one another in the source-plane. If this criteria is not met, a large penalty term is
added to likelihood that massively reduces the overall likelihood. This penalty is larger if the ``positions``
trace further from one another.

This ensures the unphysical solutions that bias a pixelization have a lower likelihood that the physical solutions
we desire. Furthermore, the penalty term reduces as the image-plane multiple image positions trace closer in the 
source-plane, ensuring Nautilus converges towards an accurate mass model. It does this very fast, as 
ray-tracing just a few multiple image positions is computationally cheap. 

The threshold of 0.3" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. The high threshold ensures only the initial mass models at the start of the fit are resampled.

Position thresholding is described in more detail in the 
script `autolens_workspace/*/modeling/imaging/customize/positions.py`
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
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
__Run Time__

The discussion above described how the run-times of a pixelization using an interferometer dataset are significantly
depending on the number of visibilities in the dataset. The discussion below is a more generic description of how
the run-time of a pixelization scales, which applies to other datasets (e.g. imaging) as well.

The log likelihood evaluation time given below is relatively fast (), because we above chose a suitable transformer
and method to solve the linear equations for the number of visibilities in the dataset.

The run time of a pixelization is longer than many other features, with the estimate below coming out at around ~0.5 
seconds per likelihood evaluation. This is because the fit has a lot of linear algebra to perform in order to
reconstruct the source on the pixel-grid.

Nevertheless, this is still fast enough for most use-cases. If run-time is an issue, the following factors determine
the run-time of a a pixelization and can be changed to speed it up (at the expense of accuracy):

 - The number of unmasked pixels in the image data. By making the mask smaller (e.g. using an annular mask), the 
   run-time will decrease.

 - The number of source pixels in the pixelization. By reducing the `shape` from (30, 30) the run-time will decrease.

This also serves to highlight why the positions threshold likelihood is so powerful. The likelihood evaluation time
of this step is below 0.001 seconds, meaning that the initial parameter space sampling is extremely efficient even
for a pixelization (this is not accounted for in the run-time estimate below)!
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

The search returns a result object, which whose `info` attribute shows the result in a readable format:
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer,
    grid=real_space_mask.derive_grid.unmasked,
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""

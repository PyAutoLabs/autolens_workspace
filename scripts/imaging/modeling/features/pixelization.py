"""
Modeling Features: Pixelization
===============================

A pixelization reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
the solution to have a degree of smoothness.

This script fits a source galaxy model which uses a pixelization to reconstruct the source's light. A Delaunay
mesh and constant regularization scheme are used, which are the simplest forms of pixelization and regularization
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

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is reconstructed using a `DelaunayMagnification` mesh and `Constant`
   regularization scheme.

__Start Here Notebook__

If any code in this script is unclear, refer to the modeling `start_here.ipynb` notebook for more detailed comments.
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

We define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Positions__

This fit also uses the arc-second positions of the multiply imaged lensed source galaxy, which were drawn onto the
image via the GUI described in the file `autolens_workspace/*/imaging/data_preparation/gui/positions.py`.
"""
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data.  In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source-galaxy's light uses a `DelaunayMagnification` mesh with fixed resolution 30 x 30 pixels (0 parameters).
 
 - This pixelization is regularized using a `ConstantSplit` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6. 
 
It is worth noting the `Pixelization`  use significantly fewer parameters (1 parameter) than 
fitting the source using `LightProfile`'s (7+ parameters). 

The lens model therefore includes a mesh and regularization scheme, which are used together to create the 
pixelization. 

__Model Cookbook__

A full description of model composition, including lens model customization, is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html
"""
# Lens:

mass = af.Model(al.mp.PowerLaw)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

mesh = af.Model(al.mesh.DelaunayMagnification)
mesh.shape = (30, 30)

regularization = af.Model(al.reg.ConstantSplit)

pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the source galaxy's has a mesh and regularization scheme, which are combined into a pixelization.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Dynesty (see `start.here.py` for a 
full description).
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="pixelization",
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

Unlike other example scripts, we also pass the `AnalysisImaging` object below a `PositionsLHPenalty` object, which
includes the positions we loaded above, alongside a `threshold`.

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
positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=0.3)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Dynesty the model is fitted to the data. 
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood=positions_likelihood,
)

"""
__Run Time__

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

The search returns a result object, which whose `info` attribute shows the result in a readable format (if this 
does not display clearly on your screen refer to `start_here.ipynb` for a description of how to fix this):

This confirms that the source galaxy's has a mesh and regularization scheme, which are combined into a pixelization.
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via dynesty.

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
__Voronoi__

The pixelization mesh which tests have revealed performs best is the `VoronoiNN` object, which uses a Voronoi
mesh with a technique called natural neighbour interpolation (full details are provided in the **HowToLens**
tutorials).

I recommend users always use these pixelizations, however they require a c library to be installed, thus they are
not the default pixelization used in this tutorial.

If you want to use this pixelization, checkout the installation instructions here:

https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/util/nn

__Wrap Up__

Pixelizations are the most complex but also most powerful way to model a source galaxy.

Whether you need to use them or not depends on the science you are doing. If you are only interested in measuring a
simple quantity like the Einstein radius of a lens, you can get away with using light profiles like a Sersic, MGE or 
shapelets to model the source. Low resolution data also means that using a pixelization is not necessary, as the
complex structure of the source galaxy is not resolved anyway.

However, fitting complex mass models (e.g. a power-law, stellar / dark model or dark matter substructure) requires 
this level of complexity in the source model. Furthermore, if you are interested in studying the properties of the
source itself, you won't find a better way to do this than using a pixelization.

Having read this script, you are probably now in a position to fit your own lens model using a pixelization. If you are
able to do this, and get a fit that is sufficiently good, you are done.

__Chaining__

If your pixelization fit does not go well, or you want for faster computational run-times, you may wish to use
search chaining which breaks the model-fit into multiple dynesty runs. This is described for the specific case of 
linking a (computationally fast) light profile fit to a pixelization in the script:

`autolens_workspace/scripts/imaging/advanced/chaining/parametric_to_pixelization.py`

__HowToLens__

You probably also don't have a good understanding of how pixelizations actually work, which comes down to a lot of
linear algebra, Bayesian statistics and geometry. If you want to understand this, you should checkout chapter 4 of
**HowToLens**.
"""

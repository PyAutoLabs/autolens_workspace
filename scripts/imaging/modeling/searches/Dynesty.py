"""
Searches: Nautilus
=================

Nautilus (https://github.com/joshspeagle/Nautilus) is a nested sampling algorithm.

A nested sampling algorithm estimates the Bayesian evidence of a model as well as the posterior.

Dynesty used to be the main model-fitting algorithm used by PyAutoLens. However, we now recommend the nested sampling
algorithm `Nautilus` instead, which is faster and more accurate than Dynesty. We include this tutorial for Dynesty
for those who are interested in comparing the two algorithms.

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
__Dataset + Masking__

Load and plot the strong lens dataset `simple__no_lens_light` via .fits files.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model__ 

Set up the model, which follows the same API as the `start_here.ipynb` tutorial.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__ 

We create the Analysis as per using.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Search__

Below we use dynesty to fit the lens model, using the model with start points as described above. See the Dynesty docs
for a description of what the input parameters below do.

There are two important inputs worth noting:

- `sample="rwalk"`: Makes dynesty use random walk nested sampling, which proved to be effective at lens modeling.
- `walks-10`: Only 10 random walks are performed per sample, which is efficient for lens modeling.

"""
search = af.DynestyStatic(
    path_prefix=path.join("searches"),
    name="DynestyStatic",
    nlive=50,
    sample="rwalk",
    walks=10,
    bound="multi",
    bootstrap=None,
    enlarge=None,
    update_interval=None,
    facc=0.5,
    slices=5,
    fmove=0.9,
    max_move=100,
    iterations_per_update=2500,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can use an `MCMCPlotter` to create a corner plot, which shows the probability density function (PDF) of every
parameter in 1D and 2D.
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
Finish.
"""

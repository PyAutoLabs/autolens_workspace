"""
Plots: ZeusPlotter
==================

This example illustrates how to plot visualization summarizing the results of a zeus non-linear search using
a `ZeusPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
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
First, lets create a result via zeus by repeating the simple model-fit that is performed in 
the `modeling/start_here.py` example.

We use a model with an initialized starting point, which is necessary for lens modeling with zeus.
"""
dataset_name = "simple__no_lens_light"

search = af.Zeus(
    path_prefix=path.join("plot", "ZeusPlotter"),
    name="ZeusPlotter",
    nwalkers=30,
    nsteps=500,
)

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

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=2.0)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge = af.Model(al.lp.Sersic)
bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
bulge.intensity = af.UniformPrior(lower_limit=0.1, upper_limit=0.5)
bulge.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.4)
bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=2.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
We now pass the samples to a `ZeusPlotter` which will allow us to use Nautilus's in-built plotting libraries to 
make figures.

The zeus readthedocs describes fully all of the methods used below 

 - https://zeus-mcmc.readthedocs.io/en/latest/api/plotting.html#cornerplot
 - https://zeus-mcmc.readthedocs.io/en/latest/notebooks/normal_distribution.html
 
 The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
zeus_plotter = aplt.ZeusPlotter(samples=result.samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
zeus_plotter.corner(
    weight_list=None,
    levels=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    truth=None,
    color=None,
    alpha=0.5,
    linewidth=1.5,
    fill=True,
    fontsize=10,
    show_titles=True,
    title_fmt=".2f",
    title_fontsize=12,
    cut=3,
    fig=None,
    size=(10, 10),
)


"""
The `trajectories` method shows the likelihood of every parameter as a function of parameter value, colored by every
individual walker.
"""
zeus_plotter.trajectories()

"""
The `likelihood_series` method shows the likelihood as a function of step number, colored by every individual walker.
"""
zeus_plotter.likelihood_series()

"""
The `time_series` method shows the likelihood of every parameter as a function of step number, colored by every
individual walker.
"""
zeus_plotter.time_series()

"""
Finish.
"""

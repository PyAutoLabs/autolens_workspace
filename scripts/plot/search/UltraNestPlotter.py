"""
Plots: UltraNestPlotter
=======================

This example illustrates how to plot visualization summarizing the results of a ultranest non-linear search using
a `UltraNestPlotter`.

Installation
------------

Because UltraNest is an optional library, you will likely have to install it manually via the command:

`pip install ultranest`

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
First, lets create a result via ultranest by repeating the simple model-fit that is performed in 
the `modeling/start_here.py` example.
"""
dataset_name = "simple__no_lens_light"

search = af.UltraNest(path_prefix="plot", name="UltraNestPlotter")

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

# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Notation__

Plot are labeled with short hand parameter names (e.g. the `centre` parameters are plotted using an `x`). 

The mappings of every parameter to its shorthand symbol for plots is specified in the `config/notation.yaml` file 
and can be customized.

Each label also has a superscript corresponding to the model component the parameter originates from. For example,
Gaussians are given the superscript `g`. This can also be customized in the `config/notation.yaml` file.

__Plotting__

We now pass the samples to a `UltraNestPlotter` which will allow us to use ultranest's in-built plotting libraries to 
make figures.

The ultranest readthedocs describes fully all of the methods used below 

 - https://johannesbuchner.github.io/UltraNest/readme.html
 - https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.plot
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
plotter = aplt.NestPlotter(samples=result.samples)

"""
The `corner_anesthetic` method produces a triangle of 1D and 2D PDF's of every parameter using the library `anesthetic`.
"""
plotter.corner_anesthetic()

"""
__Search Specific Visualization__

The internal sampler can be used to plot the results of the non-linear search. 

We do this using the `search_internal` attribute which contains the sampler in its native form.

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
__Plots__

UltraNest example plots are not shown explicitly below, so checkout their docs for examples!
"""

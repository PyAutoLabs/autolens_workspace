"""
Pipelines: Mass Total + Source Inversion
========================================

By chaining together two searches this script fits `Interferometer` dataset of a 'galaxy-scale' strong lens, where in the
final model:
.
 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy is an `Inversion`.

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

import numpy as np
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__ 

Load the `Interferometer` data, define the visibility and real-space masks and plot them.

This includes the method used to Fourier transform the real-space image of the strong lens to the uv-plane and compare 
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities.
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
    transformer_class=al.TransformerDFT,
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
settings_inversion = al.SettingsInversion(use_linear_operators=False)

"""
We now plot the `Interferometer` object which is used to fit the lens model.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("interferometer", "pipelines")

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `Sersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
model_1 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            mass=al.mp.Isothermal,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=redshift_source, bulge=al.lp.Sersic),
    ),
)

search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]_mass[sie]_source[lp]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisInterferometer(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [Parameters fixed to 
 results of search 1].
 
 - The source galaxy's light uses an `Overlay` image-mesh [2 parameters].
 
 - The source-galaxy's light uses a `Delaunay` mesh [0 parameters].
 
 - This pixelization is regularized using a `ConstantSplit` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.
"""
image_mesh = af.Model(al.image_mesh.Overlay)
image_mesh.shape = (30, 30)

pixelization = af.Model(
    al.Pixelization,
    image_mesh=image_mesh,
    mesh=al.mesh.Delaunay,
    regularization=al.reg.ConstantSplit,
)

model_2 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            mass=result_1.instance.galaxies.lens.mass,
            shear=result_1.instance.galaxies.lens.shear,
        ),
        source=af.Model(al.Galaxy, redshift=redshift_source, pixelization=pixelization),
    ),
)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]_mass[sie]_source[pix_init]",
    unique_tag=dataset_name,
    n_live=50,
)

analysis_2 = al.AnalysisInterferometer(
    dataset=dataset, settings_inversion=settings_inversion
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Model + Search (Search 3)__

We use the results of searches 1 and 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters: priors 
 initialized from search 1].
 
 - The source-galaxy's light uses an `Overlay` image-mesh [parameters fixed to results of search 2].

 - The source-galaxy's light uses a `Delaunay` mesh [parameters fixed to results of search 2].
 
 - This pixelization is regularized using a `ConstantSplit` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.
"""
model_3 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            mass=result_1.model.galaxies.lens.mass,
            shear=result_1.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=result_2.instance.galaxies.source.pixelization,
        ),
    ),
)

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]_mass[sie]_source[pix]",
    unique_tag=dataset_name,
    n_live=100,
)

"""
__Positions + Analysis + Model-Fit (Search 3)__

We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis_3 = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood=result_2.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=settings_inversion,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

We use the results of searches 2 and 4 to create the lens model fitted in search 4, where:

 - The lens galaxy's total mass distribution is an `PowerLaw` and `ExternalShear` [8 parameters: priors 
 initialized from search 3].
 
 - The source-galaxy's light uses an `Overlay` image-mesh [parameters fixed to results of search 2].

 - The source-galaxy's light uses a `Delaunay` mesh [parameters fixed to results of search 2].
 
 - This pixelization is regularized using a `ConstantSplit` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.
"""
mass = af.Model(al.mp.PowerLaw)
mass.take_attributes(result_3.model.galaxies.lens.mass)

model_4 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            mass=mass,
            shear=result_3.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=result_2.instance.galaxies.source.pixelization,
        ),
    ),
)

search_4 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[4]_mass[total]_source[pix]",
    unique_tag=dataset_name,
    n_live=150,
)

analysis_4 = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood=result_3.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=settings_inversion,
)

result_4 = search_4.fit(model=model_4, analysis=analysis_4)

"""
Finish.
"""

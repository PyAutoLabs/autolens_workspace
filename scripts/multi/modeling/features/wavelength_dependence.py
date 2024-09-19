"""
Modeling: Mass Total + Source Parametric
========================================

This script fits a multi-wavelength `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a linear parametric `Sersic` bulge where the `effective_radius` varies across wavelength.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric `SersicCore`.

Three images are fitted, corresponding to a green ('g' band), red (`r` band) and near infrared ('I' band) images.

This script assumes previous knowledge of the `multi` modeling API found in other scripts in the `multi/modeling`
package. If anything is unclear check those scripts out.

__Effective Radius vs Wavelength__

Unlike other `multi` modeling scripts, the effective radius of the lens and source galaxies as a user defined function of
wavelength, for example following a relation `y = (m * x) + c` -> `effective_radius = (m * wavelength) + c`.

By using a linear relation `y = mx + c` the free parameters are `m` and `c`, which does not scale with the number
of datasets. For datasets with multi-wavelength images (e.g. 5 or more) this allows us to parameterize the variation
of parameters across the datasets in a way that does not lead to a very complex parameter space.

For example, in other scripts, a free `effective_radius` is created for every datasets, which would add 5+ free parameters
to the model for 5+ datasets.


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
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for load each dataset.
"""
color_list = ["g", "r"]  # , "I"]

"""
__Wavelengths__

The effective_radius of each source galaxy is parameterized as a function of wavelength.

Therefore we define a list of wavelengths of each color above.
"""
wavelength_list = [464, 658, 806]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12, 0.012]

"""
__Dataset__

Load and plot each multi-wavelength strong lens dataset, using a list of their waveband colors.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "wavelength_dependence"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset_list = [
    al.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.

For multi-wavelength lens modeling, we use the same mask for every dataset whenever possible. This is not
absolutely necessary, but provides a more reliable analysis.
"""
mask_list = [
    al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )
    for dataset in dataset_list
]

dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a linear parametric `SersicCore` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp_linear.Sersic,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Model + Analysis__

We now make the lens and source `effective_radius` a free parameter across every analysis object.

Unlike other scripts, where the `effective_radius` for every dataset is created as a free parameter, we will assume that 
the `effective_radius` of the lens and source galaxies linearly varies as a function of wavelength, and therefore compute 
the `effective_radius` value for each color image using a linear relation `y = mx + c`.

The function below is not used to compose the model, but illustrates how the `effective_radius` values were computed
in the corresponding `wavelength_dependence` simulator script.
"""


def lens_effective_radius_from(wavelength):
    m = 1.0 / 100.0  # lens appears brighter with wavelength
    c = 3

    return m * wavelength + c


def source_effective_radius_from(wavelength):
    m = -(1.2 / 100.0)  # source appears fainter with wavelength
    c = 10

    return m * wavelength + c


"""
To parameterize the above relation as a model, we compose `m` and `c` as priors and use PyAutoFit's prior arithmatic
to compose a model as a linear relation.
"""
lens_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

source_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
source_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

"""
The free parameters of our model there are no longer `effective_radius` values, but the parameters `m` and `c` in the relation
above. 

The model complexity therefore does not increase as we add more parameters to the model.

__Analysis__

We create an `Analysis` object for every dataset and sum it to combine the analysis of all images.
"""

analysis_list = []

for wavelength, dataset in zip(wavelength_list, dataset_list):
    lens_effective_radius = (wavelength * lens_m) + lens_c
    source_effective_radius = (wavelength * source_m) + source_c

    # Currently buggy, need to fix

    # analysis_list.append(
    #     al.AnalysisImaging(dataset=dataset).with_model(
    #         model.replacing(
    #             {
    #                 model.galaxies.lens.bulge.effective_radius: lens_effective_radius,
    #                 model.galaxies.source.bulge.effective_radius: source_effective_radius,
    #             }
    #         )
    #     )
    # )

    analysis_list.append(al.AnalysisImaging(dataset=dataset))

analysis = sum(analysis_list)
analysis.n_cores = 1

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).
"""
search = af.Nautilus(
    path_prefix=path.join("multi", "modeling"),
    name="wavelength_dependence",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a combined analysis.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.

For example, close inspection of the `max_log_likelihood_instance` of the two results shows that all parameters,
except the `effective_radius` of the source galaxy's `bulge`, are identical.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
Plotting each result's tracer shows that the source appears different, owning to its different intensities.
"""
for result in result_list:
    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
    )
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=15). 

Therefore, the samples is identical in every result object.
"""
for result in result_list:
    plotter = aplt.NestPlotter(samples=result.samples)
    plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""

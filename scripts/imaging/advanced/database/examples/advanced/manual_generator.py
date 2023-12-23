"""
Database Optional: Manual
=========================

The main tutorials use the built-in PyAutoLens aggregator objects (e.g. `TracerAgg`) to navigate the database. For the
majority of use-cases this should be sufficient, however a user may have a use case where a more customized
generation of a `Tracer` or `FitImaging` object is desired.

This optional tutorials shows how one can achieve this, by creating lists and writing your own generator funtions
to make these objects.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Database File__

First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator.from_database("database.sqlite")

"""
__Manual Tracers via Lists (Optional)__

I now illustrate how one can create tracers via lists. This does not offer any new functionality that the `TracerAgg`
object above does not provide, and is here for illustrative purposes. It is therefore optional.

Lets create a list of instances of the maximum log likelihood models of each fit.
"""
ml_instances = [samps.max_log_likelihood() for samps in agg.values("samples")]

"""
A model instance contains a list of `Galaxy` instances, which is what we are using to passing to functions in 
PyAutoLens. 

Lets create the maximum log likelihood tracer of every fit.
"""
ml_tracers = [
    al.Tracer.from_galaxies(galaxies=instance.galaxies) for instance in ml_instances
]

print("Maximum Log Likelihood Tracers: \n")
print(ml_tracers, "\n")
print("Total Tracers = ", len(ml_tracers))

"""
Now lets plot their convergences, using a grid of 100 x 100 pixels (noting that this isn't` necessarily the grid used
to fit the data in the search itself).
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer in ml_tracers:
    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(convergence=True)


"""
__Manual Tracer via Generators (Optional / Advanced)__

I now illustrate how one can create tracers via generators. There may be occasions where the functionality of 
the `TracerAgg` object is insufficient to perform the calculation you require. You can therefore write your own 
generator to do this.

This section is optional, and I advise you only follow it if the `TracerAgg` object is sufficient for your use-case.
"""


def make_tracer_generator(fit):
    samples = fit.value(name="samples")

    return al.Tracer.from_galaxies(galaxies=samples.max_log_likelihood().galaxies)


"""
We `map` the function above using our aggregator to create a tracer generator.
"""
tracer_gen = agg.map(func=make_tracer_generator)

"""
We can now iterate over our tracer generator to make the plots we desire.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer in tracer_gen:
    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(convergence=True, potential=True)


"""
Now lets use a generator to print the Einstein Mass of every tracer.
"""


def print_max_log_likelihood_mass(fit):
    samples = fit.value(name="samples")

    instance = samples.max_log_likelihood()

    tracer = al.Tracer.from_galaxies(galaxies=instance.galaxies)

    einstein_mass = tracer.galaxies[0].einstein_mass_angular_from(grid=grid)

    print("Einstein Mass (angular units) = ", einstein_mass)

    cosmology = al.cosmo.Planck15()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=fit.instance.galaxies.lens.redshift,
            redshift_1=fit.instance.galaxies.source.redshift,
        )
    )

    einstein_mass_kpc = einstein_mass * critical_surface_density

    print("Einstein Mass (kpc) = ", einstein_mass_kpc)
    print("Einstein Mass (kpc) = ", "{:.4e}".format(einstein_mass_kpc))


print()
print("Maximum Log Likelihood Lens Einstein Masses:")
agg.map(func=print_max_log_likelihood_mass)


"""
__Manual Dataset via List (Optional)__

I now illustrate how one can create fits via lists. This does not offer any new functionality that the `FitImagingAgg`
object above does not provide, and is here for illustrative purposes. It is therefore optional.

Lets create a list of the imaging dataset of every lens our search fitted. 

The individual masked `data`, `noise_map` and `psf` are stored in the database, as opposed to the `Imaging` object, 
which saves of hard-disk space used. Thus, we need to create the `Imaging` object ourselves to inspect it. 

They are stored as .fits HDU objects, which can be converted to `Array2D` and `Kernel2D` objects via the
`from_primary_hdu` method.
"""
data_gen = agg.values(name="dataset.data")
noise_map_gen = agg.values(name="dataset.noise_map")
psf_gen = agg.values(name="dataset.psf")
settings_dataset_gen = agg.values(name="dataset.settings")

for data, noise_map, psf, settings_dataset in zip(
    data_gen, noise_map_gen, psf_gen, settings_dataset_gen
):
    data = al.Array2D.from_primary_hdu(primary_hdu=data)
    noise_map = al.Array2D.from_primary_hdu(primary_hdu=noise_map)
    psf = al.Kernel2D.from_primary_hdu(primary_hdu=psf)

    dataset = al.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_dataset,
        pad_for_convolver=True,
        check_noise_map=False,
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Manual Fit via Generators (Optional / Advanced)__

I now illustrate how one can create fits via generators. There may be occasions where the functionality of 
the `FitImagingAgg` object is insufficient to perform the calculation you require. You can therefore write your own 
generator to do this.

This section is optional, and I advise you only follow it if the `FitImagingAgg` object is sufficient for your use-case.
"""


def make_imaging_gen(fit):
    data = al.Array2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.data"))
    noise_map = al.Array2D.from_primary_hdu(
        primary_hdu=fit.value(name="dataset.noise_map")
    )
    psf = al.Kernel2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.psf"))
    settings_dataset = fit.value(name="dataset.settings")

    dataset = al.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_dataset,
        pad_for_convolver=True,
        check_noise_map=False,
    )

    return dataset


imaging_gen = agg.map(func=make_imaging_gen)

for dataset in imaging_gen:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()


"""
We now have access to the `Imaging` data we used to perform a model-fit, and the results of that model-fit in the form
of a `Samples` object. 

We can therefore use the database to create a `FitImaging` of the maximum log-likelihood model of every model to its
corresponding dataset, via the following generator:
"""


def make_fit_imaging_generator(fit):
    dataset = make_imaging_gen(fit=fit)

    tracer = al.Tracer.from_galaxies(galaxies=fit.instance.galaxies)

    return al.FitImaging(dataset=dataset, tracer=tracer)


fit_imaging_gen = agg.map(func=make_fit_imaging_generator)

for fit in fit_imaging_gen:
    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()

"""
The `AnalysisImaging` object has a `settings_inversion` attribute, which customizes how the inversion fits the 
data. The generator above uses the `settings` of the object that were used by the model-fit. 

These settings objected are contained in the database and can therefore also be passed to the `FitImaging`.
"""


def make_fit_imaging_generator(fit):
    dataset = make_imaging_gen(fit=fit)

    settings_inversion = fit.value(name="settings_inversion")

    tracer = al.Tracer.from_galaxies(galaxies=fit.instance.galaxies)

    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings_inversion=settings_inversion,
    )


fit_imaging_gen = agg.map(func=make_fit_imaging_generator)

for fit in fit_imaging_gen:
    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()


"""
__Errors: Axis Ratio__

To begin, lets compute the axis ratio of a model, including the errors on the axis ratio. In the previous tutorials, 
we saw that the errors on a quantity like the ell_comps is simple, because it was sampled by the non-linear 
search. Thus, to get their we can uses the Samples object to simply marginalize over all over parameters via the 1D 
Probability Density Function (PDF).

But what if we want the errors on the axis-ratio? This wasn`t a free parameter in our model so we can`t just 
marginalize over all other parameters.

Instead, we need to compute the axis-ratio of every model sampled by the non-linear search and from this determine 
the PDF of the axis-ratio. When combining the different axis-ratios we weight each value by its `weight`. For Nautilus,
the nested sampler we fitted our aggregator sample with, this down weight_list the model which gave lower likelihood 
fits. For other non-linear search methods (e.g. MCMC) the weight_list can take on a different meaning but can still be 
used for combining different model results.

Below, we get an instance of every Nautilus sample using the `Samples`, compute that models axis-ratio, store them in a 
list and find the value via the PDF and quantile method.

Now, we iterate over each Samples object, using every model instance to compute its axis-ratio. We combine these 
axis-ratios with the samples weight_list to give us the weighted mean axis-ratio and error.

To do this, we again use a generator. Whislt the axis-ratio is a fairly light-weight value, and this could be
performed using a list without crippling your comptuer`s memory, for other quantities this is not the case. Thus, for
computing derived quantities it is good practise to always use a generator.

[Commented out but should work fine if you uncomment it]
"""

# from autofit.non_linear.samples.pdf import quantile
# import math
#
# sigma = 3.0
#
# low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2
#
#
# def axis_ratio_error_from_agg_obj(fit):
#     samples = fit.value(name="samples")
#
#     axis_ratios = []
#     weight_list = []
#
#     for sample_index in range(samples.total_samples):
#         weight = samples.sample_list[sample_index].weight
#
#         if weight > 1e-4:
#
#             instance = samples.from_sample_index(sample_index=sample_index)
#
#             axis_ratio = al.convert.axis_ratio_from(
#                 ell_comps=instance.galaxies.lens.mass.ell_comps
#             )
#
#             axis_ratios.append(axis_ratio)
#             weight_list.append(weight)
#
#     median_axis_ratio = quantile(x=axis_ratios, q=0.5, weights=weight_list)[0]
#
#     lower_axis_ratio = quantile(x=axis_ratios, q=low_limit, weights=weight_list)
#
#     upper_axis_ratio = quantile(x=axis_ratios, q=1 - low_limit, weights=weight_list)
#
#     return median_axis_ratio, lower_axis_ratio, upper_axis_ratio
#
#
# axis_ratio_values = list(agg.map(func=axis_ratio_error_from_agg_obj))
# median_axis_ratio_list = [value[0] for value in axis_ratio_values]
# lower_axis_ratio_list = [value[1] for value in axis_ratio_values]
# upper_axis_ratio_list = [value[2] for value in axis_ratio_values]
#
# print("Axis Ratios:")
# print(median_axis_ratio_list)
#
# print("Axis Ratio Errors:")
# print(lower_axis_ratio_list)
# print(upper_axis_ratio_list)

"""
Fin.
"""

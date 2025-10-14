"""
Results: CSV
============

This example is a results workflow example, which means it provides tool to set up an effective workflow inspecting
and interpreting the large libraries of modeling results.

In this tutorial, we use the aggregator to load the results of model-fits and output them in a single .csv file.

This enables the results of many model-fits to be concisely summarised and inspected in a single table, which
can also be easily passed on to other collaborators.

__CSV, Png and Fits__

Workflow functionality closely mirrors the `png_make.py` and `fits_make.py`  examples, which load results of
model-fits and output th em as .png files and .fits files to quickly summarise results.

The same initial fit creating results in a folder called `results_folder_csv_png_fits` is therefore used.

__Interferometer__

This script can easily be adapted to analyse the results of charge injection imaging model-fits.

The only entries that needs changing are:

 - `ImagingAgg` -> `InterferometerAgg`.
 - `FitImagingAgg` -> `FitInterferometerAgg`.
 - `ImagingPlotter` -> `InterferometerPlotter`.
 - `FitImagingPlotter` -> `FitInterferometerPlotter`.

Quantities specific to an interfometer, for example its uv-wavelengths real space mask, are accessed using the same API
(e.g. `values("dataset.uv_wavelengths")` and `.values{"dataset.real_space_mask")).

__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path

import autofit as af
import autolens as al

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!

__Unique Tag__

One thing to note is that the `unique_tag` of the search is given the name of the dataset with an index for the
fit of 0 and 1. 

This `unique_tag` names the fit in a descriptive and human-readable way, which we will exploit to make our .csv files
more descriptive and easier to interpret.
"""
for i in range(2):
    dataset_name = "simple__no_lens_light"
    dataset_path = Path("dataset") / "imaging" / dataset_name

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=0.1,
    )

    mask_radius = 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                mass=al.mp.Isothermal,
                shear=al.mp.ExternalShear,
            ),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=None),
        ),
    )

    search = af.Nautilus(
        path_prefix=Path("results_folder_csv_png_fits"),
        name="results",
        unique_tag=f"simple__no_lens_light_{i}",
        n_live=100,
        iterations_per_update=10000,
    )

    class AnalysisLatent(al.AnalysisImaging):

        LATENT_KEYS = [
            "galaxies.lens.shear.magnitude",
            "galaxies.lens.shear.angle",
        ]

        def compute_latent_variables(self, parameters, model):
            """
            A latent variable is not a model parameter but can be derived from the model. Its value and errors may be
            of interest and aid in the interpretation of a model-fit.

            This code implements a simple example of a latent variable, the magn

            By overwriting this method we can manually specify latent variables that are calculated and output to
            a `latent.csv` file, which mirrors the `samples.csv` file.

            In the example below, the `latent.csv` file will contain at least two columns with the shear magnitude and
            angle sampled by the non-linear search.

            This function is called for every non-linear search sample, where the `instance` passed in corresponds to
            each sample.

            You can add your own custom latent variables here, if you have particular quantities that you
            would like to output to the `latent.csv` file.

            Parameters
            ----------
            parameters : array-like
                The parameter vector of the model sample. This will typically come from the non-linear search.
                Inside this method it is mapped back to a model instance via `model.instance_from_vector`.
            model : Model
                The model object defining how the parameter vector is mapped to an instance. Passed explicitly
                so that this function can be used inside JAX transforms (`vmap`, `jit`) with `functools.partial`.

            Returns
            -------
            A dictionary mapping every latent variable name to its value.

            """
            instance = model.instance_from_vector(vector=parameters)

            if hasattr(instance.galaxies.lens, "shear"):
                magnitude, angle = al.convert.shear_magnitude_and_angle_from(
                    gamma_1=instance.galaxies.lens.shear.gamma_1,
                    gamma_2=instance.galaxies.lens.shear.gamma_2,
                )

            return (magnitude, angle)

    analysis = AnalysisLatent(dataset=dataset)

    result = search.fit(model=model, analysis=analysis)

"""
__Workflow Paths__

The workflow examples are designed to take large libraries of results and distill them down to the key information
required for your science, which are therefore placed in a single path for easy access.

The `workflow_path` specifies where these files are output, in this case the .csv files which summarise the results.
"""
workflow_path = Path("output") / "results_folder_csv_png_fits" / "workflow_make_example"

"""
__Aggregator__

Set up the aggregator as shown in `start_here.py`.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=Path("output") / "results_folder_csv_png_fits",
)

"""
Extract the `AggregateCSV` object, which has specific functions for outputting results in a CSV format.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

"""
__Adding CSV Columns_

We first make a simple .csv which contains two columns, corresponding to the inferred median PDF values for
the y centre of the mass of the lens galaxy and its einstein radius.

To do this, we use the `add_variable` method, which adds a column to the .csv file we write at the end. Every time
we call `add_variable` we add a new column to the .csv file.

Note the API for the `centre`, which is a tuple parameter and therefore needs for `centre_0` to be specified.

The `results_folder_csv_png_fits` contained two model-fits to two different datasets, meaning that each `add_variable` 
call will add three rows, corresponding to the three model-fits.

This adds the median PDF value of the parameter to the .csv file, we show how to add other values later in this script.
"""
agg_csv.add_variable(argument="galaxies.lens.mass.centre.centre_0")
agg_csv.add_variable(argument="galaxies.lens.mass.einstein_radius")

"""
__Saving the CSV__

We can now output the results of all our model-fits to the .csv file, using the `save` method.

This will output in your current working directory (e.g. the `autolens_workspace/output.results_folder_csv_png_fits`) 
as a .csv file containing the median PDF values of the parameters, have a quick look now to see the format of 
the .csv file.
"""
agg_csv.save(path=workflow_path / "csv_simple.csv")

"""
__Customizing CSV Headers__

The headers of the .csv file are by default the argument input above based on the model. 

However, we can customize these headers using the `name` input of the `add_variable` method, for example making them
shorter or more readable.

We recreate the `agg_csv` first, so that we begin adding columns to a new .csv file.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

agg_csv.add_variable(
    argument="galaxies.lens.mass.centre.centre_0",
    name="mass_centre_0",
)
agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius",
    name="mass_einstein_radius",
)

agg_csv.save(path=workflow_path / "csv_simple_custom_headers.csv")

"""
__Maximum Likelihood Values__

We can also output the maximum likelihood values of each parameter to the .csv file, using the `use_max_log_likelihood`
input.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius",
    name="mass_einstein_radius_max_lh",
    value_types=[af.ValueType.MaxLogLikelihood],
)

agg_csv.save(path=workflow_path / "csv_simple_max_likelihood.csv")

"""
__Errors__

We can also output PDF values at a given sigma confidence of each parameter to the .csv file, using 
the `af.ValueType.ValuesAt3Sigma` input and specifying the sigma confidence.

Below, we add the values at 3.0 sigma confidence to the .csv file, in order to compute the errors you would 
subtract the median value from these values. We add this after the median value, so that the overall inferred
uncertainty of the parameter is clear.

The method below adds three columns to the .csv file, corresponding to the values at the median, lower and upper sigma 
values.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius",
    name="mass_einstein_radius",
    value_types=[af.ValueType.Median, af.ValueType.ValuesAt3Sigma],
)

agg_csv.save(path=workflow_path / "csv_simple_errors.csv")

"""
__Column Label List__

We can add a list of values to the .csv file, provided the list is the same length as the number of model-fits
in the aggregator.

A useful example is adding the name of every dataset to the .csv file in a column on the left, indicating 
which dataset each row corresponds to.

To make this list, we use the `Aggregator` to loop over the `search` objects and extract their `unique_tag`'s, which 
when we fitted the model above used the dataset names. This API can also be used to extract the `name` or `path_prefix`
of the search and build an informative list for the names of the subplots.

We then pass the column `name` and this list to the `add_label_column` method, which will add a column to the .csv file.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

unique_tag_list = [search.unique_tag for search in agg.values("search")]

agg_csv.add_label_column(
    name="lens_name",
    values=unique_tag_list,
)

agg_csv.save(path=workflow_path / "csv_simple_dataset_name.csv")


"""
__Latent Variables__

Latent variables are not free model parameters but can be derived from the model, and they are described fully in
?.

This example was run with a latent variable for the shear magnitude, and below we show that this latent variable
can be added to the .csv file using the same API as above.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

agg_csv.add_variable(
    argument="galaxies.lens.shear.magnitude",
)

agg_csv.save(path=workflow_path / "csv_example_latent.csv")

"""
__Computed Columns__

We can also add columns to the .csv file that are computed from the non-linear search samples (e.g. the nested sampling
samples), for example a value derived from the median PDF instance values of the model.

To do this, we write a function which is input into the `add_computed_column` method, where this function takes the
median PDF instance as input and returns the computed value.

Below, we add a trivial example of a computed column, where the median PDF value that is twice lens Einstein radius
is computed and added to the .csv file.
"""
agg_csv = af.AggregateCSV(aggregator=agg)


def einstein_radius_x2_from(samples):
    instance = samples.median_pdf()

    return 2.0 * instance.galaxies.lens.mass.einstein_radius


agg_csv.add_computed_column(
    name="bulge_einstein_radius_x2_computed",
    compute=einstein_radius_x2_from,
)

agg_csv.save(path=workflow_path / "csv_computed_columns.csv")

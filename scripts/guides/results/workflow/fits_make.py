"""
Results: Fits Make
==================

This example is a results workflow example, which means it provides tool to set up an effective workflow inspecting
and interpreting the large libraries of modeling results.

In this tutorial, we use the aggregator to load .fits files output by a model-fit, extract hdu images and create
new .fits files, for example all to a single folder on your hard-disk.

For example, a common use case is extracting an image from `model_galaxy_images.fits` of many fits and putting them
into a single .fits file on your hard-disk. If you have modeled 100+ datasets, you can then inspect all model images
in DS9 in .fits format n a single folder, which is more efficient than clicking throughout the `output` open each
.fits file one-by-one.

The most common use of .fits splciing is where multiple observations of the same galaxy are analysed, for example
at different wavelengths, where each fit outputs a different .fits files. The model images of each fit to each
wavelength can then be packaged up into a single .fits file.

This enables the results of many model-fits to be concisely visualized and inspected, which can also be easily passed
on to other collaborators.

Internally, splicing uses standard Astorpy functions to open, edit and save .fit files.

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

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!

__Unique Tag__

One thing to note is that the `unique_tag` of the search is given the name of the dataset with an index for the
fit of 0 and 1. 

This `unique_tag` names the fit in a descriptive and human-readable way, which we will exploit to make our .fits files
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
        n_batch=50,  # GPU batching and VRAM use explained in `modeling` examples.
        iterations_per_quick_update=10000,
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

The `workflow_path` specifies where these files are output, in this case the .fits files containing the key 
results we require.
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
Extract the `AggregateFITS` object, which has specific functions for loading .fits files and outputting results in 
.fits format.
"""
agg_fits = af.AggregateFITS(aggregator=agg)

"""
__Extract Images__

We now extract 2 images from the `fit.fits` file and combine them together into a single .fits file.

We will extract the `model_image` and `residual_map` images, which are images you are used to
plotting and inspecting in the `output` folder of a model-fit and can load and inspect in DS9 from the file
`fit.fits`.

By inspecting `fit.fits` you will see it contains four images which each have a an `ext_name`: `model_image`,
`residual_map`, `normalized_residual_map`, `chi_squared_map`.

We do this by simply passing the `agg_fits.extract_fits` method the name of the fits file we load from `fits.fit`
and the `ext_name` of what we extract.

This runs on all results the `Aggregator` object has loaded from the `output` folder, meaning that for this example
where two model-fits are loaded, the `image` object contains two images.
"""
hdu_list = agg_fits.extract_fits(
    hdus=[al.agg.fits_fit.model_data, al.agg.fits_fit.residual_map],
)

"""
__Output Single Fits__

The `image` object which has been extracted is an `astropy` `Fits` object, which we use to save the .fits to the 
hard-disk.

The .fits has 4 hdus, the `model_image` and `residual_map` for the two datasets fitted.
"""
hdu_list.writeto("fits_make_single.fits", overwrite=True)

"""
__Output to Folder__

An alternative way to output the .fits files is to output them as single .fits files for each model-fit in a single 
folder, which is done using the `output_to_folder` method.

It can sometimes be easier and quicker to inspect the results of many model-fits when they are output to individual
files in a folder, as using an IDE you can click load and flick through the images. This contrasts a single .png
file you scroll through, which may be slower to load and inspect.

__Naming Convention__

We require a naming convention for the output files. In this example, we have two model-fits, therefore two .fits
files are going to be output.

One way to name the .fits files is to use the `unique_tag` of the search, which is unique to every model-fit. For
the search above, the `unique_tag` was `simple_0` and `simple_1`, therefore this will informatively name the .fits
files the names of the datasets.

We achieve this behaviour by inputting `name="unique_tag"` to the `output_to_folder` method. 
"""
agg_fits.output_to_folder(
    folder=workflow_path,
    name="unique_tag",
    hdus=[al.agg.fits_fit.model_data, al.agg.fits_fit.residual_map],
)

"""
The `name` can be any search attribute, for example the `name` of the search, the `path_prefix` of the search, etc,
if they will give informative names to the .fits files.

You can also manually input a list of names, one for each fit, if you want to name the .fits files something else.
However, the list must be the same length as the number of fits in the aggregator, and you may not be certain of the
order of fits in the aggregator and therefore will need to extract this information, for example by printing the
`unique_tag` of each search (or another attribute containing the dataset name).
"""
print([search.unique_tag for search in agg.values("search")])

agg_fits.output_to_folder(
    folder=workflow_path,
    name=["hi_0.fits", "hi_1.fits"],
    hdus=[al.agg.fits_fit.model_data, al.agg.fits_fit.residual_map],
)

"""
__CSV Files__

In the results `image` folder .csv files containing the information to visualize aspects of a result may be present.

A common example is the file `source_plane_reconstruction_0.csv`, which contains the y and x coordinates of the 
pixelization mesh, the reconstruct values and the noise map of these values.

The `AggregateFITS` object has a method `extract_csv` which extracts this table from each .csv file in the results,
returning the extracted data as a list of dictionaries. This can then be used to visualize the data, and output
it to a .fits file elsewhere.

Below, we demonstrate a common use case for a pixelization. Each .csv file is loaded, benefitting from the fact
that because it stores the irregular mesh values it is the most accurate way to store the data whilst also using
much less hard-disk space than, for example. converting it to a 2D array and .fits file. We then use the
loaded values to interpolate the data onto a regular grid and output it to .fits files in a folder.

The code below is commented out because the model does not use a pixelization, but it does work if a
pixelization is used.
"""
# reconstruction_dict_list = agg_fits.extract_csv(
#     filename="source_plane_reconstruction_0",
# )
#
# from scipy.interpolate import griddata
#
# for i, reconstruction_dict in enumerate(reconstruction_dict_list):
#
#     y = reconstruction_dict["y"]
#     x = reconstruction_dict["x"]
#     values = reconstruction_dict["reconstruction"]
#
#     points = np.stack(
#         arrays=(reconstruction_dict["x"], reconstruction_dict["y"]), axis=-1
#     )
#
#     interpolation_grid = al.Grid2D.from_extent(
#         extent=(-1.0, 1.0, -1.0, 1.0), shape_native=(201, 201)
#     )
#
#     interpolated_array = griddata(points=points, values=values, xi=interpolation_grid)
#
#     al.output_to_fits(
#         values=interpolated_array,
#         file_path=workflow_path / f"interpolated_reconstruction_{i}.fits",
#     )


"""
__Add Extra Fits__

We can also add an extra .fits image to the extracted .fits file, for example an RGB image of the dataset.

We create an image of shape (1, 2) and add the RGB image to the left of the subplot, so that the new subplot has
shape (1, 3).

When we add a single .png, we cannot extract or make it, it simply gets added to the subplot.
"""
# image_rgb = Image.open(Path(dataset_path, "rgb.png"))
#
# image = agg_fits.extract_fits(
#     al.agg.subplot_dataset.data,
#     al.agg.subplot_dataset.psf_log_10,
#     subplot_shape=(1, 2),
# )

# image = al.add_image_to_left(image, additional_img)

# image.save("png_make_with_rgb.png")

"""
__Custom Fits Files in Analysis__

Describe how a user can extend the `Analysis` class to compute custom images that are output to the .png files,
which they can then extract and make together.
"""

"""
__Path Navigation__

Example combinig `fit.fits` from `source_lp[1]` and `mass_total[0]`.
"""

"""
SLaM (Source, Light and Mass): Subhalo Source Pixelized Sensitivity Mapping
===========================================================================

This example illustrates how to perform DM subhalo sensitivity mapping using a SLaM pipeline for a dataset where the
source is modeled using a pixelization.

The sensitivity mapping simulation procedure for a pixelized source is different light profile sources. When pixelized
sources are used, the source reconstruction on the mesh is used, such that the simulations capture the irregular
morphologies of real source galaxies.

__Model__

Using a SOURCE LP PIPELINE, LIGHT LP PIPELINE, MASS TOTAL PIPELINE and SUBHALO PIPELINE this SLaM script
fits `Imaging` of a strong lens system, where in the final model:

 - The lens galaxy's light is an MGE bulge.
 - The lens galaxy's total mass distribution is an `Isothermal`.
 - A dark matter subhalo near The lens galaxy mass is included as a`NFWMCRLudlowSph`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source_lp`
 `source_pix`
 `light_lp`
 `mass_total`
 `subhalo/detection`

Check them out for a full description of the analysis!

__Start Here Notebook__

If any code in this script is unclear, refer to the `subhalo/detect/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__SOURCE LP PIPELINE__

Identical to `slam_start_here.py`, except the lens mass uses an `Isothermal` with its centre fixed to (0.0, 0.0).
"""
def source_lp(
    settings_search,
    dataset,
    mask_radius,
    redshift_lens,
    redshift_source,
):
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (0.0, 0.0)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
            ),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to `slam_start_here.py`.
"""
def source_pix_1(
    settings_search,
    dataset,
    source_lp_result,
    mesh_shape,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)
        ],
        use_jax=True,
    )

    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_lp_result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `slam_start_here.py`.
"""
def source_pix_2(
    settings_search,
    dataset,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    over_sampling = al.util.over_sample.over_sample_size_via_adapt_from(
        data=adapt_images.galaxy_name_image_dict["('galaxies', 'source')"],
        noise_map=dataset.noise_map,
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_pixelization=over_sampling,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LIGHT LP PIPELINE__

Identical to `slam_start_here.py`.
"""
def light_lp(
    settings_search,
    dataset,
    mask_radius,
    source_result_for_lens,
    source_result_for_source,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=None,
                mass=source_result_for_lens.instance.galaxies.lens.mass,
                shear=source_result_for_lens.instance.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`.
"""
def mass_total(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    light_result,
):
    # Total mass model for the lens galaxy.
    mass = af.Model(al.mp.PowerLaw)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_result_for_source.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=True,
    )

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=light_result.instance.galaxies.lens.bulge,
                disk=light_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_result_for_lens.model.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Simulate Function Class__

We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to
simulate a strong lens dataset, which include a dark matter subhalo, which is subsequently fitted.

Additional attributes required to simulate the data (mask, PSF) can be passed to the `__init__` method, and the
simulation is  performed in the `__call__` method.

When this dataset is simulated, the quantity `instance.perturb` is used in `__call__`. This is an instance
of the `NFWMCRLudlowSph`, and it is different every time the `simulate_cls` is called based on the value of sensitivity
being computed.

In this example, this `instance.perturb` corresponds to two different subhalos with values of `mass_at_200` of
1e6 MSun and 1e13 MSun.
"""


class SimulateImagingPixelized:
    def __init__(
        self,
        mask,
        psf,
        inversion,
        interpolated_pixelized_shape: Union[tuple, tuple] = (1001, 1001),
        image_plane_subgrid_size=8,
    ):
        """
        Class used to simulate the strong lens dataset used for sensitivity mapping.

        Parameters
        ----------
        mask
            The mask applied to the real image data, which is applied to every simulated dataset.
        psf
           The PSF of the real image data, which is applied to every simulated dataset and used for each fit.
        inversion
            The `Inversion` used to reconstruct the source of the real image, included the pixelized source
            reconstruction on a mesh (e.g. Delaunay / Voronoi).
        interpolated_pixelized_shape
            The pixelized source reconstruction is interpolated from an irregular mesh to a rectangular uniform array
            and grid of this shape.
        image_plane_subgrid_size
            The size of the subgrid used to create the image-plane grid, whereby multiple image pixels
            are traced to the source-plane image and evaluated to compute the flux of the simulated image.
        """
        self.mask = mask
        self.psf = psf
        self.inversion = inversion
        self.interpolated_pixelized_shape = interpolated_pixelized_shape
        self.image_plane_subgrid_size = image_plane_subgrid_size

    def __call__(self, instance: af.ModelInstance, simulate_path: str):
        """
        The `simulate_function` called by the `Sensitivity` class which simulates each strong lens image fitted
        by the sensitivity mapper.

        The simulation procedure is as follows:

        1) Extract the pixelized reconstructed source of a previous fit, which is likely on an irregular mesh
           (e.g. Delaunay / Voronoi), and interpolate the source emission onto a rectangular uniform array and grid.

        1) Use the input galaxies of the sensitivity `instance` to set up a tracer, which generates the image-plane
           image of the strong lens system.

        2) Simulate this image using the input dataset noise (Poisson) and PSF.

        3) Apply the mask used in the analysis of the real image to the simulated image.

        4) Output information about the simulation to hard-disk.

        The `subhalo` in the sensitivity `instance` changes for every iteration of the sensitivity mapping, ensuring
        that we map out the sensitivity of the analysis to the subhalo properties (centre, mass, etc.).

        Parameters
        ----------
        instance
            The sensitivity instance, which includes the galaxies whose parameters are varied to perform sensitivity.
            The subhalo in this instance changes for every iteration of the sensitivity mapping.
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.

        Returns
        -------
        A simulated image of a strong lens, which id input into the fits of the sensitivity mapper.
        """

        """
        __Resume Fit__

        If sensitivity mapping already began on this grid cell, the dataset will have been simulated already and we
        do not want to resimulate it and change its noise properties.

        We therefore load it from the `simulate_path` instead.
        """
        try:
            dataset = al.Imaging.from_fits(
                data_path=f"{simulate_path}/data.fits",
                psf_path=f"{simulate_path}/psf.fits",
                noise_map_path=f"{simulate_path}/noise_map.fits",
                pixel_scales=self.mask.pixel_scales,
                check_noise_map=False,
            )

            dataset = dataset.apply_mask(mask=self.mask)

            over_sample_size = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[8, 4, 1],
                    radial_list=[0.3, 0.6],
                    centre_list=[(0.0, 0.0)],
                )
            )

            return dataset.apply_over_sampling(
                over_sample_size_lp=over_sample_size, over_sample_size_pixelization=4
            )

        except FileNotFoundError:
            pass

        """
        __Source Galaxy Image__

        We now load the source galaxy image from the pixelized inversion of a previous fit, which could be on an irregular
        Delaunay or Voronoi mesh.

        Irregular meshes cannot be used to simulate lensed images of a source. Therefore, we interpolate the mesh to
        a uniform grid of shape `interpolated_pixelized_shape`. This should be high resolution (e.g. 1000 x 1000)
        to ensure the interpolated source array captures all structure resolved on the Delaunay / Voronoi mesh.

        Loads source array from previous reconstruction, maps to square and wraps in AutoLens Array.
        Loads lens galaxy and perturb from provided instance
        Loads source galaxy redshift and sets up a `galaxy` class object at that redshift.
        """

        mapper = self.inversion.cls_list_from(cls=al.Mapper)[0]

        mapper_valued = al.MapperValued(
            mapper=mapper,
            values=self.inversion.reconstruction_dict[mapper],
        )

        source_image = mapper_valued.interpolated_array_from(
            shape_native=self.interpolated_pixelized_shape,
            extent=(-2.0, 2.0, -2.0, 2.0),
        )

        """
        __Create Grids__

        To create the lensed image, we will ray-trace image pixels to the source-plane and interpolate them onto the
        source galaxy image.

        We therefore need the image-plane grid of (y,x) coordinates.
        """
        grid = al.Grid2D.uniform(
            shape_native=self.mask.shape_native,
            pixel_scales=self.mask.pixel_scales,
            over_sample_size=self.image_plane_subgrid_size,
        )

        """
        __Ray Tracing__

        We create a tracer, which will create the lensed grid we overlay the interpolated source galaxy image above
        in order to create the lensed source galaxy image.

        This creates the grid we will overlay the source image, in order to created the lensed source image.

        The source-plane requires a source-galaxy with a `redshift` in order for the tracer to trace it. We therefore
        make one, noting it has no light profiles because its emission is entirely defined by the source galaxy image.
        """
        tracer = al.Tracer(
            galaxies=[
                instance.galaxies.lens,
                instance.perturb,
                al.Galaxy(redshift=instance.galaxies.source.redshift),
            ]
        )

        """
        __Simulate__

        Using the tracer above, we create the image of the lensed source galaxy on the image-plane grid. This
        uses the `source_image` and therefore capture the source's irregular and asymmetric morphological features
        which the source reconstruction procedure fitted.

        Set up the grid, PSF and simulator settings used to simulate imaging of the strong lens. These should be
        tuned to match the S/N and noise properties of the observed data you are performing sensitivity mapping on.

        The `SimulatorImaging` will be passed directly the image of the strong lens we created above, which
        will be convolved with the psf before noise is added.

        To ensure the PSF convolution extends over the whole image, the image is padded before convolution to mitigate
        edge effects and trimmed after the simulation so it retains the original `shape_native`.
        """
        simulator = al.SimulatorImaging(
            exposure_time=1000.0,
            psf=self.psf,
            background_sky_level=0.1,
            add_poisson_noise_to_data=True,
            noise_seed=1,
        )

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )

        grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

        dataset = simulator.via_source_image_from(
            tracer=tracer, grid=grid, source_image=source_image
        )

        """
        __Masking__

        The data generated by the simulate function is what is ultimately fitted.

        Therefore, we also apply the mask for the analysis before we return the simulated data.
        """
        dataset = dataset.apply_mask(mask=self.mask)

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[8, 4, 1],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )

        dataset = dataset.apply_over_sampling(
            over_sample_size_lp=over_sample_size, over_sample_size_pixelization=4
        )

        """
        Outputs info about the `Tracer` to the fit, so we know exactly how we simulated the image.
        """
        tracer_no_perturb = al.Tracer(
            galaxies=[
                instance.galaxies.lens,
                al.Galaxy(redshift=instance.galaxies.source.redshift),
            ]
        )

        self.output_info(
            simulate_path=simulate_path,
            grid=grid,
            dataset=dataset,
            source_image=source_image,
            tracer=tracer,
            tracer_no_perturb=tracer_no_perturb,
        )

        return dataset

    def output_info(
        self,
        simulate_path: str,
        grid: al.Grid2D,
        dataset: al.Imaging,
        source_image: al.Array2D,
        tracer: al.Tracer,
        tracer_no_perturb: al.Tracer,
    ):
        """
        Output information about the data simulated for this iteration of sensitivity mapping.

        This information output is as follows:

        - A subplot of the simulated imaging dataset.
        - A subplot of the tracer used to simulate this imaging dataset.
        - A .json file containing the tracer galaxies.
        - Output the simulated dataset to .fits files which are used to load the data if a run is resumed.

        Parameters
        ----------
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.
        dataset
            The simulated dataset dataset which is visualized.
        tracer
            The tracer used to simulate the dataset dataset, which is visualized and output to a .json file.
        """

        aplt.subplot_imaging_dataset(dataset=dataset)

        al.output_to_json(
            obj=tracer,
            file_path=os.path.join(simulate_path, "tracer.json"),
        )

        aplt.fits_imaging(
    dataset=dataset,
            data_path=os.path.join(simulate_path, "data.fits"),
            psf_path=os.path.join(simulate_path, "psf.fits"),
            noise_map_path=os.path.join(simulate_path, "noise_map.fits"),
            overwrite=True,
        )


"""
__Base Fit__

We have defined a `Simulate` class that will be used to simulate every dataset simulated by the sensitivity mapper.
Each simulated dataset will have a unique set of parameters for the `subhalo` (e.g. due to different values of
`perturb_model`.

We will fit each simulated dataset using the `base_model`, which quantifies whether not including the dark matter
subhalo in the model changess the goodness-of-fit and therefore indicates if we are sensitive to the subhalo.

We now write a `BaseFit` class, defining how the `base_model` is fitted to each simulated dataset and the
goodness-of-fit used to quantify whether the model fits the data well. As above, the `__init__` method can be
extended with new inputs to control how the model is fitted and the `__call__` method performs the fit.

In this example, we use a full non-linear search to fit the `base_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `base_model` fit
to the simulated data.

Fucntionality which adapts the mesh and regularization of a pixelized source reconstruction to the unlensed source's
morphology require an `adapt_images`. This is an input of the __init__ constructor which is passed to the `Analysis`
for every simulated dataset.
"""


class BaseFit:
    def __init__(self, adapt_images, number_of_cores: int = 1):
        """
        Class used to fit every dataset used for sensitivity mapping with the base model (the model without the
        perturbed feature sensitivity mapping maps out).

        In this example, the base model therefore does not include the dark matter subhalo, but the simulated
        dataset includes one.

        The base fit is repeated for every parameter on the sensitivity grid and compared to the perturbed fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `AnalysisImaging` class used to fit the model
        to the dataset.

        Parameters
        ----------
        adapt_images
            The result of the previous search containing adapt images used to adapt certain pixelized source meshs's
            and regularizations to the unlensed source morphology.
        number_of_cores
            The number of cores used to perform the non-linear search. If 1, each model-fit on the grid is performed
            in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
        """
        self.adapt_images = adapt_images
        self.number_of_cores = number_of_cores

    def __call__(self, dataset, model, paths, instance):
        """
        The base fitting function which fits every dataset used for sensitivity mapping with the base model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which does not include the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.Nautilus(
            paths=paths,
            n_live=50,
            number_of_cores=self.number_of_cores,
        )

        analysis = al.AnalysisImaging(dataset=dataset)
        analysis._adapt_images = self.adapt_images

        return search.fit(model=model, analysis=analysis)


"""
__Perturb Fit__

We now define a `PerturbFit` class, which defines how the `perturb_model` is fitted to each simulated dataset. This
behaves analogously to the `BaseFit` class above, but now fits the `perturb_model` to the simulated data (as
opposed to the `base_model`).

Again, in this example we use a full non-linear search to fit the `perturb_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `perturb_model` fit
to the simulated data.
"""


class PerturbFit:
    def __init__(self, adapt_images, number_of_cores: int = 1):
        """
        Class used to fit every dataset used for sensitivity mapping with the perturbed model (the model with the
        perturbed feature sensitivity mapping maps out).

        In this example, the perturbed model therefore includes the dark matter subhalo, which is also in the
        simulated dataset.

        The perturbed fit is repeated for every parameter on the sensitivity grid and compared to the base fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        number_of_cores
            The number of cores used to perform the non-linear search. If 1, each model-fit on the grid is performed
            in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
        """
        self.adapt_images = adapt_images
        self.number_of_cores = number_of_cores

    def __call__(self, dataset, model, paths, instance):
        """
        The perturbed fitting function which fits every dataset used for sensitivity mapping with the perturbed model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which includes the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.Nautilus(
            paths=paths,
            n_live=50,
            number_of_cores=self.number_of_cores,
        )

        analysis = al.AnalysisImaging(dataset=dataset)
        analysis._adapt_images = self.adapt_images

        return search.fit(model=model, analysis=analysis)


def base_model_narrow_priors_from(base_model, result, stretch: float = 1.0):
    """
    Returns a base model where priors are updated to `UniformPriors` with a `lower_limit` and `upper_limit` which
    are a narrow range over the `simulation_instance` parameter values.

    Using this updated model can significiantly speed up sensitivity mapping, as it dramatically reduces
    the volume of parameter space that needs to be sampled.

    The downside of this approach is that these narrow priors may remove viable solutions that alter the sensitivity
    or lead to an inaccurate estimate of the Bayesian evidence.

    The size of each parameter bounds have been chosen based on previous lens modeling intuition. For example, I have
    never seen a DM subhalo change the centre of a lens mass model by more than 0.01", therefore this is the value
    used for bounding that parameter.

    Parameters
    ----------
    base_model
        The base model which will be used for sensitivity mapping, which this function updates to have narrower priors.
    result
        The result used to set up the base model and which is used to set these updated priors.
    stretch
        A multiplicative factor which can be used to shrink or broaden the priors more.

    Returns
    -------
    A base model with priors updated to narrow uniform priors.
    """

    if hasattr(base_model.galaxies.lens, "mass"):
        base_model.galaxies.lens.mass.centre.centre_0 = (
            result.model_centred_max_lh_bounded(
                b=0.01 * stretch
            ).galaxies.lens.mass.centre.centre_0
        )
        base_model.galaxies.lens.mass.centre.centre_1 = (
            result.model_centred_max_lh_bounded(
                b=0.01 * stretch
            ).galaxies.lens.mass.centre.centre_1
        )
        base_model.galaxies.lens.mass.ell_comps.ell_comps_0 = (
            result.model_centred_max_lh_bounded(
                b=0.05 * stretch
            ).galaxies.lens.mass.ell_comps.ell_comps_0
        )
        base_model.galaxies.lens.mass.ell_comps.ell_comps_1 = (
            result.model_centred_max_lh_bounded(
                b=0.05 * stretch
            ).galaxies.lens.mass.ell_comps.ell_comps_1
        )
        base_model.galaxies.lens.mass.einstein_radius = (
            result.model_centred_max_lh_bounded(
                b=0.1 * stretch
            ).galaxies.lens.mass.einstein_radius
        )
        base_model.galaxies.lens.mass.slope = result.model_centred_max_lh_bounded(
            b=0.1 * stretch
        ).galaxies.lens.mass.slope

    if hasattr(base_model.galaxies.lens, "shear"):
        base_model.galaxies.lens.shear.gamma_1 = result.model_centred_max_lh_bounded(
            b=0.05 * stretch
        ).galaxies.lens.shear.gamma_1
        base_model.galaxies.lens.shear.gamma_2 = result.model_centred_max_lh_bounded(
            b=0.05 * stretch
        ).galaxies.lens.shear.gamma_2

    return base_model


def visualize_sensitivity(
    result,
    paths: af.DirectoryPaths,
    mass_result: af.Result,
    mask: al.Mask2D,
):
    """
    Visualize the results of strong lens sensitivity mapping via the SLaM pipeline.

    Parameters
    ----------
    result
        The result of the sensitivity mapping, which contains grids of the log evidence and log likelihood differences.
    paths
        The paths object which defines the output path for the results of the sensitivity mapping.
    mass_result
        The result of the mass pipeline, which is used to subtract the lens light from the dataset.
    mask
        The mask used to mask the dataset, which is plotted over the lens subtracted image.
    """

    result = al.SubhaloSensitivityResult(
        result=result,
    )

    data_subtracted = (
        mass_result.max_log_likelihood_fit.subtracted_images_of_planes_list[-1]
    )

    data_subtracted = data_subtracted.apply_mask(mask=mask)

    aplt.subplot_sensitivity(result=result, data_subtracted=data_subtracted)


class Visualizer:
    def __init__(self, mass_result: af.Result, mask: al.Mask2D):
        """
        Performs on-the-fly visualization of the sensitivity mapping, outputting the results of the sensitivity
        mapping so far to hard disk after each sensitivity cell fit is complete.

        Parameters
        ----------
        mass_result
            The result of the SLaM MASS PIPELINE which ran before this pipeline.
        mask
            The Mask2D that is applied to the imaging data for model-fitting.
        """

        self.mass_result = mass_result
        self.mask = mask

    def __call__(self, sensitivity_result, paths: af.DirectoryPaths):
        """
        Called by the `Sensitivity` class after every sensitivity cell has been fitted, to visualize results so far.

        Parameters
        ----------
        sensitivity_result
            The result of the sensitivity mapping search so far.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """
        visualize_sensitivity(
            result=sensitivity_result,
            paths=paths,
            mass_result=self.mass_result,
            mask=self.mask,
        )


"""
__Dataset + Masking__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "dark_matter_subhalo"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.05,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Mesh Shape__

As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_shape=mesh_shape,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
__SUBHALO PIPELINE (sensitivity mapping)__

The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
Sensitivity mapping if given in the SLaM pipeline script `slam/subhalo/sensitivity_imaging.py`.
"""
subhalo_mass = af.Model(al.mp.NFWMCRLudlowSph)
grid_dimension_arcsec = 3.0
number_of_steps = 2
sensitivity_mask = None

base_model = mass_result.model

base_model = base_model_narrow_priors_from(
    base_model=base_model, result=mass_result
)

perturb_model = af.Model(al.Galaxy, redshift=0.5, mass=subhalo_mass)

perturb_model.mass.log10m_vir = 9.0
perturb_model.mass.c_gNFW = 12.0
perturb_model.mass.overdens = 200.0
perturb_model.mass.inner_slope = 2.2
perturb_model.mass.centre.centre_0 = af.UniformPrior(
    lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
)
perturb_model.mass.centre.centre_1 = af.UniformPrior(
    lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
)
perturb_model.mass.redshift_object = mass_result.model.galaxies.lens.redshift
perturb_model.mass.redshift_source = mass_result.model.galaxies.source.redshift


def perturb_model_prior_func(perturb_instance, perturb_model):
    b = 0.05

    perturb_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=perturb_instance.mass.centre[0] - b,
        upper_limit=perturb_instance.mass.centre[0] + b,
    )

    perturb_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=perturb_instance.mass.centre[1] - b,
        upper_limit=perturb_instance.mass.centre[1] + b,
    )

    perturb_model.mass.log10m_vir = af.UniformPrior(lower_limit=6, upper_limit=12)

    return perturb_model


simulation_instance = mass_result.instance

fit = mass_result.max_log_likelihood_fit

simulation_instance.galaxies.lens = (
    fit.model_obj_linear_light_profiles_to_light_profiles.galaxies[0]
)

galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
    result=source_pix_result_1
)
adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

paths = af.DirectoryPaths(
    name=f"subhalo__sensitivity",
    path_prefix=settings_search.path_prefix,
    unique_tag=settings_search.unique_tag,
)

simulate_cls = SimulateImagingPixelized(
    mask=mask, psf=dataset.psf, inversion=mass_result.max_log_likelihood_fit.inversion
)

sensitivity = af.Sensitivity(
    paths=paths,
    simulation_instance=simulation_instance,
    base_model=base_model,
    perturb_model=perturb_model,
    simulate_cls=simulate_cls,
    base_fit_cls=BaseFit(
        adapt_images=adapt_images, number_of_cores=settings_search.number_of_cores
    ),
    perturb_fit_cls=PerturbFit(
        adapt_images=adapt_images, number_of_cores=settings_search.number_of_cores
    ),
    perturb_model_prior_func=perturb_model_prior_func,
    visualizer_cls=Visualizer(mass_result=mass_result, mask=mask),
    number_of_steps=number_of_steps,
    batch_range=None,
    mask=sensitivity_mask,
)

subhalo_results = sensitivity.run()

visualize_sensitivity(
    result=subhalo_results, paths=paths, mass_result=mass_result, mask=mask
)

"""
Finish.
"""

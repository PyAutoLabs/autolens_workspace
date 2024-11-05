import autofit as af
import autolens as al

from . import slam_util

from typing import Optional, Tuple, Union


def run_1(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    image_mesh_init: af.Model(al.AbstractImageMesh) = af.Model(al.image_mesh.Overlay),
    mesh_init: af.Model(al.AbstractMesh) = af.Model(al.mesh.Delaunay),
    image_mesh_init_shape: Tuple[int, int] = (34, 34),
    regularization_init: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The first SLaM SOURCE PIX PIPELINE, which initializes a lens model which uses a pixelized source for the source
    analysis.

    The first SOURCE PIX PIPELINE may require an adapt-image, for example to adapt the regularization scheme to the
    source's unlensed morphology. The adapt image provided by the SOURCE LP PIPELINE may not cover the entire source
    galaxy (e.g. because the MGE only captures part of the source) and produce a suboptimal fit.

    The result of this pipeline is used in the second SOURCE PIX PIPELINE to adapt the source pixelization to the
    source's unlensed morphology via an adapt image, where the adapt image produced in this pipeline will give a robust
    source image because it uses a pixelized source.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_lp_result
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    image_mesh_init
        The image mesh, which defines how the mesh centres are computed in the image-plane, used by the pixelization
        in the first search which initializes the source.
    image_mesh_init_shape
        The shape (e.g. resolution) of the image-mesh used in the initialization search (`search[1]`). This is only
        used if the image-mesh has a `shape` parameter (e.g. `Overlay`).
    mesh_init
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the first search which initializes the source.
    regularization_init
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        which fits the source light in the initialization search (`search[1]`).
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using light profiles [parameters fixed to result of SOURCE LP PIPELINE].

     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE LP PIPELINE].

     - The source galaxy's light is the input initialization image mesh, mesh and regularization scheme [parameters of 
     regularization free to vary].

    This search improves the lens mass model by modeling the source using a pixelization and computes the adapt
    images that are used in search 2.
    """

    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    image_mesh_init.shape = image_mesh_init_shape

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                point=source_lp_result.instance.galaxies.lens.point,
                mass=mass,
                shear=source_lp_result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    image_mesh=image_mesh_init,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
        extra_galaxies=al.util.chaining.extra_galaxies_from(result=source_lp_result),
        dataset_model=dataset_model,
    )

    """
    For single-dataset analyses, the following code does not change the model or analysis and can be ignored.

    For multi-dataset analyses, the following code updates the model and analysis.
    """
    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis,
        model=model,
        multi_dataset_offset=True,
        multi_source_regularization=True,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result


def run_2(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    image_mesh: af.Model(al.AbstractImageMesh) = af.Model(al.image_mesh.Hilbert),
    mesh: af.Model(al.AbstractMesh) = af.Model(al.mesh.Delaunay),
    regularization: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
    image_mesh_pixels_fixed: Optional[int] = 1000,
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The second SLaM SOURCE PIX PIPELINE, which fits a fixed lens model which uses a pixelized source for the source
    analysis.

    The second SOURCE PIX PIPELINE performs a fit using an advanced pixelizaiton which adapt the source's pixelization
    to the source's unlensed morphology.

    This feature requires an adapt-image, which is computed after the first SOURCE PIX PIPELINE.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_lp_result
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    image_mesh
        The image mesh, which defines how the mesh centres are computed in the image-plane, used by the pixelization
        in the final search which improves the source adaption.
    mesh
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the final search which improves the source adaption.
    regularization
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        in the final search which improves the source adaption.
    image_mesh_pixels_fixed
        The fixed number of pixels in the image-mesh, if an image-mesh with an input number of pixels is used
        (e.g. `Hilbert`).
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    Search 2 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using a light profiles [parameters fixed to result of SOURCE LP PIPELINE].
    - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
    - The source galaxy's light is the input final mesh and regularization.

    This search initializes the pixelization's mesh and regularization.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                point=source_lp_result.instance.galaxies.lens.point,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    image_mesh=image_mesh,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
        ),
        extra_galaxies=al.util.chaining.extra_galaxies_from(result=source_lp_result),
        dataset_model=dataset_model,
    )

    if image_mesh_pixels_fixed is not None:
        if hasattr(model.galaxies.source.pixelization.image_mesh, "pixels"):
            model.galaxies.source.pixelization.image_mesh.pixels = (
                image_mesh_pixels_fixed
            )

    """
    For single-dataset analyses, the following code does not change the model or analysis and can be ignored.

    For multi-dataset analyses, the following code updates the model and analysis.
    """
    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis,
        model=model,
        multi_dataset_offset=True,
        multi_source_regularization=True,
    )

    """
    __Search (Search 2)__

    This search uses the nested sampling algorithm Dynesty, in contrast to nearly every other search throughout the
    autolens workspace which use `Nautilus`.

    The reason is quite technical, but in a nutshell it is because the likelihood function sampled in `source_pix[2]`
    is often not smooth. This leads to behaviour where the `Nautilus` search gets stuck sampling small regions of
    parameter space indefinitely, and does not converge and terminate.

    Dynesty has proven more robust to these issues, because it uses a random walk nested sampling algorithm which
    is less susceptible to a noisy likelihood function.

    The reason this likelihood function is noisy is because it has parameters which change the distribution of source
    pixels. For example, the parameters may mean more or less source pixels cluster over the brightest regions of the
    image. In all other searches, the source pixelization parameters are fixed, ensuring that the likelihood function
    is smooth.
    """
    search = af.DynestyStatic(
        name="source_pix[2]",
        **settings_search.search_dict,
        nlive=100,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result


def run_1__mass_fixed(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_lp_result: af.Result,
    image_mesh_init: af.Model(al.AbstractImageMesh) = af.Model(al.image_mesh.Overlay),
    mesh_init: af.Model(al.AbstractMesh) = af.Model(al.mesh.Delaunay),
    image_mesh_init_shape: Tuple[int, int] = (34, 34),
    regularization_init: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    A variant of the first SLaM SOURCE PIX PIPELINE, which using a fixed mass model initializes a lens model which
    uses a pixelized source for the source analysis.

    This pipeline is used for fits to multiple datasts where a fit to the first primaary dataset has already been
    performed and its mass model is fixed and applied to all other datasets. This changes the first SOURCE PIX PIPELINE
    to use fixed mass model instances.
    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_lp_result
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    image_mesh_init
        The image mesh, which defines how the mesh centres are computed in the image-plane, used by the pixelization
        in the first search which initializes the source.
    image_mesh_init_shape
        The shape (e.g. resolution) of the image-mesh used in the initialization search (`search[1]`). This is only
        used if the image-mesh has a `shape` parameter (e.g. `Overlay`).
    mesh_init
        The mesh, which defines how the source is reconstruction in the source-plane, used by the pixelization
        in the first search which initializes the source.
    regularization_init
        The regularization, which places a smoothness prior on the source reconstruction, used by the pixelization
        which fits the source light in the initialization search (`search[1]`).
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using light profiles [parameters fixed to result of SOURCE LP PIPELINE].

     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE LP PIPELINE].

     - The source galaxy's light is the input initialization image mesh, mesh and regularization scheme [parameters of 
     regularization free to vary].

    This search improves the lens mass model by modeling the source using a pixelization and computes the adapt
    images that are used in search 2.
    """

    image_mesh_init.shape = image_mesh_init_shape

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                point=source_lp_result.instance.galaxies.lens.point,
                mass=source_lp_result.instance.galaxies.lens.mass,
                shear=source_lp_result.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    image_mesh=image_mesh_init,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
        extra_galaxies=al.util.chaining.extra_galaxies_from(result=source_lp_result),
        dataset_model=dataset_model,
    )

    """
    For single-dataset analyses, the following code does not change the model or analysis and can be ignored.

    For multi-dataset analyses, the following code updates the model and analysis.
    """
    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis,
        model=model,
        multi_dataset_offset=True,
        multi_source_regularization=True,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=75,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result

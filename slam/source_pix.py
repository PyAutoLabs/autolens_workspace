import autofit as af
import autolens as al

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
) -> af.Result:
    """
    The first SLaM SOURCE PIX PIPELINE, which initializes a lens model which uses a pixelized source for the source
    analysis.

    The first SOURCE PIX PIPELINE performs a fit using an initial pixelizaiton, which does not require an
    adapt-image to perform the fit (e.g. it does not adapt the source's pixelization to the source's unlensed
    morphology).

    The result of this pipeline is used in the second SOURCE PIX PIPELINE to adapt the source pixelization to the
    source's unlensed morphology via an adapt image.

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
        clumps=al.util.chaining.clumps_from(result=source_lp_result),
    )

    search = af.Nautilus(
        name="source_pix[1]_light[fixed]_mass[init]_source[pix_init_mag]",
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
) -> af.Result:
    """
    The second SLaM SOURCE PIX PIPELINE, which fits a fixed lens model which uses a pixelized source for the source
    analysis.

    The second SOURCE PIX PIPELINE performs a fit using an advanced pixelizaiton which adapt the source's pixelization
    to the source's unlensed morphology. This feature requires an adapt-image, which is computed after the
    first SOURCE PIX PIPELINE.

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
        clumps=al.util.chaining.clumps_from(result=source_lp_result),
    )

    if image_mesh_pixels_fixed is not None:
        if hasattr(model.galaxies.source.pixelization.image_mesh, "pixels"):
            model.galaxies.source.pixelization.image_mesh.pixels = (
                image_mesh_pixels_fixed
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
        name="source_pix[2]_light[fixed]_mass[fixed]_source[pix]",
        **settings_search.search_dict,
        nlive=100,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result

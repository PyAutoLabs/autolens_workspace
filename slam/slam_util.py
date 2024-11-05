import copy
import numpy as np
from os import path
from typing import List, Union

import autofit as af
import autolens as al
import autolens.plot as aplt


def output_model_to_fits(
    output_path: str,
    result,
    model_lens_light: bool = False,
    model_source_light: bool = False,
    source_reconstruction: bool = False,
):
    """
    Output modeling results from the SLAM pipeline to .fits files.

    These typically go to the path the dataset is stored in, so that a dataset can be extended with the modeling results
    easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result
        The result object from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    model_lens_light
        When to output a 2D image of the lens light model to a .fits file.
    model_source_light
        When to output a 2D image of the source light model to a .fits file.
    source_reconstruction
        When to output a 2D image of the source reconstruction to a .fits file, where this may be interpolated from
        an irregular pixelization like a Delaunay mesh or Voronoi mesh.
    """
    fit = result.max_log_likelihood_fit

    if model_lens_light:
        lens_subtracted_image_2d = fit.model_images_of_planes_list[0]
        lens_subtracted_image_2d.output_to_fits(
            file_path=path.join(output_path, "lens_light.fits"), overwrite=True
        )

    if model_source_light:
        source_subtracted_image_2d = fit.model_images_of_planes_list[-1]
        source_subtracted_image_2d.output_to_fits(
            file_path=path.join(output_path, "source_light.fits"), overwrite=True
        )

    if source_reconstruction:
        inversion = fit.inversion
        mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]
        mapper_valued = al.MapperValued(
            mapper=mapper, values=inversion.reconstruction_dict[mapper]
        )

        interpolated_reconstruction = mapper_valued.interpolated_array_from(
            shape_native=(601, 601)
        )

        interpolated_reconstruction.output_to_fits(
            file_path=path.join(output_path, "source_reconstruction.fits"),
            overwrite=True,
        )


def output_model_results(
    output_path: str,
    result,
    filename: str = "model.results",
):
    """
    Outputs the results of a model-fit to an easily readable `model.results` file containing the model parameters and
    log likelihood of the fit.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result
        The result object from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    filename
        The name of the file that the results are written to.
    """

    from autofit.text import text_util
    from autofit.tools.util import open_

    result_info = text_util.result_info_from(
        samples=result.samples,
    )

    with open_(path.join(output_path, filename), "w") as f:
        f.write(result_info)
        f.close()


def plot_fit_png_row(
    plotter_main,
    fit,
    tag,
    vmax,
    vmax_lens_light,
    vmax_convergence,
    image_plane_extent,
    source_plane_extent,
    visuals_2d,
):
    """
    Plots a row of a subplot which shows a fit to a list of datasets (e.g. varying across wavelengths) where each row
    corresponds to a different dataset.

    Parameters
    ----------
    plotter_main
        The main plotter object that is used to create the subplot.
    fit
        The fit to the dataset that is plotted, which corresponds to a row in the subplot.
    tag
        The tag that labels the row of the subplot.
    vmax
        The maximum pixel value of the subplot, which is chosen based on all fits in the list in order to make visual
        comparison easier.
    vmax_lens_light
        The maximum pixel value of the lens light subplot, again chosen based on all fits in the list.
    vmax_convergence
        The maximum pixel value of the convergence subplot, again chosen based on all fits in the list.
    image_plane_extent
        The extent of the image-plane grid that is plotted, chosen to be the same for all fits in the list.
    source_plane_extent
        The extent of the source-plane grid that is plotted, chosen to be the same for all fits in the list.
    visuals_2d
        The 2D visuals that are plotted on the subplot, which are chosen to be the same for all fits in the list.
    """

    plotter = aplt.FitImagingPlotter(
        fit=fit,
        include_2d=aplt.Include2D(
            light_profile_centres=False, mass_profile_centres=False
        ),
    )

    plotter_main.mat_plot_2d.axis = aplt.Axis(extent=image_plane_extent)
    plotter_main.mat_plot_2d.cmap = aplt.Cmap()
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.mat_plot_2d.use_log10 = True
    plotter.set_title(label=f"{tag} Data")
    plotter.figures_2d(data=True)
    plotter.mat_plot_2d.use_log10 = False

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.set_title(label=f"{tag} Lens Subtracted Image")
    plotter.figures_2d_of_planes(
        plane_index=1, subtracted_image=True, use_source_vmax=True
    )

    visuals_2d_original = copy.copy(plotter_main.visuals_2d)

    plotter_main.visuals_2d = visuals_2d
    plotter.visuals_2d = plotter_main.visuals_2d
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.set_title(label=f"{tag} Lensed Source Model")
    plotter.figures_2d_of_planes(plane_index=1, model_image=True, use_source_vmax=True)
    plotter.visuals_2d = visuals_2d_original
    plotter_main.visuals_2d = visuals_2d_original

    plotter_main.mat_plot_2d.axis = aplt.Axis(extent=source_plane_extent)
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"{tag} Source Plane")
    plotter.figures_2d_of_planes(plane_index=1, plane_image=True, use_source_vmax=True)

    plotter_main.mat_plot_2d.axis = aplt.Axis(extent=image_plane_extent)
    plotter.set_title(label=f"{tag} Lens Light")
    plotter.mat_plot_2d.use_log10 = True

    tracer_plotter = plotter.tracer_plotter
    tracer_plotter.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_lens_light)
    tracer_plotter.include_2d._light_profile_centres = False
    tracer_plotter.include_2d._mass_profile_centres = False
    tracer_plotter.include_2d._tangential_critical_curves = False
    tracer_plotter.include_2d._radial_critical_curves = False

    try:
        tracer_plotter.figures_2d_of_planes(
            plane_image=True, plane_index=0, zoom_to_brightest=False
        )
    except ValueError:
        plotter_main.mat_plot_2d.subplot_index += 1
        pass

    tracer_plotter = plotter.tracer_plotter
    tracer_plotter.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_convergence)

    tracer_plotter.set_title(label=f"{tag} Convergence")
    tracer_plotter.figures_2d(convergence=True)

    tracer_plotter.include_2d._light_profile_centres = True
    tracer_plotter.include_2d._mass_profile_centres = True
    tracer_plotter.include_2d._tangential_critical_curves = True
    tracer_plotter.include_2d._radial_critical_curves = True

    plotter.mat_plot_2d.use_log10 = False


def output_fit_multi_png(output_path: str, result_list, tag_list=None, filename="fit"):
    """
    Outputs a .png subplot of a fit to multiple datasets (e.g. varying across wavelengths) where each row
    corresponds to a different dataset.

    Many aspects of the plot are homogenized so that the fits can be compared easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result_list
        A list of results from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    tag_list
        A list of tags to label each row of the subplot.
    filename
        The name of the file that the results are written to.
    """
    fit_list = [result.max_log_likelihood_fit for result in result_list]
    mapper_list = [
        fit.inversion.cls_list_from(cls=al.AbstractMapper)[0] for fit in fit_list
    ]
    pixel_values_list = [
        fit.inversion.reconstruction_dict[mapper]
        for fit, mapper in zip(fit_list, mapper_list)
    ]
    extent_list = [
        mapper.extent_from(values=pixel_values)
        for mapper, pixel_values in zip(mapper_list, pixel_values_list)
    ]

    source_plane_extent = [
        np.min([extent[0] for extent in extent_list]),
        np.max([extent[1] for extent in extent_list]),
        np.min([extent[2] for extent in extent_list]),
        np.max([extent[3] for extent in extent_list]),
    ]

    vmax = (
        np.max([np.max(fit.model_images_of_planes_list[1]) for fit in fit_list]) / 2.0
    )

    image_plane_extent = fit_list[0].data.extent_of_zoomed_array()

    vmax_lens_light = np.min(
        [np.max(fit.model_images_of_planes_list[0]) for fit in fit_list]
    )

    vmax_convergence = np.min(
        [
            np.max(fit.tracer.convergence_2d_from(grid=fit.dataset.grid))
            for fit in fit_list
        ]
    )

    plotter = aplt.FitImagingPlotter(
        fit=fit_list[0],
        mat_plot_2d=aplt.MatPlot2D(
            output=aplt.Output(path=output_path, filename=filename, format="png"),
        ),
    )

    plotter.open_subplot_figure(
        number_subplots=len(fit_list) * 6,
        subplot_shape=(len(fit_list), 6),
    )

    for i, fit in enumerate(fit_list):
        visuals_2d = aplt.Visuals2D(
            light_profile_centres=al.Grid2DIrregular(
                values=[fit.tracer.galaxies[0].bulge.profile_list[0].centre]
            ),
            mass_profile_centres=al.Grid2DIrregular(
                values=[fit.tracer.galaxies[0].mass.centre]
            ),
        )

        tag = tag_list[i] if tag_list is not None else ""

        plot_fit_png_row(
            plotter_main=plotter,
            fit=fit,
            tag=tag,
            vmax=vmax,
            vmax_lens_light=vmax_lens_light,
            vmax_convergence=vmax_convergence,
            image_plane_extent=image_plane_extent,
            source_plane_extent=source_plane_extent,
            visuals_2d=visuals_2d,
        )

    plotter.mat_plot_2d.output.subplot_to_figure(auto_filename=filename)
    plotter.close_subplot_figure()


def plot_source_png_row(
    plotter_main,
    fit,
    tag,
    vmax,
    source_plane_extent,
):
    """
    Plots a row of a subplot which shows a source reconstruction to a list of datasets (e.g. varying across wavelengths)
    where each row corresponds to a different dataset.

    Parameters
    ----------
    plotter_main
        The main plotter object that is used to create the subplot.
    fit
        The fit to the dataset that is plotted, which corresponds to a row in the subplot.
    tag
        The tag that labels the row of the subplot.
    vmax
        The maximum pixel value of the subplot, which is chosen based on all fits in the list in order to make visual
        comparison
    source_plane_extent
        The extent of the source-plane grid that is plotted, chosen to be the same for all fits in the list.
    """
    plotter = aplt.FitImagingPlotter(
        fit=fit,
        include_2d=aplt.Include2D(
            light_profile_centres=False, mass_profile_centres=False
        ),
    )

    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax)
    plotter.set_title(label=f"{tag} Source Plane")
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_image=True,
        use_source_vmax=True,
        zoom_to_brightest=False,
    )

    plotter_main.mat_plot_2d.axis = aplt.Axis(extent=source_plane_extent)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"{tag} Source Plane (Zoomed)")
    plotter.figures_2d_of_planes(plane_index=1, plane_image=True, use_source_vmax=True)

    plotter.set_title(label=f"{tag} Source Plane Log10 (Zoomed)")
    plotter.mat_plot_2d.use_log10 = True
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_image=True,
    )

    plotter.set_title(label=f"{tag} Source Plane Errors (Zoomed)")
    plotter.mat_plot_2d.use_log10 = False
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_errors=True,
    )

    plotter.set_title(label=f"{tag} Source Plane S/N (Zoomed)")
    plotter.mat_plot_2d.use_log10 = False
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_signal_to_noise_map=True,
    )

    plotter_main.mat_plot_2d.axis = aplt.Axis()
    plotter.set_title(label=f"{tag} Source Plane Interpolation")
    plotter.mat_plot_2d.use_log10 = False
    plotter.figures_2d_of_planes(
        plane_index=1,
        plane_image=True,
        use_source_vmax=True,
        interpolate_to_uniform=True,
        zoom_to_brightest=False,
    )


def output_source_multi_png(
    output_path: str, result_list, tag_list=None, filename="source_reconstruction"
):
    """
    Outputs a .png subplot of the source-plane source reconstructions to multiple datasets (e.g. varying across
    wavelengths) where each row corresponds to a different dataset.

    Many aspects of the plot are homogenized so that the fits can be compared easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result_list
        A list of results from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    tag_list
        A list of tags to label each row of the subplot.
    filename
        The name of the file that the results are written to.
    """

    fit_list = [result.max_log_likelihood_fit for result in result_list]

    mapper_list = [
        fit.inversion.cls_list_from(cls=al.AbstractMapper)[0] for fit in fit_list
    ]
    pixel_values_list = [
        fit.inversion.reconstruction_dict[mapper]
        for fit, mapper in zip(fit_list, mapper_list)
    ]
    extent_list = [
        mapper.extent_from(values=pixel_values)
        for mapper, pixel_values in zip(mapper_list, pixel_values_list)
    ]

    source_plane_extent = [
        np.min([extent[0] for extent in extent_list]),
        np.max([extent[1] for extent in extent_list]),
        np.min([extent[2] for extent in extent_list]),
        np.max([extent[3] for extent in extent_list]),
    ]

    vmax = (
        np.max([np.max(fit.model_images_of_planes_list[1]) for fit in fit_list]) / 2.0
    )

    plotter_main = aplt.FitImagingPlotter(
        fit=fit_list[0],
        mat_plot_2d=aplt.MatPlot2D(
            output=aplt.Output(path=output_path, filename=filename, format="png"),
        ),
    )

    plotter_main.open_subplot_figure(
        number_subplots=len(fit_list) * 6,
        subplot_shape=(len(fit_list), 6),
    )

    for i, fit in enumerate(fit_list):
        tag = tag_list[i] if tag_list is not None else ""

        plot_source_png_row(
            plotter_main=plotter_main,
            fit=fit,
            tag=tag,
            vmax=vmax,
            source_plane_extent=source_plane_extent,
        )

    plotter_main.mat_plot_2d.output.subplot_to_figure(auto_filename=filename)
    plotter_main.close_subplot_figure()


def plot_mge_only_row(
    plotter_main,
    fit,
    tag,
    mask,
    vmax_data,
    vmax_mge,
):
    """
    Plots a row of a subplot which shows a MGE lens light subtraction to a list of datasets (e.g. varying across
    wavelengths) where each row corresponds to a different dataset.

    Parameters
    ----------
    plotter_main
        The main plotter object that is used to create the subplot.
    fit
        The fit to the dataset that is plotted, which corresponds to a row in the subplot.
    tag
        The tag that labels the row of the subplot.
    mask
        The mask applied to the data, which is used to plot the residual map.
    vmax_data
        The maximum pixel value of the data subplot, which is chosen based on all fits in the list in order to make
        visual comparison easier.
    vmax_mge
        The maximum pixel value of the MGE lens light subtraction subplot, chosen based on all fits in the list.
    """
    vmax_mge_2 = vmax_mge / 3.0
    vmax_mge_3 = vmax_mge / 10.0

    visuals = aplt.Visuals2D(
        mask=mask,
    )

    plotter = aplt.FitImagingPlotter(
        fit=fit, visuals_2d=visuals, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
    )

    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.mat_plot_2d.use_log10 = True
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_data)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"Data Pre MGE {tag}")
    plotter.figures_2d(data=True)
    plotter.mat_plot_2d.use_log10 = False

    plotter = aplt.FitImagingPlotter(fit=fit, visuals_2d=visuals)

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_data)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"Data Pre MGE {tag}")
    plotter.figures_2d(data=True)

    plotter = aplt.FitImagingPlotter(
        fit=fit,
        mat_plot_2d=aplt.MatPlot2D(
            cmap=aplt.Cmap(vmin=0.0, vmax=vmax_mge),
        ),
        visuals_2d=visuals,
    )

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_mge)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)

    plotter = aplt.FitImagingPlotter(fit=fit, visuals_2d=visuals)

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_mge_2)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)

    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=0.0, vmax=vmax_mge_3)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)

    plotter.mat_plot_2d.use_log10 = True
    plotter_main.mat_plot_2d.cmap = aplt.Cmap(vmin=1.0e-3, vmax=vmax_mge)
    plotter.mat_plot_2d = plotter_main.mat_plot_2d
    plotter.set_title(label=f"MGE Subtraction {tag}")
    plotter.figures_2d(residual_map=True)
    plotter.mat_plot_2d.use_log10 = False


def output_subplot_mge_only_png(
    output_path: str, result_list, tag_list=None, filename="mge_only"
):
    """
    Outputs a .png subplot of the MGE lens light subtraction (without mass or source models) to multiple
    datasets (e.g. varying across wavelengths) where each row corresponds to a different dataset.

    Many aspects of the plot are homogenized so that the fits can be compared easily.

    Parameters
    ----------
    output_path
        The path to the output directory where the modeling results are stored.
    result_list
        A list of results from the SLaM pipeline used to make the modeling results, typically the MASS PIPELINE.
    tag_list
        A list of tags to label each row of the subplot.
    filename
        The name of the file that the results are written to.
    """
    fit_list = [result.max_log_likelihood_fit for result in result_list]

    vmax_data = np.max([np.max(fit.data) for fit in fit_list]) / 2.0

    vmax_mge_list = []

    for fit in fit_list:
        image = fit.residual_map.native
        mask = al.Mask2D.circular(
            radius=0.3,
            pixel_scales=fit.dataset.pixel_scales,
            shape_native=image.shape_native,
        )

        vmax = image[mask].max()

        vmax_mge_list.append(vmax)

    vmax_mge = np.max(vmax_mge_list)

    plotter_main = aplt.FitImagingPlotter(
        fit=fit_list[0],
        mat_plot_2d=aplt.MatPlot2D(
            output=aplt.Output(path=output_path, filename=filename, format="png"),
        ),
    )

    plotter_main.open_subplot_figure(
        number_subplots=len(fit_list) * 6,
        subplot_shape=(len(fit_list), 6),
    )

    for i, fit in enumerate(fit_list):
        tag = tag_list[i] if tag_list is not None else ""

        plot_mge_only_row(
            plotter_main=plotter_main,
            fit=fit,
            tag=tag,
            mask=fit.mask,
            vmax_data=vmax_data,
            vmax_mge=vmax_mge,
        )

    plotter_main.mat_plot_2d.output.subplot_to_figure(auto_filename=filename)
    plotter_main.close_subplot_figure()


def analysis_multi_dataset_from(
    analysis: Union[af.Analysis, af.CombinedAnalysis],
    model,
    multi_dataset_offset: bool = False,
    multi_source_regularization: bool = False,
    source_regularization_result=None,
):
    """
    Updates the `Analysis` object to include free parameters for multi-dataset modeling.

    The following updates can be made:

    - The arc-second (y,x) offset between two datasets for multi-band fitting, where a different offset is used for each
      dataset (e.g. 2 extra free parameters per dataset).

    - The regularization parameters of the pixelization used to reconstruct the source, where different regularization
      parameters are used for each dataset (e.g. 1-3 extra free parameters per dataset).

    - The regularization parameters of the pixelization used to reconstruct the source are fixed to the max log likelihood
      instance of the regularization from a previous model-fit (e.g. the SOURCE PIPELINE).

    The function is quite rigid and should not be altered to change the behavior of the multi wavelength SLaM pipelines.
    Future updates will add more flexibility, once multi wavelength modeling is better understood.

    Parameters
    ----------
    analysis
        The sum of analysis classes that are used to fit the data.
    model
        The model used to fit the data, which is extended to include the extra free parameters.
    multi_dataset_offset
        If True, a different (y,x) arc-second offset is used for each dataset.
    multi_source_regularization
        If True, a different regularization parameters are used for each dataset.
    source_regularization_result
        The result of a previous model-fit that is used to fix the regularization parameters of the source pixelization.

    Returns
    -------
    The updated analysis object that includes the extra free parameters.
    """
    if not isinstance(analysis, af.CombinedAnalysis):
        return analysis

    if multi_dataset_offset and not multi_source_regularization:
        analysis = analysis.with_free_parameters(model.dataset_model.grid_offset)
    elif not multi_dataset_offset and multi_source_regularization:
        analysis = analysis.with_free_parameters(
            model.galaxies.source.pixelization.regularization
        )
    elif multi_dataset_offset and multi_source_regularization:
        analysis = analysis.with_free_parameters(
            model.dataset_model.grid_offset,
            model.galaxies.source.pixelization.regularization,
        )

    for i in range(1, len(analysis)):
        if multi_dataset_offset:
            analysis[i][
                model.dataset_model.grid_offset.grid_offset_0
            ] = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
            analysis[i][
                model.dataset_model.grid_offset.grid_offset_1
            ] = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    if source_regularization_result is not None:
        for i in range(len(analysis)):
            analysis[i][
                model.galaxies.source.pixelization.regularization
            ] = source_regularization_result[
                i
            ].instance.galaxies.source.pixelization.regularization

    return analysis

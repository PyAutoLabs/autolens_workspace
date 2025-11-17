import numpy as np
from typing import Tuple

import autofit as af
import autolens as al
import autolens.plot as aplt

from autofit.non_linear.grid.sensitivity.result import SensitivityResult


def visualize_subhalo_detect(
    result_no_subhalo: af.Result,
    result: af.GridSearchResult,
    analysis,
    paths: af.DirectoryPaths,
):
    """
    Visualize the results of a subhalo detection grid search using the SLaM pipeline.

    This outputs the following visuals:

    - The `log_evidence` increases in each cell of the subhalo detection grid search, which is plotted over a lens
    subtracted  image of the dataset.

    - The subhalo `mass` inferred for every cell of the grid search, plotted over the lens subtracted image.

    - A subplot showing different aspects of the fit, so that the its with and without a subhalo can be compared.

    Parameters
    ----------
    result_no_subhalo
        The result of the model-fitting without a subhalo.
    result
        The grid search result of the subhalo detection model-fitting.
    analysis
        The analysis class used to perform the model fit.
    paths
        The paths object which defines the output path for the results of the subhalo detection grid search.
    """
    result = al.subhalo.SubhaloGridSearchResult(
        result=result,
    )

    fit_no_subhalo = result_no_subhalo.max_log_likelihood_fit

    fit_imaging_with_subhalo = analysis.fit_from(
        instance=result.best_samples.max_log_likelihood(),
    )

    output = aplt.Output(
        path=paths.output_path,
        format="png",
    )

    evidence_max = 30.0
    evidence_half = evidence_max / 2.0

    colorbar = aplt.Colorbar(
        manual_tick_values=[0.0, evidence_half, evidence_max],
        manual_tick_labels=[
            0.0,
            np.round(evidence_half, 1),
            np.round(evidence_max, 1),
        ],
    )
    colorbar_tickparams = aplt.ColorbarTickParams(labelsize=22, labelrotation=90)

    mat_plot = aplt.MatPlot2D(
        axis=aplt.Axis(extent=result.extent),
        #  colorbar=colorbar,
        #  colorbar_tickparams=colorbar_tickparams,
        output=output,
    )

    subhalo_plotter = al.subhalo.SubhaloPlotter(
        result=result,
        fit_imaging_no_subhalo=fit_no_subhalo,
        fit_imaging_with_subhalo=fit_imaging_with_subhalo,
        mat_plot_2d=mat_plot,
    )

    subhalo_plotter.figure_figures_of_merit_grid(
        use_log_evidences=True,
        relative_to_value=result_no_subhalo.samples.log_evidence,
        remove_zeros=True,
    )

    subhalo_plotter.figure_mass_grid()
    subhalo_plotter.subplot_detection_imaging()
    subhalo_plotter.subplot_detection_fits()


def visualize_sensitivity(
    result: SensitivityResult,
    paths: af.DirectoryPaths,
    mass_result: af.Result,
    mask: al.Mask2D,
):
    """
    Visualize the results of strong lens sensitivity mapping via the SLaM pipeline.

    This outputs the following visuals:

    - The `log_evidences_differences` and `log_likelihood_differences` of the sensitivity mapping,
    overlaid as a 2D grid of values over the lens subtracted image of the dataset.

    - The `log_evidences_differences` and `log_likelihood_differences` of the sensitivity mapping, as a 2D array
    not overlaid an image.

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

    output = aplt.Output(
        path=paths.output_path,
        format="png",
    )

    data_subtracted = (
        mass_result.max_log_likelihood_fit.subtracted_images_of_planes_list[-1]
    )

    data_subtracted = data_subtracted.apply_mask(mask=mask)

    mat_plot_2d = aplt.MatPlot2D(axis=aplt.Axis(extent=result.extent), output=output)

    plotter = aplt.SubhaloSensitivityPlotter(
        result=result, data_subtracted=data_subtracted, mat_plot_2d=mat_plot_2d
    )

    plotter.subplot_sensitivity()

    # try:
    #     plotter.subplot_figures_of_merit_grid()
    #     plotter.figure_figures_of_merit_grid()
    # except TypeError:
    #     plotter.subplot_figures_of_merit_grid(use_log_evidences=False)
    #     plotter.figure_figures_of_merit_grid(use_log_evidences=False)


def sensitivty_mask_brightest_from(
    mass_result,
    grid_dimensions_extent: Tuple[float, float, float, float],
    number_of_pixels: int = 27,
) -> al.Mask2D:
    """
    Returns a sensitivity mask that only includes the N brightest pixels in the lensed source image in order to focus
    sensitivity mapping on the brightest regions of the source.

    This function extracts the lensed source image from the mass result and sorts the pixels by intensity, extracting
    the input `number_of_pixels` brightest pixels such that sensitivity mapping is only performed on these pixels.

    The mask is also trimmed to the grid dimensions extent to ensure that the sensitivity mapping is only performed
    within a specific region of the image, for example to remove large regions of empty space at the edges in
    visualization.

    Parameters
    ----------
    mass_result
        The result of the mass pipeline, which contains the lensed source image.
    grid_dimensions_extent
        The extent of the grid dimensions to trim the sensitivity mask to, input as a tuple of (y0, y1, x0, x1).
    number_of_pixels
        The number of brightest pixels to include in the sensitivity mask.

    Returns
    -------
    The sensitivity mask that includes only the N brightest pixels in the lensed source image.
    """

    lensed_source_image = (
        mass_result.max_log_likelihood_fit.model_images_of_planes_list[-1]
    )

    mask = mass_result.max_log_likelihood_fit.dataset.mask

    y0 = grid_dimensions_extent[0]
    y1 = grid_dimensions_extent[1]
    x0 = grid_dimensions_extent[2]
    x1 = grid_dimensions_extent[3]

    y0_pix = mask.geometry.pixel_coordinates_2d_from(scaled_coordinates_2d=(y1, x1))[0]
    y1_pix = mask.geometry.pixel_coordinates_2d_from(scaled_coordinates_2d=(y0, x0))[0]
    x0_pix = mask.geometry.pixel_coordinates_2d_from(scaled_coordinates_2d=(y0, x0))[1]
    x1_pix = mask.geometry.pixel_coordinates_2d_from(scaled_coordinates_2d=(y1, x1))[1]

    lensed_source_image = lensed_source_image.native[y0_pix:y1_pix, x0_pix:x1_pix]

    sorted_lensed_source_image = np.sort(lensed_source_image.flatten())[::-1]
    sensitivity_mask = (
        lensed_source_image < sorted_lensed_source_image[number_of_pixels - 1]
    )

    sensitivity_mask = np.flipud(sensitivity_mask)

    return al.Mask2D(
        mask=sensitivity_mask,
        pixel_scales=lensed_source_image.pixel_scales,
    )


def visualize_sensitivity_mask(mass_result, sensitivity_mask, settings_search):
    """
    Visualize the sensitivity mask used in the sensitivity mapping, as well as an image of the mask laid over
    the lensed source image.

    This visual makes it clear which regions of the lensed source image are included in the sensitivity mapping
    and which are excluded.

    The visuals are output to the output path specified in the settings search which points to where sensitivity
    mapping results are stored.

    Parameters
    ----------
    mass_result
        The result of the mass pipeline, which contains the lensed source image.
    sensitivity_mask
        The mask used in the sensitivity mapping.
    settings_search
        The settings of the sensitivity mapping search, which contains the output path for the visualization.
    """
    lensed_source_image = (
        mass_result.max_log_likelihood_fit.model_images_of_planes_list[-1]
    )

    sensitivity_mask = np.flipud(sensitivity_mask)

    sensitivity_mask = al.Mask2D(
        mask=sensitivity_mask, pixel_scales=lensed_source_image.pixel_scales
    )

    sensitivity_mask_plot = np.where(sensitivity_mask, 0.0, 1.0)

    sensitivity_mask_plot = al.Array2D(
        values=sensitivity_mask_plot,
        mask=sensitivity_mask,
    )

    paths = af.DirectoryPaths(
        name=f"subhalo__sensitivity",
        path_prefix=settings_search.path_prefix,
        unique_tag=settings_search.unique_tag,
    )

    output = aplt.Output(
        path=paths.output_path,
        format="png",
    )

    plotter = aplt.Array2DPlotter(
        array=sensitivity_mask_plot,
        mat_plot_2d=aplt.MatPlot2D(output=output),
    )
    plotter.set_filename("sensitivity_mask")
    plotter.figure_2d()

    plotter = aplt.Array2DPlotter(
        array=lensed_source_image,
        mat_plot_2d=aplt.MatPlot2D(
            axis=aplt.Axis(extent=sensitivity_mask.geometry.extent_square),
            output=output,
        ),
    )
    plotter.set_filename("sensitivity_masked_image")
    plotter.figure_2d()


class Visualizer:
    def __init__(self, mass_result: af.Result, mask: al.Mask2D):
        """
        Performs on-the-fly visualization of the sensitivity mapping, outputting the results of the sensitivity
        mapping so far to hard disk after each sensitivity cell fit is complete.

        This means that the sensitivity mapping results grid are updated throughout the sensitivity mapping run and
        can be inspected before the full analysis has completed. Due to the long run times of sensitivity mapping,
        this allows inspection of the results before the full analysis has completed.

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
        The `visualizer_cls` is called by the `Sensitivity` class after the `base_model` and `perturb_model` have been
        fitted to the simulated data, after every sensitivity cell has been fitted.

        This function receives the result of the fit to the `base_model` and `perturb_model` of all previously completed
        sensitivity cells and is able to visualize the results of all of sensitivity mapping perform so far.

        Parameters
        ----------
        sensitivity_result
            The result of the sensitivity mapping search so far.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.

        Returns
        -------

        """
        visualize_sensitivity(
            result=sensitivity_result,
            paths=paths,
            mass_result=self.mass_result,
            mask=self.mask,
        )

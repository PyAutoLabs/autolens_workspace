from typing import Union, Dict, List, Optional

import autofit as af
import autolens as al


def adapt_fit(
    setup_adapt: al.SetupAdapt,
    result: af.Result,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    search_previous: af.NonLinearSearch,
    include_hyper_image_sky: bool = False,
):
    """
    Perform a adapt-fit, which extends a model-fit with an additional fit which fixes the non-pixelization components of the
    model (e.g., `LightProfile`'s, `MassProfile`) to the `Result`'s maximum likelihood fit. The adapt-fit then treats
    only the adaptive pixelization's components as free parameters, which are any of the following model components:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.

    The adapt model is typically used in pipelines to refine and improve an `Inversion` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_adapt
        The setup of the adapt fit.
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the adapt model.
    analysis
        An analysis which is used to fit imaging or interferometer data with a model.
    include_hyper_image_sky
        Whether to include the hyper image sky component, irrespective of the `setup_adapt`.

    Returns
    -------
    af.Result
        The result of the adapt model-fit, which has a new attribute `result.adapt` that contains updated parameter
        values for the adaptive pixelization's components for passing to later model-fits.
    """

    return al.util.model.adapt_fit(
        setup_adapt=setup_adapt,
        result=result,
        analysis=analysis,
        search_previous=search_previous,
        include_hyper_image_sky=include_hyper_image_sky,
    )


def stochastic_fit(
    result: af.Result,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    search_previous: af.NonLinearSearch,
    include_lens_light: bool = False,
    include_pixelization: bool = False,
    include_regularization: bool = False,
    search_cls: af.NonLinearSearch = af.DynestyStatic,
    search_pix_dict: Optional[Dict] = None,
    info: Optional[Dict] = None,
    pickle_files: Optional[List] = None,
):
    """
    Extend a model-fit with a stochastic model-fit, which refits a model but introduces a log likelihood cap whereby
    all model-samples with a likelihood above this cap are rounded down to the value of the cap.

    This `log_likelihood_cap` is determined by sampling ~250 log likelihood values from the original model's maximum
    log likelihood model. However, the pixelization used to reconstruct the source of each model evaluation uses a
    different KMeans seed, such that each reconstruction uses a unique pixel-grid. The model must therefore use a
    pixelization which uses the KMeans method to construct the pixel-grid, for example the `VoronoiBrightnessImage`.

    The cap is computed as the mean of these ~250 values and it is introduced to avoid underestimated errors due
    to artificial likelihood boosts.

    Parameters
    ----------
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the model.
    analysis
        An analysis which is used to fit imaging or interferometer data with a model.
    include_lens_light
        If the lens light is included as a model component in the model with free parameters that are fitted for (if
        `False` it is passed as an `instance`).
    include_pixelization
        If the source pixelization is included as a model component in the model with free parameters that are fitted
        for (if `False` it is passed as an `instance`).
    include_regularization
        If the source regularization is included as a model component in the model with free parameters that are
        fitted for (if `False` it is passed as an `instance`).
    """

    if search_pix_dict is None:
        search_pix_dict = {"nlive": 100}

    stochastic_model = al.util.model.stochastic_model_from(
        result=result,
        include_lens_light=include_lens_light,
        include_pixelization=include_pixelization,
        include_regularization=include_regularization,
        subhalo_centre_width=1.0,
    )

    return al.util.model.stochastic_fit(
        stochastic_model=stochastic_model,
        search_cls=search_cls,
        search_pix_dict=search_pix_dict,
        result=result,
        analysis=analysis,
        search_previous=search_previous,
        info=info,
        pickle_files=pickle_files,
    )

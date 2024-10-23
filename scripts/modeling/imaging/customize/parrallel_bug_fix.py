"""
Modeling: Parallel Bug Fix
==========================

Depending on the operating system (e.g. Linux, Mac, Windows), Python version, if you are running a Jupyter notebook
and other factors, Python parallelization may not work (e.g. running a script with `number_of_cores` > 1 will produce
an error).

If parallelization does not work in a Jupyter notebook AND does not work when you run an example Python script, you
must use the code format illustrated here to fix the problem.

The code in this script is identical to the `autolens_workspace/scripts/modeling/imaging/start_here.py` script.
Comments have therefore been removed to avoid repetition and make the script more concise.

__The Fix__

The fix which makes parallelization work is at the end of the script, where we use the following code:

`if __name__ == "__main__":`

    `fit()`

The reason this fixes parallelization is beyond the scope of this tutorial. However, if you are curious, a quick
Google search will provide you with a detailed explanation! For example, the stack overflow page below has
some good answers:

 https://stackoverflow.com/questions/20360686/compulsory-usage-of-if-name-main-in-windows-while-using-multiprocessi

__Trouble Shooting__

If you still cannot get parallelization to work, please ask to be added to the SLACK
channel (by emailing me https://github.com/Jammy2211), where we will be able to provide support.
"""


def fit():
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
    __Dataset__
    """
    dataset_name = "simple"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Mask__
    """
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Model__
    """
    # Lens:

    bulge = af.Model(al.lp_linear.Sersic)

    mass = af.Model(al.mp.Isothermal)

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    # Source:

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    print(model.info)

    """
    __Search__ 
    """
    search = af.Nautilus(
        path_prefix=path.join("imaging", "modeling"),
        name="start_here",
        unique_tag=dataset_name,
        n_live=150,
        number_of_cores=4,
        iterations_per_update=10000,
    )

    """
    __Analysis__
    """
    analysis = al.AnalysisImaging(dataset=dataset)

    """
    __Run Times__
    """
    run_time_dict, info_dict = analysis.profile_log_likelihood_function(
        instance=model.random_instance()
    )

    print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

    print(
        "Estimated Run Time Upper Limit (seconds) = ",
        (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
        / search.number_of_cores,
    )

    """
    __Model-Fit__
    """
    result = search.fit(model=model, analysis=analysis)

    """
    __Output Folder__
    """
    print(result.info)

    print(result.max_log_likelihood_instance)

    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
    )
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

    plotter = aplt.NestPlotter(samples=result.samples)
    plotter.corner_anesthetic()

    """
    Finish.
    """


if __name__ == "__main__":
    fit()

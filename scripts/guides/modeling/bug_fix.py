"""
Modeling: Parallel Bug Fix
==========================

Depending on the operating system (e.g. Linux, Mac, Windows) and Python version, running a Python script or Jupyter
notebook may lead to a error being raised when the search begins.

The root cause of this error is that Python parallelization and JAX may only work when the script is run in a
particular format, which this script illustrates.

The code in this script is identical to the `autolens_workspace/scripts/imaging/modeling.py` script.
Comments have therefore been removed to avoid repetition and make the script more concise.

__The Fix__

The fix which makes parallelization work is at the end of the script, where we use the following code:

`if __name__ == "__main__":`

    `fit()`

The reason this fixes parallelization is beyond the scope of this tutorial. However, if you are curious, a quick
Google search will provide you with a detailed explanation! For example, the stack overflow page below has
some good answers:

 https://stackoverflow.com/questions/20360686/compulsory-usage-of-if-name-main-in-windows-while-using-multiprocessi

This fix will work for all dataset formats (e.g. `imaging`, `interferometer`) and should therefore be adopted
for any modeling script you write that has the error described above.

__Trouble Shooting__

If you still cannot get parallelization to work, please ask to be added to the SLACK
channel (by emailing me https://github.com/Jammy2211), where we will be able to provide support.
"""


def fit():
    from autoconf import (
        jax_wrapper,
    )  # Ensures JAX environment variables are set before other imports

    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

    from os import path
    from pathlib import Path
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
    mask_radius = 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    """
    __Over Sampling__
    """
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    """
    __Model__
    """
    # Lens:

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    # Source:

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    """
    __Search__ 
    """
    search = af.Nautilus(
        path_prefix=Path("imaging"),
        name="modeling",
        unique_tag=dataset_name,
        n_live=100,
        n_batch=50,  # GPU batching and VRAM use explained in `modeling` examples.
        iterations_per_quick_update=100000,
    )

    """
    __Analysis__
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
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
        tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
    )
    tracer_plotter.subplot_tracer()

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

    plotter = aplt.NestPlotter(samples=result.samples)
    plotter.corner_anesthetic()


"""
This small change in how the code is run fixes parallelization issues.
"""
if __name__ == "__main__":
    fit()

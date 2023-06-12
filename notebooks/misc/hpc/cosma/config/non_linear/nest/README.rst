Configuration files that customize the default behaviour of nested sampling non-linear searches:

**PyAutoFit** supports the following nested sampling algorithms:

 - Dynesty: https://github.com/joshspeagle/dynesty / https://dynesty.readthedocs.io/en/latest/index.html
 - UltraNest: https://github.com/JohannesBuchner/UltraNest / https://johannesbuchner.github.io/UltraNest/readme.html

Settings in the [search] and [run] entries are specific to each nested algorithm and should be determined by consulting
that nested sampler's own readthedocs.

However, because Dynesty is the default PyAutoLens search we include its specific settings here.


[search]
    nlive
        The number of live points used to sample non-linear parameter space. More points provides a more thorough
        sampling of parameter space, at the expense of taking longer to run. The number of live points required for
        accurate sampling depends on the complexity of parameter space.
    bound
        Method used to approximately bound the prior using the current set of live points. Conditions the sampling
        methods used to propose new live points. Choices are no bound ('none'), a single bounding ellipsoid
        ('single'), multiple bounding ellipsoids ('multi'), balls centered on each live point ('balls'), and cubes
        centered on each live point ('cubes'). Default is 'multi'.
    samples
        Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.
        Unique methods available are: uniform sampling within the bounds('unif'), random walks with fixed
        proposals ('rwalk'), , multivariate slice
        sampling along preferred orientations ('slice'), “random” slice sampling along all orientations ('rslice'),
        “Hamiltonian” slices along random trajectories ('hslice'), and any callable function which follows the
        pattern of the sample methods defined in dynesty.sampling. 'auto' selects the sampling method based on the
        dimensionality of the problem (from ndim). When ndim < 10, this defaults to 'unif'. When 10 <= ndim <= 20,
        this defaults to 'rwalk'. When ndim > 20, this defaults to 'hslice' if a gradient is provided and 'slice'
        otherwise. 'rstagger' and 'rslice' are provided as alternatives for 'rwalk' and 'slice', respectively.
        Default is 'auto'.
    bootstrap
        Compute this many bootstrapped realizations of the bounding objects. Use the maximum distance found to the
        set of points left out during each iteration to enlarge the resulting volumes. Can lead to unstable
        bounding ellipsoids. Default is 0 (no bootstrap).
    enlarge : float
        Enlarge the volumes of the specified bounding object(s) by this fraction. The preferred method is to
        determine this organically using bootstrapping. If bootstrap > 0, this defaults to 1.0. If bootstrap=None,
        this instead defaults to 1.25.
    vol_dec : float
        For the 'multi' bounding option, the required fractional reduction in volume after splitting an ellipsoid
        in order to accept the split. Default is 0.5.
    vol_check : float
        For the 'multi' bounding option, the factor used when checking if the volume of the original bounding
        ellipsoid is large enough to warrant > 2 splits via ell.vol > vol_check * nlive * pointvol. Default is 2.0.
    walks
        For the 'rwalk' sampling option, the minimum number of steps (minimum 2) before proposing a new live point.
        Default is 25.
    update_interval or float
        If an integer is passed, only update the proposal distribution every update_interval-th likelihood call.
        If a float is passed, update the proposal after every round(update_interval * nlive)-th likelihood call.
        Larger update intervals larger can be more efficient when the likelihood function is quick to evaluate.
        Default behavior is to target a roughly constant change in prior volume, with 1.5 for 'unif', 0.15 * walks
        for 'rwalk' and 'rstagger', 0.9 * ndim * slices for 'slice', 2.0 * slices for 'rslice', and 25.0 * slices
        for 'hslice'.
    facc : float
        The target acceptance fraction for the 'rwalk' sampling option. Default is 0.5. Bounded to be between
        [1. / walks, 1.].
    slices
        For the 'slice', 'rslice', and 'hslice' sampling options, the number of times to execute a “slice update”
        before proposing a new live point. Default is 5. Note that 'slice' cycles through all dimensions when
        executing a “slice update”.
    fmove : float
        The target fraction of samples that are proposed along a trajectories (i.e. not reflecting) for the 'hslice'
        sampling option. Default is 0.9.
    max_move
        The maximum number of timesteps allowed for 'hslice' per proposal forwards and backwards in time.
        Default is 100.


[updates]
   iterations_per_update
        The number of iterations of the non-linear search performed between every 'update', where an update performs
        visualization of the maximum log likelihood model, backing-up of the samples, output of the model.results
        file and logging.
   visualize_every_update
        For every visualize_every_update updates visualization is performed and output to the hard-disk during the
        non-linear using the maximum log likelihood model. A visualization_interval of -1 turns off on-the-fly
        visualization.
   backup_every_update
        For every backup_every_update the results of the non-linear search in the samples foler and backed up into the
        samples_backup folder. A backup_every_update of -1 turns off backups during the non-linear search (it is still
        performed when the non-linear search terminates).
   model_results_every_update
        For every model_results_every_update the model.results file is updated with the maximum log likelihood model
        and parameter estimates with errors at 1 an 3 sigma confidence. A model_results_every_update of -1 turns off
        the model.results file being updated during the model-fit (it is still performed when the non-linear search
        terminates).
   log_every_update
        For every log_every_update the log file is updated with the output of the Python interpreter. A
        log_every_update of -1 turns off logging during the model-fit.


[printing]
    silence
        If True, the default print output of the non-linear search is silcened and not printed by the Python
        interpreter.


[prior_passer]
    sigma : float
        For non-linear search chaining and model prior passing, the sigma value of the inferred model parameter used
        as the sigma of the passed Gaussian prior.
    use_errors
        If ``True``, the errors of the previous model's results are used when passing priors.
    use_widths
        If ``True`` the width of the model parameters defined in the priors config file are used.


[parallel]
    number_of_cores
        For non-linear searches that support parallel procesing via the Python multiprocesing module, the number of
        cores the parallel run uses. If number_of_cores=1, the model-fit is performed in serial omitting the use
        of the multi-processing module.
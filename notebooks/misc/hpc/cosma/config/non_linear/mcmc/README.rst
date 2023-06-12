Configuration files that customize the default behaviour of MCMC non-linear searches:

**PyAutoFit** supports the following MCMC algorithms:

 - Emcee: https://github.com/dfm/emcee / https://emcee.readthedocs.io/en/stable/
 - Zeus: https://github.com/minaskar/zeus / https://zeus-mcmc.readthedocs.io/en/latest/

Settings in the [search] and [run] entries are specific to each MCMC algorithm and should be determined by consulting
that MCMC algorithm's own readthedocs (the settings for Emcee are included at the bottom).


[initialize]
    method
        The method used to generate where walkers are initialized in parameter space, with options:
            ball (default):
                Walkers are initialized by randomly drawing unit values from a uniform distribution between the
                initialize_ball_lower_limit and initialize_ball_upper_limit values. It is recommended these limits are
                small, such that all walkers begin close to one another.
            prior:
                Walkers are initialized by randomly drawing unit values from a uniform distribution between 0 and 1,
                thus being distributed over the prior.
    ball_lower_limit : float
        The lower limit of the uniform distribution unit values are drawn from when initializing walkers using the
        ball method.
    ball_upper_limit : float
        The upper limit of the uniform distribution unit values are drawn from when initializing walkers using the
        ball method.


[auto_correlations]
    check_for_convergence
        Whether the auto-correlation lengths of the Emcee samples are checked to determine the stopping criteria.
        If `True`, this option may terminate the Emcee run before the input number of steps, nsteps, has
        been performed. If `False` nstep samples will be taken.
    check_size
        The length of the samples used to check the auto-correlation lengths (from the latest sample backwards).
        For convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
        thus requires more samples meet the auto-correlation threshold, taking longer to terminate sampling.
        However, shorter chains risk stopping sampling early due to noise.
    required_length
        The length an auto_correlation chain must be for it to be used to evaluate whether its change threshold is
        sufficiently small to terminate sampling early.
    change_threshold : float
        The threshold value by which if the change in auto_correlations is below sampling will be terminated early.


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


The settings for Emcee:

[search]
    nwalkers -> int
        The number of walkers in the ensemble used to sample parameter space.
    nsteps -> int
        The number of steps that must be taken by every walker. The `NonLinearSearch` will thus run for nwalkers *
        nsteps iterations.
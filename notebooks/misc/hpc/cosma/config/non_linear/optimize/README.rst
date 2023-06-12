Configuration files that customize the default behaviour of optimizer non-linear searches:

**PyAutoFit** supports the following optimizer algorithms:

 - PySwarms: https://github.com/ljvmiranda921/pyswarms / https://pyswarms.readthedocs.io/en/latest/index.html

Settings in the [search] and [run] entries are specific to each optimizer algorithm and should be determined by consulting
that optimizer algorithm's own readthedocs.


[initialize]
    methoding {"ball", "prior"}
        How to spawn the initial optimizer walker. The options are `ball` (all near one another in a small region of
        parameter space using narrow range of priors) and `prior` (random draws from each parameter prior).
    ball_lower_limit : float
        The unit value of the lower narrow prior range for the ball initialization (e.g. 0.49 will draw from the 49th
        percentile of the prior in unit space).
    ball_upper_limit : float
        The unit value of the upper narrow prior range for the ball initialization (e.g. 0.51 will draw from the 51th
        percentile of the prior in unit space).


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
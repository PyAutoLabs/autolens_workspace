The ``log_likelihood_function`` folder contains example scripts showing step-by-step visual guides to how likelihoods
are evaluated in **PyAutoLens**.

The likelihood function for fitting multiple datasets simply evaluates the likelihood function of each dataset
individually and sum the likelihoods together.

Therefore, no specific likelihood function description is given in the ``multi``` package and readers should instead
refer to:

- ``autolens_workspace/*/imaging/log_likelihood_function``.
- ``autolens_workspace/*/interferometer/log_likelihood_function``.
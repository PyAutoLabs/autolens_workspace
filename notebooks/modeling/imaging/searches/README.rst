The ``searches`` folder contains example scripts which fit a lens model using different non-linear searches:

Files
-----

- ``priors.py``: Set custom priors on every parameter of a model-fit, informing the non-linear search how to sample parameter space.
- ``start_point.py``: Set the start-point of certain parameters in a model-fit.
- ``Emcee.py``: Fit a lens model using the MCMC algorithm Emcee.
- ``PySwarms.py``: Fit a lens model using the Particle Swarm Optimizer (PSO) PySwarms.
- ``Zeus.py``: Fit a lens model using the MCMC algorithm Zeus.
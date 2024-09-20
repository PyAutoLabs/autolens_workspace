The ``customize`` folder contains example scripts showing how customize the lens model fit:

Files
-----

- ``custom_mask.py``: Customize the mask applied to the imaging dataset.
- ``model_cookbook.py``: Customize the lens model by adding and removing lens mass profiles and source light profiles.
- ``over_sampling.py``: Fit a model with a different over sampling schemes for the image grid.
- ``parallel_bug_fix.py``: How to get parallel model fits to run if they crash for default examples.
- ``positions.py``: Resample unphysical mass models during lens modeling which do not ray-trace multiple images of the lensed source close to one another.
- ``priors.py``: Customize the priors on the lens model parameters.
- ``noise_covariance_matrix.py``: Account for covariance in the noise of the data.
- ``redshifts.py``: Change the redshifts of the lens and source galaxies in a lens model.

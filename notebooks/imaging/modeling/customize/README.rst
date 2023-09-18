The ``customize`` folder contains example scripts showing how customize the lens model fit:

Files
-----

- ``custom_mask.py``: Customize the mask applied to the imaging dataset.
- ``model_cookbook.py``: Customize the lens model by adding and removing lens mass profiles and source light profiles.
- ``positions.py``: Resample unphysical mass models during lens modeling which do not ray-trace multiple images of the lensed source close to one another.
- ``priors.py``: Customize the priors on the lens model parameters.
- ``sub_grid_size.py``: Fit a lens model with a different sub-grid size for the image grid.
- ``noise_covariance_matrix.py``: Account for covariance in the noise of the data.
- ``redshifts.py``: Change the redshifts of the lens and source galaxies in a lens model.

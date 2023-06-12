The ``customize`` folder contains example scripts showing how customize the lens model fit:

Files
-----

- ``custom_mask.py``: Customize the mask applied to the imaging dataset.
- ``positions.py``: Resample unphysical mass models during lens modeling which do not ray-trace multiple images of the lensed source close to one another.
- ``sub_grid_size.py``: Fit a lens model with a different sub-grid size for the image grid.
- ``noise_covariance_matrix.py``: Account for covariance in the noise of the data.
- ``redshifts.py``: Change the redshifts of the lens and source galaxies in a lens model.

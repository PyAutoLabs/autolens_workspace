The ``multi/features/slam`` folder contains example scripts showing how to analyse multiple datasets simultaneously
using the Source, Light and Mass (SLAM) pipeline.

This combines all dataset types, so it could be multple imaging datasets (e.g. multi wavelength) observations,
imaging and interferometry, combining point source modeling with extended source modeling of CCD imaging data, etc.

Files
-----

- ``independent``: Fitting the multiple datasets independently using SLAM, mapping the mass model across datasets at the end.
- ``simultaneous``: Fitting the multiple datasets simultaneously using SLAM.
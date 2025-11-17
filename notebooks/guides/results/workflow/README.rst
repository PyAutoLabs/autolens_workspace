The ``workflow`` folder contains example scripts for creating workflows which efficient inspect the results of large lens modeling results.

Workflows are designed by creating .png, .csv and .fits files from the results in the ``output`` folder for fast inspection
of results. For example, .png files showing the maximum likelihood model on a single line for quick inspection, or
.csv files showing a catalogue of all lens models for scientific interpration.

Files
-----

- ``png_make``: Make custom .png files showing lens modeling results or other quantities for quick result inspection.
- ``csv_make``: Make .csv catalogues of lens modeling results for scientific interpretation.
- ``fits_make``: Make .fits files of lens modeling results for scientific interpretation, for example .fits file images debelending the lens and lensed source light.
The ``point_source/data_preparation`` package provides tools for preparing a point source
dataset (e.g. Hubble Space Telescope) before **PyAutoLens** analysis:

However, this guide is not written yet, fortunately preparing a point source dataset is straight forward and
follows one of two procedures:

1) Extract the arc-second coordinates of the multiple images via an external processing tool or using the GUI
found in ``autolens_workspace/*/data_preparation/imaging/gui/positions.ipynb``.

2) Perform detailed modeling to do a clean deblending of the multiple images and foreground lens data in your
CCD imaging data, as described in ``autolens_workspace/*/point_source/modeling/features/deblending.ipynb``.
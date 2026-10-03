The `imaging/data_preparation` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoLens** analysis:

# Files

- `start_here`: A concise overview of how to prepare CCD imaging data for **PyAutoLens** analysis.

# Folders

- `examples`: A folder containing example scripts of how to prepare imaging data for **PyAutoLens** analysis.
- `gui`: Graphical user interface tools for marking up a dataset by hand, in the recommended order `mask_extra_galaxies` -> `positions` -> `mask_arcs` (optional), plus the fit-region mask and the lens light / extra-galaxy centres; its README lists where every product is written.
- `manual`: Preparing the same products manually in code, without the GUI (e.g. an irregular mask).

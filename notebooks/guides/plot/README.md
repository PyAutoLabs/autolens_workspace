The `plot` folder contains guides for the **PyAutoLens** plotting API.

The API provides a simple interface with matplotlib for making plots that does not require the user to
write any matplotlib code themselves.

New users should begin with `start_here` / `start_here.ipynb`.

# Files

- `start_here`: An introduction to plotting and visualization, covering `plot_array` / `plot_grid`, customizing
  figures, output to disk, config defaults and overlays and subplots.
- `plotters`: Object-by-object figures, for example light profiles, mass profiles, galaxies, tracers and 1D profiles.
- `searches`: Visualizing non-linear search results, including corner plots, GetDist plots and search-specific plots.
- `visuals`: Overlaying visuals on figures, for example critical curves, caustics, positions and profile centres.

Dataset and fit plotting is covered in each dataset package's `plot.py` script (e.g. `scripts/imaging/plot.py`).

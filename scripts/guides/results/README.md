The `results` folder contains example scripts for using the results of a **PyAutoLens** interferometer model-fit.

# Files

- `start_here`: An overview of inspecting results from an individual model-fit.

# Folders

- `examples`: Result inspection and analysis for different aspects of the fit and lens model types.
- `aggregator`: Loading model results (samples, posteriors, errors) via the aggregator, which is more efficient for large libraries of results.
- `database`: Building an SQLite3 database to manage large suites of modeling results, which scales better than scraping the `output` folder.
- `workflow`: Develop a fast workflow to inspect and manage libraries of lens modeling results, using .csv, .png and .fits files.

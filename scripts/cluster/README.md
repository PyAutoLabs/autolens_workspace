The `cluster` folder contains example scripts showing how to analyse cluster scale strong lenses,
which are lenses where 10-100+ galaxies are responsible for the lensing of 3-50+ sources.

Clusters are the top rung of the regime ladder (`multi_galaxy` -> `group` -> `cluster`; every cluster is a
multi-galaxy system, but not vice versa). A cluster's mass framework is the same as a group's — host halo(s)
plus many tidally truncated members on luminosity scaling relations. What distinguishes the cluster regime is
the **analysis strategy**: with many sources spanning a wide redshift range, the default workflow fits
point-source multiple-image positions (multi-plane, per-source redshifts) rather than reconstructing extended
sources at pixel level, and the lens galaxies' light is not modeled. Extended-source reconstruction of
individual systems is a specialised follow-up analysis.

# Start Here

New users should read the `start_here` example, which gives an overview of all examples in the folder.

# Files

- `start_here`: A simple example illustrating how to analyse cluster scale strong lenses.
- `modeling`: Detailed example of performing lens modeling of a cluster scale strong lens using the multiple image locations.
- `simulator`: Detailed example of how to simulate a cluster scale strong lens.
- `likelihood_function`: Step-by-step walkthrough of the point-source likelihood (source-plane and image-plane chi-squared).
- `csv_api`: The named-galaxy CSV schema (`mass.csv` / `light.csv` / `point.csv`) used to define cluster models in spreadsheets.
- `mass_parameterizations`: The standard Lenstool cluster model (dPIE throughout, velocity dispersions, scaling relation) built in PyAutoLens.
- `mass_parameterizations_pyautolens`: How the Lenstool model maps onto the PyAutoLens multi-galaxy parameterization (NFW halo, Isothermal mains, dPIE scaling tier anchored to the BCG).
- `plot`: How to plot a cluster-scale strong lens dataset, including the multiple-image positions of its lensed sources.

# Folders

- `lenstool`: A worked example for Lenstool users — a published cluster model (SMACS J0723) converted, reconstructed and refit in PyAutoLens.

# Results

The `modeling` example performs lens modeling but only give a brief overview of how to analyse and interpret the
results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

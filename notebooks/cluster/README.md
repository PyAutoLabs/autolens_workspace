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

- `start_here`: A simple example illustrating how to to analyse cluster scale strong lenses.
- `modeling`: Detailed example of performing lens modeling of a cluster scale strong lens using the multiple image locations.
- `simulator`: Detailed example of how to simulate a cluster scale strong lens.
- `data_preparation`: How to prepare data for cluster scale lens modeling, including marking the multiple image positions and lens galaxy centres.

# Folders

- `features`: Examples illustrating different core features for cluster scale analysis and lens modeling.

# Results

The `modeling` example performs lens modeling but only give a brief overview of how to analyse and interpret the
results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

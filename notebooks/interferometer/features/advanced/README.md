The `interferometer/features/advanced` folder contains example scripts showing how to fit a lens model to interferometer
data using different **PyAutoLens** features.

The scripts in this folder are advanced, and generally provide more niche functionality which will only be useful
for specific scientific topics.The following example scripts illustrating lens modeling where:

- `operated_light_profile`: Compact point-source emission (e.g. an AGN) fitted with operated light profiles, whose image-plane shape is specified directly.
- `shapelets`: The source (or lens) is reconstructed using shapelet basis functions.
- `potential_correction`: Gravitational imaging — pixelized corrections to the lensing potential reconstructed jointly with the source.
- `subhalo`: Fitting lens models for dark matter subhalo detection and sensitivity mapping.

# Notes

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

The `group` folder contains example scripts showing how to analyse group scale strong lenses,
which are lenses where 2-10 galaxies are responsible for the lensing of 1 or a few sources, and a dominant
group-scale dark matter halo (~10^13-10^14 solar masses) may enter the model as an explicit choice.

Groups sit on the middle rung of the regime ladder: below them, `multi_galaxy` lenses have 2+ co-dominant
galaxies but no shared halo and no tiered member populations; above them, `cluster` lenses keep the same mass
framework but switch the analysis to point-source multiple-image positions of many sources. All groups are
multi-galaxy systems, but not vice versa.

# Start Here

New users should read the `start_here` example, which gives an overview of all examples in the folder.

# Files

- `start_here`: A simple example illustrating how to analyse group scale strong lenses.

- `modeling`: Detailed example of performing lens modeling of a group scale strong lens.

- `simulator`: Detailed example of how to simulate a group scale strong lens.

- `data_preparation`: See `imaging/data_preparation` which has all tools for preparing group scale CCD imaging data.

- slam\`\`Using the Source, Light and Mass (SLAM) pipeline to perform lens modeling of group-scale strong lenses.

# Folders

- `features`: Examples illustrating different core features for group scale analysis and lens modeling.

# Results

The `modeling` example performs lens modeling but only give a brief overview of how to analyse and interpret the
results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

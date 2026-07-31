The `detect` folder contains the subhalo **detection** pipeline for a multi-galaxy strong lens.

# Files

- `start_here`: The full SLaM pipeline (five stages, copied from `multi_galaxy/slam.py`) followed by the three
  SUBHALO stages:
  1. **no subhalo** — refit the smooth model to establish a Bayesian evidence baseline.
  2. **grid search** — fit a subhalo confined to each cell of a grid over the image plane, so no single search
     has to locate a compact perturber anywhere in two dimensions at once.
  3. **refine** — refit the highest-evidence cell with the subhalo's position free.

# Reading the result

A detection is the log evidence increase between stages 1 and 3, interpreted on the scale in `start_here`.

Before believing one, apply the multi-galaxy check that script performs: compare each deflector's mass in the
subhalo model against the smooth model's. A mis-split between two co-dominant deflectors produces residuals of
the same character, in the same place, as a subhalo — so a detection that arrived alongside a shifted mass
split is more likely the split than a perturber. The folder README explains why.

The grid search's log evidence array is often more informative than the single number: a detection confined to
one cell is a different situation from one smeared across many.

# Related

- `multi_galaxy/features/advanced/subhalo/simulator`: the dataset this pipeline is run on.
- `multi_galaxy/slam.py`: the baseline whose stages this pipeline copies.
- `imaging/features/advanced/subhalo/detect`: the galaxy-scale version.

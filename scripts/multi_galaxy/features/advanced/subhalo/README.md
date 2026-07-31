The `subhalo` folder contains example scripts for detecting a **dark matter subhalo** in a multi-galaxy strong
lens — a compact, dark perturber that distorts the lensed source's arcs without producing light of its own.

Detection only. Sensitivity mapping is not covered here, the same boundary `group/features/advanced/subhalo`
draws; `imaging/features/advanced/subhalo` has it.

# What this folder is for

Subhalo detection works by comparison. A smooth model — deflectors, shear and source, no perturber — is fitted
first, and a subhalo is then added and the Bayesian evidence compared. A detection is the difference between the
two fits, never a statement about the subhalo fit alone.

That makes the detection floor a property of the smooth model: whatever the smooth model cannot describe, a
subhalo will be offered the chance to describe instead.

# The multi-galaxy difference

The smooth model here has a known weak spot. The split of mass between the two co-dominant deflectors is far
less well constrained than their total, which is the regime's standing degeneracy
(`multi_galaxy/modeling.py`).

**A wrong mass split produces residuals that look like a subhalo.** They are compact, they sit on the arcs, and
they are of the same character a perturber produces — a mis-split of roughly one percent of the total Einstein
radius is enough to produce residual power comparable to a 10^10 solar mass subhalo, and its residual is if
anything the more concentrated of the two.

Two consequences, both acted on in `detect/start_here.py`:

- **The comparison model must carry every deflector, freely fitted.** A baseline missing one is a mis-split
  model, and the grid search will find a "subhalo" compensating for the absent galaxy.
- **A detection is not conclusive on its own.** Check that the deflectors' masses in the subhalo model agree
  with the smooth model's. A detection that arrived alongside a shifted mass split is more likely the split.

# Files

- `simulator`: Simulating the pair with a dark matter subhalo on the arcs.
- `detect/start_here`: The full SLaM pipeline plus the three SUBHALO stages — evidence baseline, grid search,
  refine.

# Related

- `multi_galaxy/modeling`: the mass-split degeneracy that sets the detection floor.
- `multi_galaxy/slam.py`: the SLaM baseline whose five stages the detection pipeline copies.
- `imaging/features/advanced/subhalo`: the galaxy-scale walkthrough, including sensitivity mapping.
- `group/features/advanced/subhalo`: the same feature at group scale.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

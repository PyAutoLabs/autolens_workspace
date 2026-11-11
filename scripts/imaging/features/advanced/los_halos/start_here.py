"""
Features: Los Halos
===================

Matter along the line of sight can perturb a strong lens even when it is not at the main lens redshift. These
simulations include a population of halos on different planes and trace the source light through the resulting
geometry.

__Population Assumptions and Individual Realizations__

A halo realization depends on the assumed population and the particular objects drawn from it. A feature seen
in one simulated image should therefore be understood in the context of that realization, rather than treated
as a prediction for every lens.

Redshift matters as well as projected position. Moving an object between planes changes its role in the
multi-plane calculation; a line-of-sight halo is not automatically equivalent to a subhalo placed in the main
lens plane.

__Using the Simulators__

Read the population settings, random-number choices and redshift construction alongside the simulator code.
The alternative execution examples demonstrate their particular implementation choices. This folder supplies
simulations; it should not be read as a completed inference or detection workflow.

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
foreground light, deflecting mass and background source play different roles; retain the particular component
choices stated in each recipe.

__Choose An Example__

- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/simulator.ipynb) — Generate the example dataset.
- [simulator_jax.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/los_halos/simulator_jax.ipynb) — Explore the JAX simulation route.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

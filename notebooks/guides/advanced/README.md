The `guides/advanced` folder contains advanced guides explaining how specific aspects of **PyAutoLens** work.

These scripts are intended to help users understand how **PyAutoLens** works.

# Files

- `add_a_profile`: How to add mass and light profiles to **PyAutoLens**.
- `auto_setup`: How `setup="auto"` and `number_of_cores="auto"` choose the fastest way to compute each likelihood (JAX or numba, CPU or GPU, sparse or dense), how to read the lines printed at the start of a fit, and how to override them.
- `custom_analysis`: Write your own custom `Analysis` class to fit a different type of data or custom lens model (e.g. weak lensing).
- `multi_plane`: How multi-plane ray-tracing is implemented.
- `over_sampling`: How to use over-sampling evaluate a light profile integral within a pixel more accurately.
- `over_sampling_chaining`: How to update the over-sampling of the lensed source between chained searches.

Graphical models, which fit multiple strong lens datasets simultaneously, are in `guides/modeling/advanced`.

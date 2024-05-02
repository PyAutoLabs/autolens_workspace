The ``guide`` folder contains guides explaining how specific aspects of **PyAutoLens** work.

These scripts are intended to help users understand how **PyAutoLens** works.

Files (Beginner)
----------------

- ``tracer.py`` Performing ray-tracing and lensing calculations.
- ``fits.py`` Fitting CCD imaging data with a lens system.
- ``galaxies.py`` Creating and using galaxies and their mass and light profiles.
- ``data_structures.py``: How the NumPy arrays containing results are structured and the API for using them.
- ``mass_to_light_ratio_units.py``: The units of light and mass profiles and how to convert between them.
- ``over_sampling.py``: How to use over-sampling evaluate a light profile integral within a pixel more accurately.
- ``units_and_cosmology.py``: Unit conversions and Cosmological quantities (converting to kiloparsecs, Einstein masses, etc.).

Files (Advanced)
----------------

- ``add_a_profile.py``: How to add mass and light profiles to **PyAutoLens**.
- ``custom_analysis.py``: Write your own custom ``Analysis`` class to fit a different type of data or custom lens model (e.g. weak lensing).
- ``multi_plane.py``: How multi-plane ray-tracing is implemented.
- ``scaling_relation.py``: Use scaling relations to compose lens models for many galaxies.
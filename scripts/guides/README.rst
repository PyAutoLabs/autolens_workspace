The ``guide`` folder contains guides explaining how specific aspects of **PyAutoLens** work.

These scripts are intended to help users understand how **PyAutoLens** works.

Files (Beginner)
----------------

- ``data_structures.py``: How the NumPy arrays containing results are structured and the API for using them.
- ``plotting.py``: How to customize the appearance of plots, for example matplotlib settings.
- ``mass_to_light_ratio_units.py``: The units of light and mass profiles and how to convert between them.
- ``over_sampling.py``: How to use over-sampling evaluate a light profile integral within a pixel more accurately.
- ``units_and_cosmology.py``: Unit conversions and Cosmological quantities (converting to kiloparsecs, Einstein masses, etc.).

Folders (Beginner)
-----------------

- ``api.py``: Illustration of how to use basic **PyAutoGalaxy** API features (e.g. `Galaxy`, `FitImaging`, etc).

Files (Advanced)
----------------

- ``add_a_profile.py``: How to add mass and light profiles to **PyAutoLens**.
- ``custom_analysis.py``: Write your own custom ``Analysis`` class to fit a different type of data or custom lens model (e.g. weak lensing).
- ``multi_plane.py``: How multi-plane ray-tracing is implemented.
- ``scaling_relation.py``: Use scaling relations to compose lens models for many galaxies.
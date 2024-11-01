The ``slam`` folder contains the Source, Light and Mass (SLaM) pipeline, which are **advanced** template pipelines for
automated lens modeling:

Files (Advanced)
----------------

- ``source_lp.py``: The source pipeline which fits a light-profile source model and starts a SLaM pipeline run.
- ``source_pix.py``: The source pipeline which fits a pixelized source following the he SOURCE LP PIPELINE.

- ``light_parametric.py``: The (lens) light pipeline which fits a model to the lens galaxy's light, following the SOURCE PIPELINE.

- ``mass_total.py``: The mass pipeline which fits a total mass model (e.g. a single mass profile for the stellar and dark matter) following the source or LIGHT PIPELINE.
- ``mass_light_dark.py``: The mass pipeline which fits a light plus dark mass model (e.g. separate mass profile for the stellar and dark matter) following the LIGHT PIPELINE.

- ``slam_util.py``: Utility functions for the SLaM pipelines, for example outputting visualizations of the results.

Folders (Advanced)
------------------

- ``subhalo``: Extensions to the SLaM pipelines which perform dark matter substructure detection and sensitivity mapping following the MASS PIPELINE.
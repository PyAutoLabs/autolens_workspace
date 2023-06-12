The ``slam`` folder contains the Source, Light and Mass (SLaM) pipeline, which are **advanced** template pipelines for
automated lens modelingtick_maker.min_value:

Files (Advanced)
----------------

- ``source_lp.py``: The source pipeline which fits a parametric source model which starts a SLaM pipeline run.
- ``source_pixelization.py``: The source pipeline which fits a pixelized source reconstruction which follows the SOURCE LP PIPELINE.

- ``light_parametric.py``: The light pipeline which fits a model to the lens galaxy's light, which follows the source pipeline.

- ``mass_light_dark.py``: The mass pipeline which fits a ``light_dark`` mass model (e.g. separate mass profile for the stellar and dark matter) which follows the Light pipeline.
- ``mass_total.py``: The mass pipeline which fits a ``total`` mass model (e.g. a single mass profile for the stellar and dark matter) which follows the source or light pipeline.

- ``subhalo.py``: The subhalo pipeline which fits a dark matter substructure and follows a mass pipeline.

- ``slum_util.py``: Utilities used in the SLaM pipelines.
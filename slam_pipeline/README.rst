The ``slam_pipeline`` folder contains the Source, Light and Mass (SLaM) pipeline, which are **advanced** template pipelines for
automated lens modeling:

Files
-----

- ``source_lp``: The source pipeline which fits a light-profile source model and starts a SLaM pipeline run.
- ``source_pix``: The source pipeline which fits a pixelized source following the he SOURCE LP PIPELINE.

- ``light_parametric``: The (lens) light pipeline which fits a model to the lens galaxy's light, following the SOURCE PIPELINE.

- ``mass_total``: The mass pipeline which fits a total mass model (e.g. a single mass profile for the stellar and dark matter) following the source or LIGHT PIPELINE.
- ``mass_light_dark``: The mass pipeline which fits a light plus dark mass model (e.g. separate mass profile for the stellar and dark matter) following the LIGHT PIPELINE.

- ``subhalo``: Extensions to the SLaM pipelines which perform dark matter substructure detection and sensitivity mapping following the MASS PIPELINE.
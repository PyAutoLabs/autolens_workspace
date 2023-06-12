The ``graphical`` folder contains examples scripts illustrating how to fit a lens model to many lenses simultaneously
using graphical model, hierarchical models and expectation propagation.

**PyAutoFit** has dedicated tutorials describing graphical models, which I recommend users not familiar with
this approach to modeling read -- https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_graphical_models.html.

Files (Advanced)
----------------
- ``tutorial_1_individual_models.py``: An example of inferring global parameters from a dataset by a model to lenses one-by-one.
- ``tutorial_2_graphical_models.py``: Fitting many lenses with a graphical model that fits all datasets simultaneously to infer the global parameters.
- ``tutorial_3_graphical_benefits.py``: Illustrating the benefits of graphical modeling over simpler approaches using a more complex model.
- ``tutorial_4_hierarchical_models.py``: Fitting a hierarchical model via a graphical model where certain parameters are drawn from a parent distribution..
- ``tutorial_5_expectation_propagation.py``: Scaling graphical models up to extremely large galaxy datasets using expectation propagation.
- ``tutorial_6_science_case.py``: Example science case using EP of estimating the cosmological parameters from double Einstein rings.
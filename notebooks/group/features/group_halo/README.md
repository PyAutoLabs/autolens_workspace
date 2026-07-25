The `group_halo` feature is the group regime's signature tutorial: whether a group-scale lens model
includes a separate dark-matter halo is an **explicit modelling choice**, not an assumption. `modeling.py`
fits the same dataset with and without the halo (identical truncated-dPIE member tier in both) and walks
through the decision: the radii the arcs probe, Bayesian model comparison, and external information.
`simulator.py` simulates the dataset — its truth includes the halo; flip `include_group_halo = False` and
re-simulate to watch the verdict invert.

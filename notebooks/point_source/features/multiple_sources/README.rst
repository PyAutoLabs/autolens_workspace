The ``multiple_sources`` folder contains example notebooks for simulating and fitting a strong lens with
multiple lensed point sources at different redshifts (e.g. an Einstein Cross configuration).

Files
-----

- ``simulator``: Simulate a strong lens with two lensed point sources at two different redshifts and write each source's multiple images out as a separate ``PointDataset``.
- ``modeling``: Fit the simulated multi-source point dataset jointly using the multi/factor-graph API (one ``AnalysisPoint`` per dataset, combined into a ``FactorGraphModel``).

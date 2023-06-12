The ``database/examples`` folder contains tutorials explaining how to use a SQLite3 database for
managing large suites of  modeling results.

Files (Beginner)
----------------

- ``samples.py``: Loads the non-linear search results from the SQLite3 database and inspect the samples (e.g. parameter estimates, posterior).
- ``queries.py``: Query the database to get certain  modeling results (e.g. all lens models where `einstein_radius > 1.0`).
- ``models.py``: Inspect the models in the database (e.g. visualize their deflection angles).
- ``data_fitting.py``: Inspect the data-fitting results in the database (e.g. visualize the residuals).

Folders (Advanced)
------------------

- ``advanced``: Examples of how to use advanced database functionality.




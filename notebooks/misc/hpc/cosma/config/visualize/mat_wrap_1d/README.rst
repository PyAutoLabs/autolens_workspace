Config files which specify the default matplotlib settings when figures and subplots are plotted.

For example, the ``Figure.yaml`` config file has the following lines:

.. code-block:: bash

    figure:
    figsize=(7, 7)
    aspect=square

    subplot:
    figsize=auto
    aspect=square

This means that when a figure (e.g. a single image) is plotted it will use ``figsize=(7,7)`` and ``aspect="square`` if
the values of these parameters are not manually set by the user via the ``mat_plot_2d``.

Subplots (e.g. more than one image) will always use ``figsize="auto`` by default.

These configuration files can be customized such that the appearance of figures and subplots for a user is optimal for
your computer set up.

Examples
--------
Example scripts using all of the plot objects which have corresponding configuration files here are given at
`autolens_workspace.plot`.

Files
-----

EinsteinRadiusAXVLine.yaml
    Customizes the appearance of a 1D line (plotted via ``plt.axvline()``)showing the Einstein radius on 1D plots of mass quantities.
FillBetween.yaml
    Customizes how 1D plots are filled when making PDFs via ``plt.fill()``.
HalfLightRadiusAXVLine.yaml
    Customizes the appearance of a 1D line (plotted via ``plt.axvline()``) showing the half-light radius on 1D plots of light profile quantities.
YXPlot.yaml
    Customizes how 1D plots are plotted when plotted via``plt.plot()``.
YXScatter.yaml
    Customizes how 1D plots are plotted when plotted via``plt.scatter()``.
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

Axis.yaml
    Customizes the matplotlib axis via the ``plt.axis()`` function.
Cmap.yaml
    Customizes the matplotlib colormap via the ``colors`` objects.
Colorbar.yaml
    Customizes the matplotlib colorbar via the ``plt.colorbar()`` function.
ColorbarTickParams.yaml
    Customizes the matplotlib colorbar tick parameters via the ``plt.tick_params()`` function.
Legend.yaml
    Customizes the matplotlib legend via the ``plt.legend()`` function.
TickParams.yaml
    Customizes the matplotlib tick parameters via the ``plt.tick_params()`` function.
Title.yaml
    Customizes the matplotlib title via the ``plt.title()`` function.
XLabel.yaml
    Customizes the matplotlib xlabel via the ``plt.xlabel()`` function.
XTicks.yaml
    Customizes the matplotlib xticks via the ``plt.xticks()`` function.
YLabel.yaml
    Customizes the matplotlib ylabel via the ``plt.ylabel()`` function.
YTicks.yaml
    Customizes the matplotlib yticks via the ``plt.yticks()`` function.
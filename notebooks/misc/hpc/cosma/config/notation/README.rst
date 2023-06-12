The notation configs define the labels of every model parameter and its derived quantities, which are used when
visualizing results (for example labeling the axis of the PDF triangle plots output by a non-linear search).


Two examples using the 1D data fitting example for the config file **label.yaml** are:

[label]
    centre_0
        The label given to that parameter for non-linear search plots using that parameter, e.g. cornerplot PDF plots.
        For example, if centre_1=x, the plot axis will be labeled 'x'.

[superscript]
    Isothermal
        The superscript used on certain plots that show the results of different model-components. For example, if
        ``Isothermal=mass``, plots where the Isothermal are plotted will have a superscript ``mass``.


The **label_format.yaml** config file specifies the format certain parameters are output as in output files like the
*model.results* file. This uses standard Python formatting strings.
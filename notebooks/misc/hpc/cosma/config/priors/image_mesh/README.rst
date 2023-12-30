The ``image_mesh`` folder contains configuration files for the default priors assumed for ``image_mesh`` objects.

These model components construct the (y,x) grid of coordinates used for a mesh in the image-plane.

For example, the `Hilbert` image-mesh computes the centres of the image mesh by running a Hilbert clustering algorithm
on the data, and clusters points in its brighter regions.
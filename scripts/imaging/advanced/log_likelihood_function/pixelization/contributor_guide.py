"""
__Log Likelihood Function: Pixelization__

The `pixelization` script accompanying this one provides a step-by-step guide of log_likelihood_function used to
fit `Imaging` data with a pixelization.

This script provides a contributor guide, that gives links to every part of the source-code that performs each step
of the likelihood evaluation.

This gives contributors a sequential run through of what functions, modules and packages in the source code are called
when the likelihood is evaluated, and should help them navigate the source code.


__Source Code__

The likelihood evaluation is spread over the following two GitHub repositories:

**PyAutoArray**: https://github.com/Jammy2211/PyAutoArray
**PyAutoGalaxy**: https://github.com/Jammy2211/PyAutoGalaxy


__LH Setup: Lens Galaxy Light (Setup)__

To see examples of all light profiles checkout  the `light_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/light_profiles


Each light profile has an `image_2d_from` method that returns the image of the profile, which is used in the
likelihood function example.

Each function uses a `@aa.grid_dec.transform` decorator, which performs the coordinate transformation of the
grid to elliptical coordinates via the light profile's geometry.


__LH Setup: Lens Galaxy Mass__

To see examples of all mass profiles in **PyAutoLens** checkout the `mass_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/mass_profiles


Each light profile has an `deflections_yx_2d_from` method that returns the deflection angles of the profile,
which is used in the likelihood function example.

Each function uses a `@aa.grid_dec.transform` decorator, which performs the coordinate transformation of the
grid to elliptical coordinates via the mass profile's geometry.

These coordinate transforms are performed in the following modules:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/geometry/geometry_util.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/decorators/transform.py


__LH Setup: Lens Galaxy__

The galaxy package and module contains the `Galaxy` object:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/galaxy

The `Galaxy` object also has an `image_2d_from` method and `deflections_yx_2d_from` method that returns the image and
deflection angles of the galaxy, which call the `image_2d_from` function of each light profile, and the
`deflections_yx_2d_from` function of each mass profile and sums them.


__LH Setup: Source Galaxy Pixelization and Regularization__

To see examples of all pixelizations and regularization schemes checkout the pixelization packages in **PyAutoArray**:

 https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/pixelizations
 https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/regularization


__LH Step 1: Lens Light__

This step calls the `image_2d_from` method of the `Galaxy` object, as described above.

The input to this function is the masked dataset's `Grid2D` object, which is stored in a dataset:

- See the method `grids`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/abstract/dataset.py

The grids themselves and calculations used by them are computed in the following modules:

- https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/grids.py

The calculation also used a `blurring_grid` to evaluate light which the PSF convolution blurred into the mask,
which is named `blurring` in the grids.py module above.


__LH Step 2: Lens Light Convolution + Subtraction__

Convolution uses the `Convolver` object and its method `convolve_image`

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 3: Source Pixel Centre Calculation__

The `Overlay` `image_mesh` used in the example required the source pixel centres to be computed in the image
plane.

Calculations like this are performed in the PyAutoArray `image_mesh` module:

https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/pixelization/image_mesh

The specific calculation for the overlay is performed here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/image_mesh/overlay.py



__LH Step 4: Ray Tracing__

Ray tracing is handled in `lens` package and `tracer` module:

https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/tracer.py


To create the `Inversion` object which uses the pixelization and regularization scheme, the following modules are used:

https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/to_inversion.py


__LH Step 5: Border Relocation__

Checkout the following module for a full description of the border relocation:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/border_relocator.py


__LH Step 6: Delaunay Mesh__

The `structures.mesh` package handles the construction of the Delaunay mesh (and other meshes):

https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/structures/mesh

The `Delaunay` mesh is used in the example, and its specific implementation is at:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/mesh/triangulation_2d.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/mesh/delaunay_2d.py


__LH Step 7: Image-Source Mapping__

Checkout the modules below for a full description of a `Mapper` and the `mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/mappers
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/abstract.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/voronoi.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/mapper_util.py


__LH Step 8: Mapping Matrix__

 `Mapper.__init__`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/abstract.py
 `mapping_matrix_from`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/mapper_util.py


__LH Step 9: Blurred Mapping Matrix (f)__

This uses the methods in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 10: Data Vector (D)__

The calculation is performed in the `data_vector` attribute of an `Inversion` object:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/mapping.py

The function called is `inversion_imaging_util.data_vector_via_blurred_mapping_matrix_from`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/inversion_imaging_util.py


__LH Step 11: Curvature Matrix (F)__

The calculation is performed in the `curvature_matrix` attribute of an `Inversion` object:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/mapping.py

The function called is `inversion_imaging_util.curvature_matrix_via_blurred_mapping_matrix_from`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/imaging/inversion_imaging_util.py


__LH Step 12: Regularization Matrix (H)__

 A complete description of regularization is at the link below.

 https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/inversion/regularization
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/regularization/abstract.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/regularization/constant.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/regularization/regularization_util.py

 An `Inversion` object has a property `regularization_matrix` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 13: F + Lamdba H__

 An `Inversion` object has a property `curvature_reg_matrix` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/matrices.py


__LH Step 14: Source Reconstruction (S)__

 An `Inversion` object has a property `reconstruction` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/matrices.py


__LH Step 15: Image Reconstruction__

 The calculation is performed by the method `mapped_reconstructed_data_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/leq_util.py

 This function is called by `AbstractInversion.mapped_reconstructed_data`:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 18: Regularization Term__

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__LH Step 19: Complexity Terms__

 An `Inversion` object has a property `log_det_curvature_reg_matrix_term` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/matrices.py

 An `Inversion` object has a property `log_det_regularization_matrix_term` to perform this calculation:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/abstract.py


__Fit__

 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/tracer.py
 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/imaging/fit.py
"""

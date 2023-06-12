"""
__Log Likelihood Function: Inversion (pix.VoronoiMagnification + reg.Constant)__

The `inversion` script accompanying this one provides a step-by-step guide of the **PyAutoLens** `log_likelihood_function`
which is used to fit `Imaging` data with an inversion (specifically a `VoronoiMagnification` pixelization and `Constant`
regularization scheme`).

This script provides a contributor guide, that gives links to every part of the source-code that performs a LH
evaluation.

This gives contributors a linear run through of what functions, modules and packages in the source code are called
when the likelihood is evaluated, and should help them navigate the source code.

__Source Code__

__LH Setup: Lens Galaxy Light (Setup)__

To see examples of all light profiles in **PyAutoLens** checkout  the `light_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/light_profiles


__LH Setup: Lens Galaxy Mass__

To see examples of all mass profiles in **PyAutoLens** checkout the `mass_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/master/autogalaxy/profiles/mass_profiles


__LH Setup: Lens Galaxy__

The galaxy package and module contains the `Galaxy` object:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/master/autogalaxy/galaxy


__LH Setup: Source Galaxy Pixelization and Regularization__

To see examples of all pixelizations and regularization schemes in **PyAutoLens** checkout
the `Pixelization`  packages:

 https://github.com/Jammy2211/PyAutoArray/tree/master/autoarray/inversion/pixelizations
 https://github.com/Jammy2211/PyAutoArray/tree/master/autoarray/inversion/regularization


__LH Step 1: Lens Light__

- See the method `blurring_from`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/mask/mask.py
- See the method `blurring_grid_from`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/two_d/grid.py


__LH Step 3: Source Pixel Centre Calculation__

Checkout the functions `Grid2DSparse.__init__` and `Grid2DSparse.from_grid_and_unmasked_2d_grid_shape`

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/two_d/grid.py


__LH Step 2: Lens Light Convolution__

 This uses the methods in `Convolver.__init__` and `Convolver.convolve_image`

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 4: Ray Tracing__

 Ray tracing is handled in `lens` package and `ray_tracing` module:

 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/ray_tracing.py


__LH Step 5: Border Relocation__

 Checkout the following for a description of the border calculation:

 - `border_slim`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/mask/mask.py
 - `border_native`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/mask/mask.py
 - `sub_border_slim`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/mask/mask.py
 - `sub_border_grid`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/uniform_2d.py

 Checkout the function `relocated_grid_from` for a full description of the method:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/two_d/abstract_grid.py


__LH Step 6: Voronoi Mesh__

 Checkout `Mesh2DVoronoi.__init__` and `Mesh2DVoronoi.voronoi` property for a full description:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/grid_pixelization.py


__LH Step 7: Image-Source Mapping__

 Checkout the modules below for a full description of a `Mapper` and the `mapping_matrix`:

 https://github.com/Jammy2211/PyAutoArray/tree/master/autoarray/inversion/mappers
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/abstract.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/voronoi.py
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/mapper_util.py


__LH Step 8: Mapping Matrix__

 `Mapper.__init__`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/abstract.py
 `mapping_matrix_from`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/mapper_util.py


__LH Step 9: Blurred Mapping Matrix (f)__

 This uses the methods in `Convolver.__init__` and `Convolver.convolve_mapping_matrix` (it here where our
 use of real space convolution can exploit sparcity to speed up the convolution compared to an FFT):

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 10: Data Vector (D)__

The calculation is performed by the method `data_vector_via_blurred_mapping_matrix_from` at:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/leq_util.py

This function is called by `LEqMapping.data_vector_from()` to make the `data_vector`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/imaging.py


__LH Step 11: Curvature Matrix (F)__

 The calculation is performed by the method `curvature_matrix_via_mapping_matrix_from` at:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/leq_util.py

 This function is called by `LEqMapping.curvature_matrix` to make the `curvature_matrix`:

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/leq/imaging.py


__LH Step 12: Regularization Matrix (H)__

 A complete description of regularization is at the link below.

 https://github.com/Jammy2211/PyAutoArray/tree/master/autoarray/inversion/regularization
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


__LH Step 15 Image Reconstruction__

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

 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/ray_tracing.py
 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/imaging/fit.py
"""

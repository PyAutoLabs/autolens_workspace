"""
__Log Likelihood Function: Light Profile__

The `light_profile` script accompanying this one provides a step-by-step guide of the log_likelihood_function used to
fit `Imaging` data with a parametric light profile.

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

These coordinate transforms are performed in the following modules:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/geometry/geometry_util.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/decorators/transform.py


__LH Setup: Lens Galaxy Mass__

To see examples of all mass profiles in **PyAutoLens** checkout the `mass_profiles` package:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/profiles/mass_profiles


Each light profile has an `deflections_yx_2d_from` method that returns the deflection angles of the profile,
which is used in the likelihood function example.

Each function uses a `@aa.grid_dec.transform` decorator, which performs the coordinate transformation of the
grid to elliptical coordinates via the mass profile's geometry.


__LH Setup: Lens Galaxy__

The galaxy package and module contains the `Galaxy` object:

 https://github.com/Jammy2211/PyAutoGalaxy/tree/main/autogalaxy/galaxy

The `Galaxy` object also has an `image_2d_from` method and `deflections_yx_2d_from` method that returns the image and
deflection angles of the galaxy, which call the `image_2d_from` function of each light profile, and the
`deflections_yx_2d_from` function of each mass profile and sums them.


__LH Step 1: Lens Light__

This step calls the `image_2d_from` method of the `Galaxy` object, as described above.

The input to this function is the masked dataset's `Grid2D` object, which is computed here:

- See the method `grid`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/abstract/dataset.py

The calculation also used a `blurring_grid` to evaluate light which the PSF convolution blurred into the mask,
which is computed here:

- See the method `blurring_grid`: https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/imaging/dataset.py



__LH Step 2: Ray Tracing__

Ray tracing is handled in `lens` package and `tracer` module:

https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/tracer.py


__LH Step 3: Source Image__

This uses the same `image_2d_from` method as the lens galaxy light profile, but for the source galaxy.


__Likelihood Step 4: Lens + Source Light Addition__

This step is just a simple addition of the lens and source light images.



__LH Step 5: Convolution__

Convlution uses the `Convolver` object and its method `convolve_image`

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py


__LH Step 6: Likelihood Function__

The `model_image` was subtracted from the observed image, and the residuals, chi-squared
and log likelihood computed.

This is performed in the `FitImaging` object:

https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/imaging/fit_imaging.py

The following methods are relevant in this module:

`blurred_image`: Computes the blurred image via convolution described above.
`model_data`: This is the blurred image, but the variable is renamed as for more advanced fits it is extended.

The steps of subtracting the model image from the observed image and computing the residuals, chi-squared and
log likelihood are performed in the following `FitDataset` and `FitImaging` objects::

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_dataset.py
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_imaging.py

Many calculations occur in `fit_util.py`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_util.py

These modules also ihnclude the calcualtion of the `noise_normalization`.




__Fit__

 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/lens/tracer.py
 https://github.com/Jammy2211/PyAutoLens/blob/main/autolens/imaging/fit.py
"""

"""
Features: Linear Light Profiles
===============================

Linear light profiles simplify lens modeling by removing each light profile's intensity from the parameters
explored by the non-linear search. Instead, the best-fitting intensities are solved together at every
likelihood evaluation, conditional on the current lens model and light-profile shapes.

Solved does not mean fixed: the brightnesses change as the model changes. The search can concentrate on
quantities such as galaxy size, shape and lens mass without separately exploring the correlated intensity
directions. This can make fitting more efficient, especially for models with many light components, although
the underlying physical degeneracies remain.

__Scientific Use Cases__

Use linear profiles when fitting foreground lens light and parametric source light, or when combining many
components in a flexible light model. They are particularly useful for multi-Gaussian expansions, where
sampling a separate intensity for every Gaussian would enlarge the non-linear search substantially.

For lens-light subtraction and source science, the component brightnesses must respond to changes in both
the light profiles and the lensing geometry. Solving them together provides that adjustment at every trial
model. It does not establish that the chosen components are a physically unique decomposition of the data.

![Brightness sampled versus brightness solved](../../../../images/features/linear_light_profiles.png)

Conceptual illustration: both panels represent the same lens and source. The search diagrams illustrate
removing brightness from the sampled parameters; they are not measured likelihood surfaces or a timing benchmark.

__Solved at Every Step, Not Fixed in Advance__

For ordinary light profiles, intensity is a parameter sampled alongside properties such as effective radius
and Sersic index. For linear profiles, the search proposes the remaining parameters and the fit solves the
intensities that best match the data under the adopted constraints. The process repeats at the next proposal.

The example fits linear Sersic light profiles for both the lens and source. Two intensity parameters therefore
leave the non-linear search. Their values still depend on the proposed shapes, lens mass and the other model
components: the light has not been assigned a single brightness before fitting begins.

Removing an explicit search direction does not remove the information limits in the data. Brightness and size
can still trade off across the fitted models, and uncertain lensing geometry can still affect source properties.

__What Is Linear About It?__

Once the profile geometry and lens model are specified, changing a profile's intensity scales its image.
Several profiles contribute a weighted sum of images. In imaging fits, those images include the appropriate
lensing and PSF convolution before their intensities are solved using the data and noise weights.

This is the linear part of the calculation. Centres, shapes, sizes and mass parameters generally remain
non-linear because changing them changes the component images themselves. Recomputing those images and solving
new intensities are both part of each likelihood evaluation.

__Non-Negative Intensities__

An unconstrained linear solve can assign negative coefficients, allowing one light component to cancel another.
For these physical light-profile components, PyAutoLens uses a non-negative solver: intensities may be positive
or zero, but not negative. A component that is not needed can therefore receive zero intensity.

This constraint prevents negative component light; it does not guarantee that the adopted profile family
accurately describes the galaxy. Inspect residuals and individual components as well as the overall fit.

__Search Efficiency and Computational Cost__

The benefit is a smaller non-linear search, with the intensity adjustment handled conditionally at every step.
The cost is an additional linear calculation. Total run time depends on the number of components, image size,
implementation and how difficult the remaining search is; there is no universal speed-up or fixed likelihood
slow-down factor for every model.

Compare complete fits when evaluating efficiency. A slightly slower likelihood can still lead to a faster
analysis if the search needs fewer evaluations. Conversely, adding many poorly constrained components is not
made scientifically useful merely by solving their intensities linearly.

__Inspecting the Results__

Solved intensities are obtained from the fit rather than read as ordinary sampled intensity parameters.
Start with the direct-fit example to see how to retrieve them and construct profiles for visualization, then
follow the modeling example to inspect what remains in the non-linear model.

An intensity from the best-fitting model is conditional on that model. It is not, by itself, a full uncertainty
estimate for luminosity. For derived light measurements, account for uncertainty in the model geometry and
use an uncertainty treatment appropriate to the linear solution.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/fit.ipynb) — Specify the non-linear model properties, solve the intensities and inspect the resulting fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/modeling.ipynb) — Fit lens and source profiles while solving their intensities at every likelihood evaluation.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/likelihood_function.ipynb) — Follow the component-image calculation, linear system and non-negative solution.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/slam.ipynb) — Use linear light profiles within staged source, light and mass modeling.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the
workspace root.
"""

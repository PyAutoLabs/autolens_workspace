"""
Features: Sky Background
========================

Fit the sky background alongside the lens and source light so that an uncertain background level is not
mistaken for faint galaxy emission. This matters especially when measuring the extended outskirts of the
foreground galaxy, where a small sky offset can change its inferred size, profile shape and total light.

__Scientific Use Cases__

- **Measure faint galaxy outskirts:** allow uncertainty in the sky level to contribute to uncertainty in the
  extended light profile, including quantities such as effective radius and Sersic index when those are fitted.
- **Account for imperfect sky subtraction:** fit a residual offset left after image reduction, using a prior
  appropriate to the already-subtracted data.
- **Fit imaging with a retained sky level:** include the background explicitly, as demonstrated by the simulator
  in this folder, rather than asking the galaxy light model to explain it.

In a strong-lens fit, the foreground-light decomposition also affects the light assigned to the background
source. The sky, foreground galaxy and source should therefore be considered together.

![Galaxy light and a separately fitted sky level](../../../../../images/features/sky_background.png)

Conceptual illustration: the right panel separates the galaxy emission from an additive sky component.
The floating layers are a diagram of the model, not physical planes along the line of sight. Fitting or
subtracting the mean sky level does not remove the noise contributed by sky photons.

__Why the Outskirts Are Sensitive__

Bright central emission is usually easier to distinguish from the background than a faint outer envelope.
A sky level that is too low can leave light for the galaxy model to absorb; a level that is too high can
remove genuine extended emission. These changes can alter the outer profile even when the central fit looks good.

The degeneracy works both ways. A flexible light model can absorb an incorrect sky, while a freely fitted sky
can absorb real low-surface-brightness emission. Joint fitting exposes this trade-off within the adopted model;
it does not guarantee that a small cutout contains enough information to separate the components.

__What the Example Fits__

The example uses one spatially constant background level. It is a dataset component, separate from the lens
and source galaxies, and changes the additive level against which their predicted light is compared.

The simulator deliberately retains a background of 5.0 electrons per second instead of subtracting it. It
also includes the sky when generating the noise map. The direct-fit example supplies a known sky level;
the modeling example infers it alongside the lens and source model.

A constant level cannot describe a spatial gradient, scattered-light structure or contaminating galaxy.
Inspect the background across the fitted region before deciding whether this approximation suits your image.

__Choose the Prior for Your Data__

Keep the data units and reduction history explicit. A retained positive sky level and a residual offset after
sky subtraction are different quantities. A residual can have either sign, depending on whether the reduction
under- or over-subtracted the background.

The current modeling example uses a uniform prior from 0.0 to 5.0 electrons per second, putting the simulated
value at its upper boundary. Treat this as a demonstration setting, not a recommended prior: choose bounds
from your own sky estimates and uncertainties, and inspect whether the posterior accumulates at a boundary.

__Use the Available Background Information__

The mask and field of view determine which pixels constrain the sky. A region dominated by galaxy emission
may provide little independent background information. Use suitable background measurements to inform the
prior and test whether reasonable changes to the mask, sky assumptions or light model change your conclusions.

Inspect residuals across the image, including the outskirts, rather than judging only the bright centre.
Uncertainty in the fitted mean level and pixel noise are distinct: retain an appropriate noise map even when
the best-fitting background has been removed for visualization.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/sky_background/fit.ipynb) — Supply a sky level and inspect its effect on the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/sky_background/modeling.ipynb) — Infer the sky level alongside the lens and source model.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/sky_background/simulator.ipynb) — Generate imaging with a retained sky background and its noise contribution.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""

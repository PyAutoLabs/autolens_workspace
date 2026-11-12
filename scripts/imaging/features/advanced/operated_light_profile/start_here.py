"""
Features: Operated Light Profile
================================

Separate compact nuclear emission from the extended light of a lens galaxy. A bright unresolved nucleus can
appear as a PSF-shaped feature spread across several pixels, overlapping the galaxy light that you need to
model accurately. Operated light profiles describe that observed compact component directly, while ordinary
extended light profiles continue to be convolved with the point-spread function (PSF).

__Scientific Use Cases__

Use an operated profile when an appropriate model of the observed compact emission helps separate it from
the host galaxy: for example, when a bright nuclear component would otherwise be absorbed into the extended
light profile. This can improve the description of foreground light used in a lens fit and help assess how
its decomposition affects the inferred lens and source.

The worked example places compact emission at the centre of the foreground lens galaxy. It models that
component with an operated Gaussian alongside extended lens light and a lensed background source. This is
not a complete workflow for multiply imaged quasars or a method for resolving structure below the PSF scale.

![Compact nuclear emission and extended galaxy light](../../../../../images/features/operated_light_profile.png)

Conceptual illustration: the left panel shows a schematic Gaussian PSF. The compact component in the right
panel already includes that blur; it is not convolved with the PSF a second time. This does not recover
intrinsic nuclear structure or improve angular resolution.

__What Does Operated Mean?__

An ordinary light profile describes intrinsic emission. For imaging, the model is convolved with the PSF
before comparison with the observed pixels. An operated profile is defined after that operation: its shape
already represents the observed, PSF-blurred light.

The distinction is about what the parameters mean, not simply whether a component is small. A narrow intrinsic
Gaussian still needs the appropriate convolution. An operated Gaussian instead describes an already blurred
component and bypasses that convolution. Its width must be interpreted accordingly.

__Convolve Each Component Once__

The compact and extended components can coexist in one model. The extended galaxy and lensed source undergo
the normal imaging response; the operated component is added in its already operated form. Convolving that
component again would broaden it twice and change the predicted central surface brightness.

By describing the observed footprint directly, an operated profile can also avoid representing an unresolved
source as an extremely narrow intrinsic profile on the image grid before convolution. It is not a replacement
for checking image sampling or for using an appropriate PSF model.

__Separate the Nucleus and Host__

The example uses a Gaussian for the compact component and shares its centre with the lens galaxy's extended
light profile. This is a specific assumption about the nucleus and host, not a requirement that every compact
component in real data must be concentric with a galaxy.

The script also shows the linear-operated form, in which the compact component's intensity is solved during
each likelihood evaluation. These are separate choices: operated determines whether the profile already
includes the instrumental operation, while linear determines how its intensity is fitted.

Inspect the compact component, extended light and residuals individually. A plausible total image can still
hide an incorrect allocation of central light. Test whether the host model is absorbing PSF mismatch or whether
the compact component is compensating for an inadequate extended profile.

__The Adopted PSF Shape Matters__

An operated Gaussian is an analytic description of the observed light, not automatic knowledge of the true
telescope PSF. Its suitability depends on how well that shape captures the compact emission, including any
important wings or asymmetry. Other profile choices require the same scientific justification.

An unresolved component's fitted width should not be reported as an intrinsic physical size without accounting
for the instrumental response. Likewise, uncertainty in its shape and normalization can affect the inferred
host light. Use an appropriate PSF characterization and assess the stability of the decomposition when those
choices matter for the scientific result.

__Reading the Examples__

Read the simulator alongside the modeling script to see how the compact and extended components enter the
mock image and the fit. Check which components are operated, which are linear, and which still receive PSF
convolution. The simulation provides a controlled example, not evidence that a Gaussian is adequate for all data.

This introduction concerns imaging. Other data types have different instrumental operations; bypassing imaging
PSF convolution does not mean that an arbitrary component can bypass every response in an interferometer model.

__Choose An Example__

- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/operated_light_profile/modeling.ipynb) — Fit compact foreground emission alongside extended lens light and the lensed source, including the linear-operated construction.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/operated_light_profile/simulator.ipynb) — Inspect how the example represents compact and extended light in a mock imaging dataset.

There is no separate direct-fit script in this folder. For that workflow, use an imaging fit example and adapt
the profile construction as demonstrated in the modeling script.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the
workspace root.
"""

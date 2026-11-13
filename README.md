# PyAutoLens Workspace

<img src="https://github.com/Jammy2211/PyAutoLogo/blob/main/gifs/pyautolens.gif?raw=true" width="900" />

Example scripts and notebooks for modeling strong gravitational lenses with **PyAutoLens**. Start with an introductory example, then explore the tools for your own data.

## Getting Started

### Explore It Yourself

- **[Try it in your browser](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.11.13.1/start_here.ipynb)** — run the introductory notebook on Google Colab without installing anything.
- **[Run it locally](https://pyautolens.readthedocs.io/en/latest/installation/overview.html)** — follow the installation guide, then open `start_here.ipynb`.
- **[Browse examples with images](https://github.com/PyAutoLabs/autolens_workspace/blob/main/markdown/README.md)** — read worked examples and view their results directly on GitHub.

See [Find Your Examples](#find-your-examples) below to choose where to start.

### Explore with AI

The [**PyAutoLens AI Assistant**](https://github.com/PyAutoLabs/autolens_assistant) helps you learn lensing, run examples and analyse your own data through conversation. Visit its README for setup instructions and a starting prompt.

## Find Your Examples

Examples are available as Python scripts in `scripts/` and Jupyter notebooks in `notebooks/`.

### Lens Scale

![Galaxy, multi-galaxy, group and cluster lenses, with indicative Einstein radii.](https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_2/lens_scales.png)

For a single lens galaxy, choose your data type below. For larger systems, start with [multi_galaxy/](scripts/multi_galaxy), [group/](scripts/group) or [cluster/](scripts/cluster).

### Dataset Type

Choose the example for your observations. Each folder has a `start_here` example to get you going.

<table>
<tr>
<td colspan="2" align="center">
<strong>Imaging</strong><br>
<a href="scripts/imaging"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_2/imaging.png" alt="A pixelated telescope image of a foreground galaxy surrounded by extended lensed arcs." width="440"></a><br>
Fit images from Hubble, JWST and other telescopes with <a href="scripts/imaging">imaging/</a>.
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>Interferometry</strong><br>
<a href="scripts/interferometer"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_2/interferometry.png" alt="Radio dishes observing emission from a lensed galaxy." width="440"></a><br>
Fit radio and submillimetre observations with <a href="scripts/interferometer">interferometer/</a>.
</td>
<td width="50%" valign="top">
<strong>Multi-band Imaging</strong><br>
<a href="scripts/multi_dataset"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_2/multi_wavelength.png" alt="The same lens in three wavebands, with different relative brightness of the lens galaxy and arcs." width="440"></a><br>
Fit several wavebands together with <a href="scripts/multi_dataset">multi_dataset/</a>.
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>Data Cubes</strong><br>
<a href="scripts/interferometer/features/datacube"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_2/data_cubes.png" alt="Separate spectral-channel images showing different parts of a lensed arc." width="440"></a><br>
Fit spectral channels together with <a href="scripts/interferometer/features/datacube">interferometer/features/datacube/</a>.
</td>
<td width="50%" valign="top">
<strong>Point Sources</strong><br>
<a href="scripts/point_source"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_2/point_sources.png" alt="Four compact lensed images around a foreground galaxy, with faint host-galaxy arcs." width="440"></a><br>
Model lensed quasars and supernovae with <a href="scripts/point_source">point_source/</a>.
</td>
</tr>
</table>

### Weak Lensing

![Strong-lensing arcs near a cluster centre and weakly distorted background galaxies across a wider field.](https://raw.githubusercontent.com/PyAutoLabs/PyAutoLens/main/docs/overview/images/overview_3/strong_weak_lensing.png)

To model the subtle distortions of many background galaxies, start with [weak/](scripts/weak). You can also combine weak and strong lensing to study a cluster across both scales.

*AI-generated illustrations, not observations. Lens panels use different angular scales; dashed contours represent a model. See the [new user guide](https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html) for details.*

### Browse All Examples

| I want to… | Start here |
|---|---|
| Learn the API, customise models or inspect results | [Guides](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/guides) |
| Model a lens in imaging data | [Imaging](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/imaging) |
| Model interferometer observations | [Interferometry](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/interferometer) |
| Fit observations in multiple wavebands | [Multiple datasets](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/multi_dataset) |
| Model lensed point sources | [Point sources](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/point_source) |
| Model a lens with several main galaxies | [Multiple galaxies](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/multi_galaxy) |
| Model a galaxy group | [Groups](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/group) |
| Model a galaxy cluster | [Clusters](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/cluster) |
| Explore weak lensing | [Weak lensing](https://github.com/PyAutoLabs/autolens_workspace/tree/main/scripts/weak) |

Each folder has a README explaining its examples. You’ll find sample data in `dataset/`; analysis results are written to `output/`.

## Learn Lensing

[**HowToLens**](https://github.com/PyAutoLabs/HowToLens) teaches gravitational lensing and Bayesian inference through worked notebooks. Start here if you’re new to the subject or would like more explanation alongside the workspace examples.

## Community and Contributing

Ask questions, suggest improvements and share your work in [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions), or chat with us on [Slack](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg).

To contribute examples or explore community projects, visit the [PyAutoLens community page](https://pyautolens.readthedocs.io/en/latest/general/community.html).

## Citations

If you use PyAutoLens in your research, see the [citation guide](https://github.com/PyAutoLabs/PyAutoLens/blob/main/CITATIONS.md).

## License

See the [PyAutoLens license](https://github.com/PyAutoLabs/PyAutoLens/blob/main/LICENSE).

"""
Multi Dataset Features
======================

These features build on the topic’s main example. Each introduction explains the method and its trade-offs; the worked scripts retain their model choices and detailed calculations.

__Choose a Feature__

- [Dataset Offsets](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/start_here.ipynb): Observations of the same object may have small relative coordinate offsets. Joint modeling can account for those offsets while fitting the physical model shared by the datasets.
- [Imaging And Interferometer](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/start_here.ipynb): Imaging and interferometer observations can be fitted together using the parameters that their physical models share. Each dataset still requires the forward calculation appropriate to its measurement.
- [Imaging And Point Source](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_point_source/start_here.ipynb): Extended arcs and point-source measurements provide different constraints on a shared lens model. A joint analysis connects them through the mass distribution while retaining the appropriate likelihood for each observable.
- [One By One](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/start_here.ipynb): Multi-dataset modeling does not always require fitting every observation simultaneously. A sequential approach fits one dataset first, then uses the result to guide the analysis of another.
- [Pixelization](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/pixelization/start_here.ipynb): A pixelization represents light using a mesh whose brightness values are solved during the fit. It provides a flexible alternative to describing all of the emission with a small collection of analytic profiles.
- [Same Wavelength](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/start_here.ipynb): Repeated observations at the same wavelength can constrain a common light model while retaining their individual instrumental responses and noise properties.
- [Slam](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/slam/start_here.ipynb): SLaM organizes lens modeling into linked source, light and mass stages. Results from earlier stages inform the setup of later stages, allowing the model to be developed in a controlled sequence.
- [Wavelength Dependence](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/start_here.ipynb): A wavelength-dependent model describes how specified galaxy properties change across bands. It links observations through a chosen relationship rather than treating every parameter in every band independently.

For the common setup, begin with [multi_dataset/start_here.ipynb](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/multi_dataset/start_here.ipynb). Run Python scripts from the workspace root.
"""

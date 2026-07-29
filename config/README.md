The `config` folder contains configuration files which customize default **PyAutoLens**.

# Folders

- `non_linear`: Configs for default non-linear search (e.g. MCMC, nested sampling) settings.
- `priors`: Configs defining default priors assumed on every model component and set of parameters.
- `visualize`: Configs defining what images are output by a model fit.
- `build`: Configs used by the automated build and test system (not relevant to normal use).

# Files

- `general.yaml`: Customizes general **PyAutoLens** settings.
- `logging.yaml`: Customizes the logging behaviour of **PyAutoLens**.
- `notation.yaml`: Defines labels and formatting of model parameters when used for visualization.
- `output.yaml`: Customizes what a model-fit writes to the output folder.
- `latent.yaml`: Configures latent variables derived from a model-fit.

# %%
"""
HPC: Example CPU and GPU
========================

This example illustrates how to set up lens modeling on a High Performance Computing (HPC) system, using either
multiple CPUs or a GPU.

It illustrates two different forms of parallelization:

1) Set off many single CPU jobs in a single HPC submission script, where each job fits a different dataset using the
same lens model analysis. This form of parallelization is efficient when we have many datasets we wish to fit
simultaneously, but each individual fit only uses one CPU so overall run times are slower.

2) Fit a single dataset using a parallelized Nautilus model-fit, where the non-linear search distributes the model-fit
over multiple CPUs. This form of parallelization is efficient when we have a single dataset to fit, but we wish to
speed up the overall run time of the model-fit by using multiple CPUs. However, parallelizing over multiple CPUs
have communication overheads, and so this form of parallelization is less efficient than fitting many single CPU jobs.

The example assumes the HPC environment uses SLURM for job management, which is standard for many academic HPCs but
may not necessarily be the case for your HPC. If your HPC does not use SLURM, you should still be able to adapt this
example to your HPC's job management system.

This example will likely require adaptation for you to run it on your HPC environment, its goal is to simply
illustrate the general principles of how to set up lens modeling on an HPC.

The SLURM batch scripts referenced throughout this example are in:

  scripts/guides/hpc/batch_cpu/submit   -- CPU array job (many datasets or parallel fit)
  scripts/guides/hpc/batch_gpu/submit   -- GPU array job (same pattern, GPU partition)

See README.md in this folder for the full setup guide, including how to transfer files
using the sync script.
"""

# %%
"""
__Contents__

- **HPC Output Path:** Set the output path for results on the HPC.
- **HPC Dataset Path:** Set the path where datasets are stored on the HPC.
- **HPC Home Path:** Set the home directory path and config path on the HPC.
- **Batch Script: Many Datasets (Array Job):** Configure a SLURM array job to fit many lenses in parallel.
- **Dataset:** Load the dataset this job has been asked to fit.
- **Batch Script: Single Dataset with Multiple CPUs:** Configure a parallelized single-dataset Nautilus fit.
- **Nautilus CPUs:** Configure Nautilus to use multiple CPU cores for parallelized sampling.
- **Lens Modeling:** Perform a standard lens model-fit on the HPC.
- **GPU Jobs:** Run GPU-accelerated model-fits using the GPU batch script.

__HPC Output Path__

We first set the `hpc_output_path`, where the results of lens modeling are output on your HPC.

On certain HPCs this may be different from your home directory or where you store data, because lens modeling has more
IO and outputs many individual files.

This example assumes results are output to the directory, where `hpc_username` is your hpc username:

 `/hpc/data/hpc_username/output`.

You will need to update `hpc_username` to your hpc username below.
"""
from pathlib import Path

hpc_output_path = Path("/") / "hpc" / "data" / "hpc_username" / "output"

"""
__HPC Dataset Path__

We next set the `hpc_dataset_path`, which is the root folder where datasets are stored on the hpc.

This may be the same as your output path, or you may have been advised to store datasets in a different location,
especially if they are large in file size.

We therefore define it separately from the `hpc_output_path`.

Note that this is the *root* dataset folder, not the folder of any one dataset. The specific dataset this job fits
is appended to it further below, once the batch script has told us which dataset to load. Below, we set
`hpc_dataset_path=/hpc/data/hpc_username/dataset`.
"""
hpc_dataset_path = Path("/") / "hpc" / "data" / "hpc_username" / "dataset"

"""
__HPC Home Path__

The `home_path` is in your the hpc home directory, which again may be different from your output and dataset paths.

The home path often has signficant storage restrictions, so is not a good location to store datasets or output results.
But may be where you store the python lens modeling scripts you run on the HPC, the config files, batch scripts
and other files you use to set up lens modeling on the hpc.
"""
home_path = Path("/", "hpc", "home", "hpc_username")

"""
On the HPC, most likely in your home directory, you should have a config folder which contains the config files used by
modeling.

This `config_path` sets the path to the config files that are used in this analysis, which are contained within the `hpc`
directory of the example project in your the hpc home directory.
"""
config_path = Path(home_path, "hpc", "config")

"""
Set the config and output paths using autonerves, as you would for a laptop run.
"""
from autolens import conf

try:
    conf.instance.push(new_path=config_path, output_path=hpc_output_path)
except Exception as e:
    print(
        "The config and output paths above need to be updated to match the paths on your HPC. Please update the code and try again."
    )

"""
Above, we set up many different paths required to run modeling on the hpc. You should basically determine where
all the different paths are on your HPC are which correspond to the paths above, and update the code accordingly.

__Batch Script: Many Datasets (Array Job)__

HPC submissions require a batch script, which tells SLURM the hardware resources you need and the Python
script to execute. SLURM then schedules the job on available nodes.

The batch script for fitting many lenses is at:

  scripts/guides/hpc/batch_cpu/submit

The modern approach uses a SLURM array job, where a single sbatch submission launches one independent job
per dataset. The key directives are:

 `#SBATCH --cpus-per-task=8` - Number of CPU cores per job. Increase this to speed up Nautilus sampling.
 `#SBATCH --mem=64gb` - Memory per job. Increase for large pixelized source reconstructions.
 `#SBATCH -t 18:00:00` - Wall-clock time limit; the job is killed if it overruns.
 `#SBATCH --array=0-0` - Launch one job per dataset. SLURM sets $SLURM_ARRAY_TASK_ID to a different
                          value in each job. The shipped script fits the single example dataset, so it
                          uses `0-0`; for three lenses you would write `--array=0-2`.
 `#SBATCH -o output/output.%A_%a.out` - Stdout log, named with the job ID and array index.
 `#SBATCH -e error/error.%A_%a.err` - Stderr log, one file per array task.

Before submitting, set PROJECT_PATH so the batch script can find the workspace:

    export PROJECT_PATH=/path/to/autolens_workspace
    sbatch scripts/guides/hpc/batch_cpu/submit

The following line activates the PyAutoLens virtual environment for this run:

    source $PROJECT_PATH/activate.sh

These lines tell JAX to use the CPU backend (not attempt to use a GPU that was not allocated):

    export JAX_PLATFORM_NAME=cpu
    export JAX_PLATFORMS=cpu

These lines prevent NumPy linear algebra libraries from spawning more threads than the allocated CPUs,
which would cause slowdowns when many array jobs share the same node:

    THREADS=$SLURM_CPUS_PER_TASK
    export OPENBLAS_NUM_THREADS=$THREADS
    export MKL_NUM_THREADS=$THREADS
    ...

The dataset is selected using the array index:

    datasets=(dataset_0 dataset_1 dataset_2)
    dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
    python3 $PROJECT_PATH/scripts/guides/hpc/example_cpu_and_gpu.py --dataset=$dataset

Each array job is fully independent — it loads a different dataset and writes to a separate output path.
Adding more datasets to the list and updating --array is all that is required to scale to larger samples.

The dataset name chosen by the batch script arrives here as the `--dataset` argument, which we read below
with argparse. We use `parse_known_args` rather than `parse_args` so this script also runs as a notebook,
where the Jupyter kernel adds command line arguments of its own.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="simple__no_lens_light",
    help="Name of the dataset subdirectory to fit.",
)
args, _ = parser.parse_known_args()

import autofit as af
import autolens as al

"""
__Dataset__

The dataset name passed from the batch script selects which dataset this job fits. Each array job receives a
different value, so all lenses are fitted in parallel.
"""
dataset_type = "imaging"
dataset_name = args.dataset

"""
We now create the path to the dataset this specific job fits, which for `--dataset=simple__no_lens_light` is:

 `/hpc/data/hpc_username/dataset/imaging/simple__no_lens_light`

If that path does not exist we are not running on the HPC, so we fall back to the `dataset` folder of your local
workspace. This means you can run this script end-to-end on your laptop before submitting it to the HPC.
"""
dataset_path = Path(hpc_dataset_path, dataset_type, dataset_name)

if not dataset_path.exists():
    dataset_path = Path(Path.cwd(), "dataset", dataset_type, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
You now have all the code you need to set off many single-CPU jobs on the hpc!

You would simply append the batch scripts and Python code above to the lens modeling script you are using,
which is given below for completeness.

However, first we describe how to set up a single multi-CPU Nautilus job on the hpc.

__Batch Script: Single Dataset with Multiple CPUs__

The same batch script (scripts/guides/hpc/batch_cpu/submit) handles this case by leaving --array=0-0
(a single-element array) and increasing --cpus-per-task. The dataset list then contains only one entry.

The key difference is that Nautilus is given the number of cores via `number_of_cores`, which it uses
to distribute likelihood evaluations across CPUs.

__Nautilus CPUs__

For a parallelized single-dataset run, we tell Nautilus how many CPU cores to use. This is read from
the SLURM environment variable set by --cpus-per-task in the batch script, so the Python script does
not need to be changed when adjusting the CPU count — only the batch script header changes.
"""
import os

number_of_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

search = af.Nautilus(
    path_prefix="hpc",
    name="example",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=number_of_cores,
)

"""
We now have everything we need to fit a single dataset using multiple CPUs on the hpc!

The code below performs standard lens modeling, which is unchanged from normal modeling on a laptop. It can be
used for either many single-CPU jobs or a single multi-CPU Nautilus job.

__Lens Modeling__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is a `SersicCore` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__GPU Jobs__

The Python modeling code above is identical for GPU runs — no changes are needed here.

The only difference is which batch script you submit. The GPU script at:

  scripts/guides/hpc/batch_gpu/submit

requests a GPU partition and a single GPU via `--gres=gpu:1`. JAX auto-detects the GPU and uses it
automatically because `use_jax=True` is set on the analysis object. The CPU thread-pinning and
JAX_PLATFORM_NAME overrides in the CPU batch script are omitted, and `nvidia-smi` is called at the
start of the job to confirm the GPU allocation in the log.

To submit a GPU job:

  export PROJECT_PATH=/path/to/autolens_workspace
  sbatch scripts/guides/hpc/batch_gpu/submit

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

This guide simulates its dataset on demand rather than loading committed
FITS, and runs green under either dataset regime. `full_datasets` is declared
so the example runs at the full 100x100 shape the surrounding prose assumes;
under the small-dataset cap the mask is padded to fit the PSF footprint and
the reported image-pixel count no longer matches the guide's text.

ENV: full_datasets
"""

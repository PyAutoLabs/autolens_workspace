# PyAutoLens on High-Performance Computing (HPC)

## Introduction

This guide describes how to set up and run PyAutoLens on a High-Performance
Computing (HPC) system. It assumes:

- You have access to an HPC cluster via SSH
- The cluster uses the SLURM batch scheduling system
- Python is available via system modules, Conda, or a custom installation

Although every HPC system differs slightly (filesystem layout, module names,
Python versions, storage quotas), the core steps are universal. Where
system-specific details are required, these are clearly marked and easy
to adapt.

## Overview of the HPC Workflow

On most HPC systems:

- Home directories have limited storage and are best used for:
  \- source code
  \- virtual environments
  \- configuration files
- Scratch or data filesystems are designed for:
  \- large datasets (e.g. `.fits` files)
  \- PyAutoLens output directories
  \- intermediate results

This guide follows best practice by separating:

- the PyAutoLens workspace
- the data and output directories

## PyAutoLens Virtual Environment

Before running PyAutoLens on an HPC system, you should create a Python
virtual environment in your home directory. This environment will contain
PyAutoLens and all its dependencies.

```{eval-rst}
.. note::

   Throughout this guide, replace values written in ALL CAPS (e.g.
   ``YOUR_USERNAME`` or ``HPC_LOGIN_HOST``) with the appropriate values for
   your system.
```

## Installing PyAutoLens: Available Options

There are two supported ways to install PyAutoLens on an HPC system.

### Option 1 (Recommended): Install via `pip` or `conda`

- Simple and robust
- Ideal if you do not need to modify PyAutoLens source code

### Option 2: Clone the GitHub repositories

- Useful if you are developing PyAutoLens itself
- Follow the instructions in `README_Repos.md` instead of this guide

This document assumes **Option 1**.

## Step-by-Step Guide (Installation via `pip`)

### 1. SSH into the HPC

From your local machine:

```
ssh -X YOUR_USERNAME@HPC_LOGIN_HOST
```

You should now be logged into your home directory on the HPC system.

### 2. (Optional) Confirm Your Location

```
pwd
```

This should point to something like:

```
/home/YOUR_USERNAME
```

### 3. Create a Python Virtual Environment

```
python3 -m venv PyAuto
```

This creates a virtual environment called `PyAuto` in your home directory.

```{eval-rst}
.. note::

   You may rename this directory if you wish, but you must update paths
   consistently later in this guide.
```

### 4. Create an Activation Script

Create a helper script that loads required modules (if applicable),
activates the virtual environment, and sets environment variables.

```
nano activate.sh
```

(Use `emacs -nw activate.sh` or `vi activate.sh` if you prefer.)

### 5. Edit `activate.sh`

Paste the following template and adapt it to your system:

```
#!/bin/bash

# Reset loaded modules (optional but recommended)
module purge

# Load required modules (EDIT THESE FOR YOUR HPC)
module load python/3.X.Y

# Activate the virtual environment
source $HOME/PyAuto/bin/activate

# Ensure Python can see your workspace
export PYTHONPATH=$HOME:$HOME/PyAuto
```

Make the script executable:

```
chmod +x activate.sh
```

```{eval-rst}
.. note::

   - Some HPC systems do not use environment modules. If so, remove the
     ``module`` lines.
   - If you use Conda instead of ``venv``, activate your Conda environment
     here instead.
```

### 6. Activate the Environment

```
source activate.sh
```

You should now see `(PyAuto)` at the start of your command prompt.

### 7. Install PyAutoLens

```
pip install autolens
```

This installs PyAutoLens and all required dependencies into the virtual
environment.

### 8. (Optional) Auto-Activate on Login

If you primarily use PyAutoLens on this HPC, you can automatically activate
the environment when logging in.

Edit your shell configuration file:

```
nano ~/.bashrc
```

Add:

```
source $HOME/activate.sh
```

## PyAutoLens Workspace

Your workspace mirrors the structure you use on your laptop, but excludes
large datasets and output directories.

On the HPC, this typically lives in:

```
$HOME/autolens_workspace
```

## HPC-Specific Workspace Folder

The workspace contains an `hpc` folder (`scripts/guides/hpc`) with:

- `batch_cpu/` — SLURM submission script for CPU jobs, plus its `output/` and
  `error/` log folders
- `batch_gpu/` — SLURM submission script for GPU jobs, plus its `output/` and
  `error/` log folders
- `sync` — bash script for transferring files to and from the HPC
- `sync.conf.example` — configuration template for the sync script
- `example_cpu_and_gpu.py` — the Python walkthrough of an HPC model-fit

## Data and Output Directories

Large datasets and PyAutoLens output should be stored on a high-capacity
filesystem, often named something like:

- `/scratch`
- `/work`
- `/data`
- `/gpfs`
- `/lustre`

Consult your HPC documentation to find the correct location.

### Example Directory Setup (On HPC)

```
cd /PATH/TO/LARGE_STORAGE/YOUR_USERNAME
mkdir -p dataset
mkdir -p output
```

## Transferring Files: Manual rsync

`rsync` transfers files between your laptop and the HPC efficiently. The
most useful flags are:

- `--update` — only copy files that are newer than the destination
- `-a` — archive mode (preserves timestamps, permissions, symlinks)
- `-z` — compress data in transit (good for scripts and config; minimal
  benefit for `.fits` files)
- `--partial` — resume interrupted transfers instead of restarting

### Uploading Code

From your local workspace root:

```
rsync --update -v -r scripts/ \
  YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO/WORKSPACE/scripts/
```

### Uploading Data

```
rsync --update -v -r dataset/ \
  YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/dataset/
```

### Uploading a Single Dataset

```
rsync --update -v -r dataset/example \
  YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/dataset/
```

### Downloading Results

```
rsync --update -v -r \
  YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/output/ \
  ./output/
```

## Transferring Files: Sync Script (Recommended)

The `sync` script in `scripts/guides/hpc/` automates the above rsync
commands and handles push, pull, and dry-run in a single tool. It is the
preferred way to transfer files once you are comfortable with the manual
rsync approach.

### Setup

Copy the example config and fill in your values:

```
cp scripts/guides/hpc/sync.conf.example scripts/guides/hpc/sync.conf
```

Edit `sync.conf`:

```
# SSH host alias (from ~/.ssh/config) or user@hostname
HPC_HOST=my_hpc

# Base directory on the HPC under which your projects sit
HPC_BASE=/scratch/YOUR_USERNAME

# Project name on the HPC (defaults to local folder name)
PROJECT_NAME=autolens_workspace
```

`sync.conf` is gitignored and stays on your local machine only.

### Usage

Run from the workspace root:

```
bash scripts/guides/hpc/sync push     # upload code, config, and data
bash scripts/guides/hpc/sync pull     # download results
bash scripts/guides/hpc/sync sync     # push then pull (default)
bash scripts/guides/hpc/sync status   # dry run — show what would transfer
```

The script uses different rsync strategies per directory type:

- **Code and config** (`scripts/`, `config/`) — changed files are overwritten,
  new files are added.
- **Dataset** — `--ignore-existing`: any `.fits` file already on the HPC
  is never re-transferred, which avoids checksumming a large archive on
  every sync.
- **Output** — `--update`: only files newer on the HPC are pulled down,
  and the large `search_internal/` sampler state is excluded automatically.

```{eval-rst}
.. note::

   ``$HPC_BASE/$PROJECT_NAME`` in ``sync.conf`` is equivalent to
   ``$PROJECT_PATH`` used inside the SLURM batch scripts, so activation
   and script paths stay consistent.
```

## Submitting Jobs: CPU

SLURM jobs are submitted with `sbatch`. Before submitting, set
`PROJECT_PATH` so the batch script can find your workspace:

```
export PROJECT_PATH=/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/autolens_workspace
sbatch scripts/guides/hpc/batch_cpu/submit
```

The example CPU script is at `scripts/guides/hpc/batch_cpu/submit`. The
key SLURM directives are:

- `--partition=cpu` — submit to the CPU partition (name varies by HPC)
- `--cpus-per-task=8` — number of CPU cores per job; increase for faster
  Nautilus sampling
- `--mem=64gb` — memory per job; increase for large pixelized source reconstructions
- `--time=18:00:00` — wall-clock time limit; job is killed if it overruns
- `--array=0-0` — launch one job per dataset; SLURM sets
  `$SLURM_ARRAY_TASK_ID` to a different value in each job. The shipped script
  fits the single example dataset, so it uses `0-0`; for three datasets you
  would write `--array=0-2`

The array index selects the dataset:

```
datasets=(dataset_0 dataset_1 dataset_2)
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
```

Add entries to `datasets=(...)` and widen `--array` to match — those two edits
are all that is needed to scale from one lens to a large sample.

This is the standard pattern for fitting many lenses simultaneously: each
array job is independent, runs on its own CPU allocation, and fits a
different dataset.

### JAX and thread pinning

Two environment variables tell JAX to use the CPU backend and prevent it
from attempting to use a GPU that is not allocated to this job:

```
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
```

The thread-pinning block sets every linear-algebra library to use exactly
the number of CPUs SLURM allocated, preventing threads from overloading
the node:

```
THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OMP_NUM_THREADS=$THREADS
...
```

## Submitting Jobs: GPU

The GPU script at `scripts/guides/hpc/batch_gpu/submit` uses the same
array-job pattern but requests a GPU instead of extra CPUs.

Key differences from the CPU script:

- `--partition=gpu` — submit to the GPU partition
- `--gres=gpu:1` — request one GPU per job
- `--cpus-per-task=1` — only one CPU core is needed; the GPU handles
  parallelism
- `--mem=32gb` — GPU jobs typically need less host memory than large
  multi-CPU jobs
- No `JAX_PLATFORM_NAME` override — JAX auto-detects and uses the GPU
- No CPU thread-pinning block — not needed when a GPU is doing the work
- `nvidia-smi` — prints GPU information to the log at job start, useful
  for confirming the correct GPU was allocated

Submit with:

```
export PROJECT_PATH=/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/autolens_workspace
sbatch scripts/guides/hpc/batch_gpu/submit
```

```{eval-rst}
.. note::

   GPU jobs use JAX with ``use_jax=True`` in the analysis object, which is
   already the default in PyAutoLens. No code changes are required to switch
   from CPU to GPU — only the batch script changes.
```

## Monitoring Jobs

After submitting, standard SLURM commands apply:

```
squeue -u YOUR_USERNAME          # list your running and pending jobs
scancel JOB_ID                   # cancel a job
sacct -j JOB_ID --format=...     # inspect a completed job
```

Output and error logs are written to `batch_cpu/output/` and
`batch_cpu/error/` (or `batch_gpu/`) with filenames containing the
job and array index, making it easy to trace a specific dataset failure.

## Next Steps

- Read `example_cpu_and_gpu.py` in this folder for a detailed Python walkthrough
  of how paths, datasets, and Nautilus are configured for HPC runs.
- Adapt `batch_cpu/submit` or `batch_gpu/submit` for your HPC partition
  names, memory requirements, and dataset list.
- Use `scripts/guides/hpc/sync` for day-to-day file transfer once you
  have filled in `sync.conf`.

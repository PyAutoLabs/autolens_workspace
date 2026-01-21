# %%
"""
HPC: Example CPU
================

This example illustrates how to set up lens modeling on a High Performance Computing (HPC) system using multiple CPUs.

It illustrates two different forms of parallelization:

1) Set off many single CPU jobs in a single HPC submission script, where each job fits a different dataset using the
same lens model analysis. This form of parallelization is efficient when we have many datasets we wish to fit
simultaneously, but each individual fit only uses one CPU so overall run times are slower.

2) Fit a single dataset using a parallelized Nautilus model-fit, where the non-linear search distributes the model-fit
over multiple CPUs. This form of parallelization is efficient when we have a single dataset to fit, but we wish to
speed up the overall run time of the model-fit by using multiple CPUs. However, parallelizing over multiple CPUs
have communication overheads, and so this form of parallelization is less efficient than fitting many single CPU jobs.

The example assumes the HPC environment uses slurm for job management, which is standard for many academic HPCs but
may not necessarily be the case for your HPC. If your HPC does not use slurm, you should still be able to adapt this
example to your HPC`s job management system.

This example will likely require adaptation for you to run it on your HPC enviroment, its goal is to simply
illustrate the general principles of how to set up lens modeling on an HPC.
"""

# %%
"""
__HPC Output Path__

We first set the `hpc_output_path`, where the results of lens modeling are output on your HPC. 
 
On certain HPCs this may be different from your home directory or where you store data, because lens modeling has more 
IO and outputs many individual files. 

This example assumes results are output to the directory, where `hpc_username` is your hpc username:

 `/hpc/data/hpc_username/output`.
 
You will need to update `hpc_username` to your hpc username below.
"""
from os import path
from pathlib import Path

hpc_output_path = Path(path.sep) / "hpc" / "data" / "hpc_username" / "output"

"""
__HPC Dataset Path__

We next set the `hpc_dataset_path`, which is the path where datasets are stored on the hpc.

This may be the same as your output path, or you may have been advised to store datasets in a different location,
especially if they are large in file size.

We therefore define it separately from the `hpc_output_path`.

Below, we set `hpc_dataset_path=/hpc/data/hpc_username/dataset/example/simple__no_lens_light`.
"""
dataset_folder = "example"
dataset_name = "simple__no_lens_light"

hpc_dataset_path = (
    Path(path.sep)
    / "hpc"
    / "data"
    / "hpc_username"
    / "dataset"
    / dataset_folder
    / dataset_name
)

"""
__HPC Home Path__

The `home_path` is in your the hpc home directory, which again may be different from your output and dataset paths.

The home path often has signficant storage restrictions, so is not a good location to store datasets or output results.
But may be where you store the python lens modeling scripts you run on the HPC, the config files, batch scripts
and other files you use to set up lens modeling on the hpc.
"""
home_path = Path(path.sep, "hpc", "home", "hpc_username")

"""
On the HPC, most likely in your home directory, you should have a config folder which contains the config files used by 
modeling.

This `config_path` sets the path to the config files that are used in this analysis, which are contained within the `hpc` 
directory of the example project in your the hpc home directory.
"""
config_path = Path(home_path, "hpc", "config")

"""
Set the config and output paths using autoconf, as you would for a laptop run.
"""
from autoconf import conf

conf.instance.push(new_path=config_path, output_path=hpc_output_path)

"""
Above, we set up many different paths required to run modeling on the hpc. You should basically determine where
all the different paths are on your HPC are which correspond to the paths above, and update the code accordingly.

__Batch Script Many Lenses__

HPC submissions require a batch script, which tells the HPC the CPU hardware you want the job to run on and the 
PyAutoLens Python script you want it to execute. This script then distributes the job to nodes and CPUs. 

Lets look at the batch script 

 `autolens_workspace/*/guides/hpc/batch/example_cpu_one_dataset_parallel
    
The following options are worth noting:

 `#SBATCH -N 1` - The number of nodes we require, where 1 node contains 28 CPUs on the hpc.
 `#SBATCH --ntasks=16` - The total number of task we are submitting.
 `#SBATCH --cpus-per-task=1` - The number of tasks per CPU.
 `#SBATCH -J example` - The name of the job, which is how it appears on hpc when you inspect it.
 `#SBATCH -o output/output.%A.out` - Python interpreter output is placed in a file in the `output` folder.
 `#SBATCH -o error/error.%A.out` - Python interpreter errors are placed in a file in the `error` folder.
 `#SBATCH -p hpc` - Signifies we are running the job on the hpc.
 `#SBATCH -A dp004` - The project code of the submission.
 `#SBATCH -t 48:00:00` - The job will terminate after this length of time (if it does not end naturally).
 `#SBATCH --mail-type=END` - If you input your email, when you`ll get an email about the job (END means once finished).
 `#SBATCH --mail-user=fill@me.co.uk` - The email address the hpc sends the email too.

The following line activates the PyAutoLens virtual environment we set up on hpc for this run:

 `source /hpc/home/hpc_username/activate.sh`

These lines prevent the NumPy linear algebra libraries from overloading the CPUs during calculations.
    
export CPUS_PER_TASK=1

export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK

This line sets off the job:

    srun -n 16 --multi-prog conf/example.conf

Lets checkout the file `example_cpu_many_datasets.conf`:

    0 python3 /hpc/home/hpc_username/runners/example.py 0
    1 python3 /hpc/home/hpc_username/runners/example.py 1
    2 python3 /hpc/home/hpc_username/runners/example.py 2
    3 python3 /hpc/home/hpc_username/runners/example.py 3
    4 python3 /hpc/home/hpc_username/runners/example.py 4
    5 python3 /hpc/home/hpc_username/runners/example.py 5
    6 python3 /hpc/home/hpc_username/runners/example.py 6
    7 python3 /hpc/home/hpc_username/runners/example.py 7
    8 python3 /hpc/home/hpc_username/runners/example.py 8
    9 python3 /hpc/home/hpc_username/runners/example.py 9
    10 python3 /hpc/home/hpc_username/runners/example.py 10
    11 python3 /hpc/home/hpc_username/runners/example.py 11
    12 python3 /hpc/home/hpc_username/runners/example.py 12
    13 python3 /hpc/home/hpc_username/runners/example.py 13
    14 python3 /hpc/home/hpc_username/runners/example.py 14
    15 python3 /hpc/home/hpc_username/runners/example.py 15
    
This file contains lines of python3 commands which set off our modeling script script(s)! It is now clear how to set 
off many hpc jobs; just add each modeling script you want to run to this script. 

The numbers on the left running from 0-15 specify the CPU number and should always run from 0. 

The numbers on the right are inputting an integer, which is then used to load a specific dataset. Below, using 
the `sys.argv[1]` command, we load each integer into the Python script. For example, the first job loads the integer
0, the second job the integer 1 and so forth. Each job will therefore have a unique integer value in the `hpc_id` 
variable.
"""
import sys

hpc_id = int(sys.argv[1])

"""
We can now use this variable to load a specific piece of data for this run!
"""

dataset_type = "imaging"
pixel_scales = 0.1

dataset_name = []
dataset_name.append("example_image_1")  # Index 0
dataset_name.append("example_image_2")  # Index 1
dataset_name.append("example_image_3")  # Index 2
dataset_name.append("example_image_4")  # Index 3
dataset_name.append("example_image_5")  # Index 4
dataset_name.append("example_image_6")  # Index 5
dataset_name.append("example_image_7")  # Index 6
dataset_name.append("example_image_8")  # Index 7
# ...and so on.

"""
We now extract the dataset name specific to this hpc id, meaning every CPU run will load and fit a different dataset.
"""
dataset_name = dataset_name[hpc_id]

"""
We now create the overall path to the dataset this specific call of the script fits, which for the first line in the 
`.conf` file above (which has integer input 0) is: 

 `/hpc/data/hpc_username/dataset/imaging/example_image_1`
"""
dataset_path = Path(hpc_dataset_path, dataset_type, dataset_name)

"""
You now have all the code you need to set up many single-CPU jobs on the hpc!

You would simply append the batch scripts and Python code aboves to the lens modeling script script you are using,
which is given below for completeness.

However, first we describe how to set up a single multi-CPU Nautilus job on the hpc.

__Batch Script One Lenses__

Lets now look at the second example batch script, `example_cpu_one_dataset_parallel`, which fits a single dataset
using multiple CPUs.

#!/bin/bash -l

 `#SBATCH -N 1` - The number of nodes we require, where 1 node contains 28 CPUs on the hpc.
 `#SBATCH --ntasks=1` - The total number of task we are submitting.
 `#SBATCH --cpus-per-task=16` - The number of tasks per CPU.
 `#SBATCH -J example` - The name of the job, which is how it appears on hpc when you inspect it.
 `#SBATCH -o output/output.%A.out` - Python interpreter output is placed in a file in the `output` folder.
 `#SBATCH -o error/error.%A.out` - Python interpreter errors are placed in a file in the `error` folder.
 `#SBATCH -p hpc` - Signifies we are running the job on the hpc.
 `#SBATCH -A dp004` - The project code of the submission.
 `#SBATCH -t 48:00:00` - The job will terminate after this length of time (if it does not end naturally).
 `#SBATCH --mail-type=END` - If you input your email, when you`ll get an email about the job (END means once finished).
 `#SBATCH --mail-user=fill@me.co.uk` - The email address the hpc sends the email too.

source /hpc/home/hpc_username/activate.sh

export CPUS_PER_TASK=1

export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK

python3 /hpc/home/hpc_username/runners/example.py 16

The key difference in this batch script is the line which sets off the job:

- ntasks=1 - We only require one task because we are only fitting one dataset.

- cpus-per-task=16 - We require 16 CPUs for this single task, which the Nautilus non-linear search will use
to parallelize the model-fit over.

- python3 ... example.py 16 - We input the integer 16, which is used below to set the number of CPUs Nautilus

Lets now look at the beginning of the modeling script script again, which now does not use a list of datasets 
to load, but instead has the dataset name hard coded.
"""
import autofit as af
import autolens as al

"""
__Dataset__

Load and plot the strong lens dataset `example_image_1` via .fits files, which we will fit with the lens model.
"""
dataset_folder = "example"
dataset_name = "simple__no_lens_light"

dataset_path = Path(hpc_dataset_path, dataset_folder, dataset_name)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Nautilus CPUs__

The final change we need to make is to set the number of CPUs Nautilus uses in the model-fit.

We do this by loading the integer input form the batch script, which we set to be the number of CPUs Nautilus uses.
"""
number_of_cores = int(sys.argv[1])

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
 
 - The source galaxy's light is an MGE [6 parameters].

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

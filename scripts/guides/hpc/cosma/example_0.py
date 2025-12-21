# %%
"""
__WELCOME__

Welcome to a cosma modeling script Python script, which illustrates how to load a strong lens dataset and analyse it on cosma.

This example shows how to set off many single CPU jobs in a single COSMA submission script, where each job
fits a different imaging dataset using the same lens model analysis. This form of parallelization is therefore
beneficial when we have many datasets we wish to fit simultaneously.

The script `example_1.py` describes how to fit a single dataset with a parallelized Nautilus model-fit. You should
only read this example after reading and understanding this example.

This fits a lens model using a simple example taken from the autolens_workspace.
"""

# %%
"""
__COSMA PATHS SETUP__

Setup the path to the cosma output directory.

This example assumes you are using cosma7 and outputting results to the cosma7 output directory:

 `/cosma7/data/dp004/cosma_username`.
"""
from os import path
from pathlib import Path

cosma_path = Path(path.sep) / "cosma7" / "data" / "dp004" / "cosma_username"

"""
Use this path to set the path to the dataset directory on COSMA, as well as the folders within this directory the .fits
are stored in.

Below, we set `cosma_dataset_path=/cosma7/data/dp004/cosma_username/dataset/example/mass_sie__source_seric`.
"""
dataset_folder = "example"
dataset_name = "simple__no_lens_light"

cosma_dataset_path = Path(cosma_path) / "dataset" / dataset_folder / dataset_name

"""
We also set the output path on COSMA to `cosma_output_path=/cosma7/data/dp004/cosma_username/output`.
"""

cosma_output_path = Path(cosma_path, "output")

"""
In contrast to the dataset and output folders, our workspace path is in your COSMA home directory.
"""
workspace_path = "/cosma/home/dp004/cosma_username/autolens_workspace/"

"""
Use this to set the path to the config files that are used in this analysis, which are contained within the `cosma` 
directory of the example project in your COSMA home directory.
"""
config_path = Path(workspace_path, "cosma", "config")

"""
Set the config and output paths using autoconf, as you would for a laptop run.
"""
from autoconf import conf

conf.instance.push(new_path=config_path, output_path=cosma_output_path)

"""
Cosma submissions require a`batch script`, which tells Cosma the PyAutoLens runners you want it to execute and 
distributes them to nodes and CPUs. Lets look at the batch script 

 `autolens_workspace/misc/hpc/batch/example
    
The following options are worth noting:

 `#SBATCH -N 1` - The number of nodes we require, where 1 node contains 28 CPUs on COSMA7.
 `#SBATCH --ntasks=16` - The total number of task we are submitting.
 `#SBATCH --cpus-per-task=1` - The number of tasks per CPU.
 `#SBATCH -J example` - The name of the job, which is how it appears on cosma when you inspect it.
 `#SBATCH -o output/output.%A.out` - Python interpreter output is placed in a file in the `output` folder.
 `#SBATCH -o error/error.%A.out` - Python interpreter errors are placed in a file in the `error` folder.
 `#SBATCH -p cosma7` - Signifies we are running the job on COSMA7.
 `#SBATCH -A dp004` - The project code of the submission.
 `#SBATCH -t 48:00:00` - The job will terminate after this length of time (if it does not end naturally).
 `#SBATCH --mail-type=END` - If you input your email, when you`ll get an email about the job (END means once finished).
 `#SBATCH --mail-user=fill@me.co.uk` - The email address COSMA sends the email too.

The following line activates the PyAutoLens virtual environment we set up on cosma for this run:

 `source /cosma/home/dp004/cosma_username/autolens_workspace/activate.sh`

These lines prevent the NumPy linear algebra libraries from overloading the CPUs during calculations.
    
export CPUS_PER_TASK=1

export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK

This line sets off the job:

    srun -n 16 --multi-prog conf/example.conf

Lets checkout the file `example.conf`:

    0 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 0
    1 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 1
    2 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 2
    3 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 3
    4 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 4
    5 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 5
    6 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 6
    7 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 7
    8 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 8
    9 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 9
    10 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 10
    11 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 11
    12 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 12
    13 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 13
    14 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 14
    15 python3 /cosma/home/dp004/cosma_username/autolens_workspace/cosma/runners/example.py 15
    
This file contains lines of python3 commands which set off our modeling script script(s)! It is now clear how to set off many 
cosma jobs; just add each modeling script you want to run to this script. 

The numbers on the left running from 0-15 specify the CPU number and should always run from 0. 

The numbers on the right are inputting an integer, which is then used to load a specific dataset. Below, using 
the `sys.argv[1]` command, we load each integer into the Python script. For example, the first job loads the integer
0, the second job the integer 1 and so forth. Each job will therefore have a unique integer value in the `cosma_id` 
variable.
"""
import sys

cosma_id = int(sys.argv[1])

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
We now extract the dataset name specific to this cosma id, meaning every CPU run will load and fit a different dataset.
"""
dataset_name = dataset_name[cosma_id]

"""
We now create the overall path to the dataset this specific call of the script fits, which for the first line in the 
`.conf` file above (which has integer input 0) is: 

 `/cosma7/data/dp004/cosma_username/dataset/imaging/example_image_1`
"""
dataset_path = Path(cosma_dataset_path, dataset_type, dataset_name)

"""
COMPLETE

This completes all COSMA specific code required for submitting jobs to COSMA. All of the code below is not specific to 
COSMA, and is simply the code you are used to running in modeling script scripts not on COSMA.

In this example, we assumed that every job used a single CPU and we paralleized over the datasets being fitted. 
Checkout the file `example_1.py` for a description of how to fit a single dataset and parallelie the Nautilus search
over multiply cores.
"""
import autofit as af
import autolens as al


"""
__Dataset__

Load and plot the strong lens dataset `example_image_1` via .fits files, which we will fit with the lens model.
"""
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Mask__

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
__Search__

The lens model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

The folders: 

 - `autolens_workspace/*/guides/modeling/searches`.
 - `autolens_workspace/*/guides/modeling/customize`
  
Give overviews of the non-linear searches **PyAutoLens** supports and more details on how to customize the
model-fit, including the priors on the model.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/modeling/imaging/simple__no_lens_light/mass[sie]_source[bulge]/unique_identifier`.
"""
search = af.Nautilus(
    path_prefix="cosma_example",
    name="mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    n_live=100,
)

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

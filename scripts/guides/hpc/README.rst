PyAutoLens on High-Performance Computing (HPC)
==============================================

Introduction
------------

This guide describes how to set up and run PyAutoLens on a High-Performance
Computing (HPC) system. It assumes:

- You have access to an HPC cluster via SSH
- The cluster uses the SLURM batch scheduling system
- Python is available via system modules, Conda, or a custom installation

Although every HPC system differs slightly (filesystem layout, module names,
Python versions, storage quotas), the core steps are universal. Where
system-specific details are required, these are clearly marked and easy
to adapt.

Overview of the HPC Workflow
----------------------------

On most HPC systems:

- Home directories have limited storage and are best used for:
  - source code
  - virtual environments
  - configuration files

- Scratch or data filesystems are designed for:
  - large datasets (e.g. ``.fits`` files)
  - PyAutoLens output directories
  - intermediate results

This guide follows best practice by separating:

- the PyAutoLens workspace
- the data and output directories

PyAutoLens Virtual Environment
------------------------------

Before running PyAutoLens on an HPC system, you should create a Python
virtual environment in your home directory. This environment will contain
PyAutoLens and all its dependencies.

.. note::

   Throughout this guide, replace values written in ALL CAPS (e.g.
   ``YOUR_USERNAME`` or ``HPC_LOGIN_HOST``) with the appropriate values for
   your system.

Installing PyAutoLens: Available Options
----------------------------------------

There are two supported ways to install PyAutoLens on an HPC system.

Option 1 (Recommended): Install via ``pip`` or ``conda``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Simple and robust
- Ideal if you do not need to modify PyAutoLens source code

Option 2: Clone the GitHub repositories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Useful if you are developing PyAutoLens itself
- Follow the instructions in ``README_Repos.rst`` instead of this guide

This document assumes **Option 1**.

Step-by-Step Guide (Installation via ``pip``)
---------------------------------------------

1. SSH into the HPC
^^^^^^^^^^^^^^^^^^

From your local machine:

::

   ssh -X YOUR_USERNAME@HPC_LOGIN_HOST

You should now be logged into your home directory on the HPC system.

2. (Optional) Confirm Your Location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   pwd

This should point to something like:

::

   /home/YOUR_USERNAME

3. Create a Python Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   python3 -m venv PyAuto

This creates a virtual environment called ``PyAuto`` in your home directory.

.. note::

   You may rename this directory if you wish, but you must update paths
   consistently later in this guide.

4. Create an Activation Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a helper script that loads required modules (if applicable),
activates the virtual environment, and sets environment variables.

::

   nano activate.sh

(Use ``emacs -nw activate.sh`` or ``vi activate.sh`` if you prefer.)

5. Edit ``activate.sh``
^^^^^^^^^^^^^^^^^^^^^^

Paste the following template and adapt it to your system:

::

   #!/bin/bash

   # Reset loaded modules (optional but recommended)
   module purge

   # Load required modules (EDIT THESE FOR YOUR HPC)
   module load python/3.X.Y

   # Activate the virtual environment
   source $HOME/PyAuto/bin/activate

   # Ensure Python can see your workspace
   export PYTHONPATH=$HOME:$HOME/PyAuto

Make the script executable:

::

   chmod +x activate.sh

.. note::

   - Some HPC systems do not use environment modules. If so, remove the
     ``module`` lines.
   - If you use Conda instead of ``venv``, activate your Conda environment
     here instead.

6. Activate the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   source activate.sh

You should now see ``(PyAuto)`` at the start of your command prompt.

7. Install PyAutoLens
^^^^^^^^^^^^^^^^^^^^

::

   pip install autolens

This installs PyAutoLens and all required dependencies into the virtual
environment.

8. (Optional) Auto-Activate on Login
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you primarily use PyAutoLens on this HPC, you can automatically activate
the environment when logging in.

Edit your shell configuration file:

::

   nano ~/.bashrc

Add:

::

   source $HOME/activate.sh

PyAutoLens Workspace
--------------------

Your workspace mirrors the structure you use on your laptop, but excludes
large datasets and output directories.

On the HPC, this typically lives in:

::

   $HOME/autolens_workspace

Uploading Workspace Files from Your Laptop
------------------------------------------

From your local machine, navigate to your workspace:

::

   cd /path/to/autolens_workspace

We will use ``rsync`` to transfer files efficiently.

Recommended ``rsync`` options:

- ``--update`` – only copy newer files
- ``-v`` – verbose output
- ``-r`` – recursive directory transfer

Uploading Pipelines (Example: SLaM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   rsync --update -v -r slam \
     YOUR_USERNAME@HPC_LOGIN_HOST:$HOME/autolens_workspace

HPC-Specific Workspace Folder
-----------------------------

Create an ``hpc`` folder inside your workspace (for example by copying
``misc/hpc``).

This folder typically contains:

- ``config``:
  - HPC-specific configuration files
  - ``general.yaml`` with ``hpc_mode: true``

- ``runners``:
  - runner scripts adapted for batch execution

- ``batch``:
  - SLURM submission scripts

Upload the folder:

::

   rsync --update -v -r hpc \
     YOUR_USERNAME@HPC_LOGIN_HOST:$HOME/autolens_workspace

Data and Output Directories
---------------------------

Large datasets and PyAutoLens output should be stored on a high-capacity
filesystem, often named something like:

- ``/scratch``
- ``/work``
- ``/data``
- ``/gpfs``
- ``/lustre``

Consult your HPC documentation to find the correct location.

Example Directory Setup (On HPC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   cd /PATH/TO/LARGE_STORAGE/YOUR_USERNAME
   mkdir -p dataset
   mkdir -p output

Uploading Data from Your Laptop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From your local workspace:

::

   rsync --update -v -r dataset/* \
     YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/dataset/

Uploading Existing Output (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   rsync --update -v -r output/* \
     YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/output/

Uploading a Single Dataset or Output Folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   rsync --update -v -r dataset/example \
     YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO/LARGE_STORAGE/YOUR_USERNAME/dataset/

::

   rsync --update -v -r output/example \
     YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO_LARGE_STORAGE/YOUR_USERNAME/output/

Downloading Results Back to Your Laptop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   rsync --update -v -r \
     YOUR_USERNAME@HPC_LOGIN_HOST:/PATH/TO_LARGE_STORAGE/YOUR_USERNAME/output/* \
     ./output/

Next Steps
----------

You are now fully set up to run PyAutoLens on an HPC system.

To submit your first job:

- Navigate to the example runner script:

  ::

     autolens_workspace/misc/hpc/example_cpu.py

- Review the SLURM batch scripts in:

  ::

     autolens_workspace/hpc/batch

These scripts demonstrate how to configure and submit PyAutoLens jobs using
SLURM.

Step-by-Step Guide (Installation via GitHub Repositories)
---------------------------------------------------------

To set up your PyAutoLens workspace on an HPC system, the following commands
should work from a terminal. You may need to adapt hostnames, module names,
and paths for your specific system.

1. SSH into the HPC
^^^^^^^^^^^^^^^^^^

From your local machine:

::

   ssh -X YOUR_USERNAME@HPC_LOGIN_HOST

This logs you into the HPC and places you in your home directory (e.g.
``/home/YOUR_USERNAME``).

2. (Optional) Check Your Home Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   pwd

This step is optional, but useful for confirming the location of your home
directory.

3. Create the Python Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   python3 -m venv PyAuto

This creates a virtual environment called ``PyAuto`` in your home directory.
This environment will contain PyAutoLens and all of its dependencies.

.. note::

   The virtual environment must be named ``PyAuto`` unless you update all
   paths accordingly in later steps.

4. Create the PyAutoLens Workspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   mkdir autolens_workspace
   cd autolens_workspace

This creates and enters an ``autolens_workspace`` directory on the HPC, which
will mirror your local PyAutoLens workspace (excluding large datasets and
outputs).

5. Create the Activation Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an activation script that will be sourced every time you log into
the HPC.

::

   emacs -nw activate.sh

(You may use ``nano`` or ``vi`` instead if preferred.)

6. Edit ``activate.sh``
^^^^^^^^^^^^^^^^^^^^^^

Paste the following template into ``activate.sh`` and adapt it to your
system. In particular, update module names and paths as required.

::

   module purge

   # Load required modules (EDIT THESE FOR YOUR HPC)
   module load python/3.X.Y

   # Activate the virtual environment
   source $HOME/PyAuto/bin/activate

   # Add PyAutoLens and all parent projects to PYTHONPATH
   export PYTHONPATH=$HOME:\
   $HOME/PyAuto/PyAutoConf:\
   $HOME/PyAuto/PyAutoFit:\
   $HOME/PyAuto/PyAutoArray:\
   $HOME/PyAuto/PyAutoGalaxy:\
   $HOME/PyAuto/PyAutoLens

Save and exit the editor.

.. note::

   - If your HPC does not use environment modules, remove the ``module`` lines.
   - If you use Conda instead of ``venv``, activate your Conda environment here.

Make the script executable:

::

   chmod +x activate.sh

7. Activate the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   source activate.sh

If successful, ``(PyAuto)`` should appear at the start of your command prompt.

8. Clone the PyAuto GitHub Repositories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to the virtual environment directory:

::

   cd $HOME/PyAuto

Clone all required repositories:

::

   git clone https://github.com/rhayes777/PyAutoConf
   git clone https://github.com/rhayes777/PyAutoFit
   git clone https://github.com/Jammy2211/PyAutoArray
   git clone https://github.com/Jammy2211/PyAutoGalaxy
   git clone https://github.com/Jammy2211/PyAutoLens

9. Install Python Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the dependencies for each package:

::

   pip install -r PyAutoConf/requirements.txt
   pip install -r PyAutoFit/requirements.txt
   pip install -r PyAutoArray/requirements.txt
   pip install -r PyAutoGalaxy/requirements.txt
   pip install -r PyAutoLens/requirements.txt

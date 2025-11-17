**INTRODUCTION**

These scripts describe how to set up PyAutoLens on a high performance computer (HPC). The scripts are written for
the Durham University COSMA super computer, which uses the SLURM batch proceessing system. This guide may not be a 
100% applicable to your HPC resources, however the majority of tasks required for setting up PyAutoLens on HPC are 
general, so it should help out either way!

**PyAutoLens Virtual Environment**

Before running autolens scripts on a HPC, we need to setup a workspace in your Cosma home directory. Your home
directory has a limited amount of hard-disk space, so it is key that files with large filesizes (e.g. the .fits datasets
and model-fit output folders) are omitted from this workspace and stored elsewhere on Cosma.

**Installing PyAutoLens Options**

There are two options for how we install PyAutoLens on cosma):

  - Installing PyAutoLens into the virtual environment via pip (or conda).
  - Cloning the PyAutoLens (and parent project) GitHub repositories.

 If you are unsure which to use, I would recommend you install via pip, which uses the instructions given in this
 document. If you wish to clone the PyAutoLens (and parent projecT) repositires follows the instructions given in the
 file `autolens_workspace/misc/hpc/doc_repos`

**Step by Step Guide (installation via pip)**

To setup your cosma PyAutoLens workspace, the following commands should work from a terminal:

1) ssh -X cosma_username@login.cosma.dur.ac.uk

   This command ssh's you into cosma. It'll log you into your home directory ('/cosma/home/durham/cosma_username').

2) pwd

   This command is not necessary, but allows us to quickly check the path of our cosma home directory.

3) python3 -m venv PyAuto

   This makes a 'PyAuto' virtual environment and directory on your cosma home, where we will install PyAutoLens and
   its associated dependencies. You MUST name your workspace PyAuto to use this setup file without changing any commands.

4) mkdir autolens_workspace
   cd autolens_workspace

   This creates and takes us into an autolens_workspace directory on COSMA.

5) emacs -nw activate.sh

   This opens an 'activate.sh' script, which we'll use every time we log into cosma to activate the virtual environment.

6) Copy and paste the following commands into the activate.sh script, which should be open in emacs. Make sure to
   change the "cosma_username" entries to your cosma username!

   To paste into an emacs window, use the command "CTRL + SHIFT + V"
   To exit and save, use the command "CTRL + SHIFT + X" -> "CTRL + SHIFT + C" and push 'y' for yes.

   module purge
   module load cosma/2018
   module load python/3.6.5
   source /cosma/home/dp004/dc-nigh1/PyAuto/bin/activate
   export PYTHONPATH=/YOUR_COSMA_HOME_DIRECTORY/:\
   /YOUR_COSMA_HOME_DIRECTORY/PyAuto

7) source activate.sh

   This activates your PyAuto virtual environment. '(PyAuto)' should appear at the bottom left of you command line,
   next to where you type commands.

8) Install autolens, as per usual, using the command

   pip install autolens

Whenever you log into Cosma, you will need to 'activate' your PyAuto environment by running command 6) above. If you
want, you can make it so Cosma does this automatically whenever you log in. To make this your default setup (e.g. if
you're only going to be using PyAuto on Cosma) you can add the activate line to your .bashrc file:

    emacs -nw $HOME/.bashrc

then copy and paste (CTRL + SHIFT + V):

    source $HOME/autolens_workspace/activate.sh

And save and exit ("CTRL + SHIFT + X" -> "CTRL + SHIFT + C" and push 'y' for yes).



**PyAutoLens WORKSPACE**

Now we've set up our PyAutoLens virtual environment, we want to setup our workspace on Cosma, which will behave similar
to the workspace you're used to using on your laptop. First, make sure you are in the autolens_workspace directory.

    cd $HOME/autolens_workspacee

We are going to need to send files from your laptop to Cosma, and visa versa. On Cosma, the data and output files of
PyAutoLens are stored in a separate directory to the workspace (we'll cover that below). Therefore, all we need to do
is transfer your config files, pipelines and runners to a workspace folder on Cosma.

Thus, we need to upload these folders from our laptop to this directory on Cosma and eventually download the results
of a PyAutoLens analysis on Cosma to our workspace.

The command 'rsync' does this and we'll use 3 custom options of rsync:

 1) --update, which only sends data which has been updated on your laptop / Cosma since the data was previously
 uploaded or downloaded. This ensures we don't resend our entire dataset or set of results every time we perform a
 file transfer. Phew!

 2) -v, this stands for verbose, and gives text output of the file transfer's progress.

 3) -r, we'll send folders full of data rather than individual files, and this r allows us to send entire folders.

Before running rsync, you should navigate your command line terminal to your laptop's `autolens_workspace`.

    cd /path/to/autolens_workspace

These tutorials will assume you are running the SLaM pipelines in PyAutoLens. However, example scripts and runners can
easily be uploaded instead by simply changing the commands below.

To upload the SLaM pipelines from your laptop to Cosma

    rsync --update -v -r slam cosma_username@login.cosma.dur.ac.uk:/cosma/home/durham/cosma_username/autolens_workspace

Next, you should create a 'cosma' directory in your autolens_workspace on your laptop. To do this, you can start by
copy and pasting the folder from 'autolens_workspace/misc/hpc/cosma to the main root of your workspace,
'autolens_workspace'.

The cosma folder contains the following:

 - 'config': which is the same as the config's you've seen in the autolens_workspace. However, we want HPC runs to
   behave differently to our laptop runs, for example using a different matplotlib backend for visualization and
   zipping up the results to reduce file-storage usage. To faciliate this, the 'general.yaml' config file has
   a 'hpc_mode' option, which for this config files is now set to True.

 - 'runners': these are the runners you'll run on COSMA. A script 'example.py' is here, which changes how a runner
   is set up compared to runners on your laptop. We'll cover this in more detail script below.

 - 'batch': the scripts we use to send a 'job' to cosma, which we will again cover in detail below.

We'll now send the cosma folder to your autolens workspace on cosma (note how by doing this, we do not send the .fits
data to cosma yet).

    rsync --update -v -r cosma cosma_username@login.cosma.dur.ac.uk:/cosma/home/durham/cosma_username/autolens_workspace


**PyAutoLens DATA AND OUTPUT FOLDERS**

Now, we need to setup the Cosma directories that store our data and PyAutoLens output. Our data and output are stored
in a different location than our workspace on Cosma, because of the large amounts of data storage they require.

Logged into cosma (e.g. via ssh), type the following command to go to your data directory:

    COSMA5: cd /cosma5/data/cosma_username
    COSMA6: cd /cosma6/data/dp004/cosma_username
    COSMA7: cd /cosma7/data/dp004/cosma_username

NOTE: It is common for cosma data directories to be different to this. You should check emails from the cosma support
team to find your exact directory.

In the directory of you cosma_username, lets make the dataset and output folders we'll next transfer our data into.

    mkdir dataset
    mkdir output

On your laptop you should still be in your workspace, as you were when sending the pipelines and cosma folders.

The following rsync command can be used to send your data to Cosma (the example below uses the 'cosma5/data
cosma' directory which you should change if necessary):

    rsync --update -v -r dataset/* cosma_username@login.cosma.dur.ac.uk:/cosma5/data/autolens/cosma_username/dataset/

And this command can send your output, if you have any results from your laptop you wish to continue from on cosma (you
can omit this if you want you cosma runs to begin from scratch) (the 'cosma5/data' directory may need changing again):

    rsync --update -v -r output/* cosma_username@login.cosma.dur.ac.uk:/cosma5/data/autolens/cosma_username/output/

If we wanted to just send one dataset or output folder, (e.g., named 'example'), we would remove the * wildcards and write:

    rsync --update -v -r dataset/example cosma_username@login.cosma.dur.ac.uk:/cosma5/data/autolens/cosma_username/dataset/
    rsync --update -v -r output/example cosma_username@login.cosma.dur.ac.uk:/cosma5/data/autolens/cosma_username/output/

The following rsync commands can be used to download your dataset and output from Cosma:

    rsync --update -v -r cosma_username@login.cosma.dur.ac.uk:/cosma5/data/autolens/cosma_username/dataset/* ./dataset/
    rsync --update -v -r cosma_username@login.cosma.dur.ac.uk:/cosma5/data/autolens/cosma_username/output/* ./output/



Now you're setup, we're ready to run our first PyAutoLens analysis on Cosma. go to the
'autolens_workspace/misc/hpc/example_0.py' script to learn about how we submit PyAutoLens jobs to Cosma.
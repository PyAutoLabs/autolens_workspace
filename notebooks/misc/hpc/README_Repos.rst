**Step by Step Guide (installation via pip)**

To setup your cosma PyAutoLens workspace, the following commands should work from a terminal:

1) ssh -X cosma_username@login.cosma.dur.ac.uk

   This command ssh's you into cosma. It'll log you into your home directory ('/cosma/home/durham/cosma_username').

2) pwd

   This command is not necessary, but allows us to quickly check the path of our cosma home directory.

2) python3 -m venv PyAuto

   This makes a 'PyAuto' virtual environment and directory on your cosma home, where we will install PyAutoLens and
   its associated dependencies. You MUST name your workspace PyAuto to use this setup file without changing any commands.

3) mkdir autolens_workspace
   cd autolens_workspace

   This creates and takes us into an autolens_workspace directory on COSMA.

4) emacs -nw activate.sh

   This opens an 'activate.sh' script, which we'll use every time we log into cosma to activate the virtual environment.

5) Copy and paste the following commands into the activate.sh script, which should be open in emacs. Make sure to
   change the "cosma_username" entries to your cosma username! This includes the path to PyAutoLens and all of its
   parent projects, which will be cloning on Cosma in a moment.

   To paste into an emacs window, use the command "CTRL + SHIFT + V"
   To exit and save, use the command "CTRL + SHIFT + X" -> "CTRL + SHIFT + C" and push 'y' for yes.

   module purge
   module load cosma/2018
   module load python/3.6.5
   source /cosma/home/dp004/dc-nigh1/PyAuto/bin/activate
   export PYTHONPATH=/YOUR_COSMA_HOME_DIRECTORY/:\
   /YOUR_COSMA_HOME_DIRECTORY/PyAuto/PyAutoConf:\
   /YOUR_COSMA_HOME_DIRECTORY/PyAuto/PyAutoFit:\
   /YOUR_COSMA_HOME_DIRECTORY/PyAuto/PyAutoArray:\
   /YOUR_COSMA_HOME_DIRECTORY/PyAuto/PyAutoGalaxy:\
   /YOUR_COSMA_HOME_DIRECTORY/PyAuto/PyAutoLens

6) source activate.sh

   This activates your PyAuto virtual environment. '(PyAuto)' should appear at the bottom left of you command line,
   next to where you type commands.

7) Go into the `PyAuto` direction on cosma and clone all GitHub reposities.

   cd $HOME/PyAuto

   git clone https://github.com/rhayes777/PyAutoConf
   git clone https://github.com/rhayes777/PyAutoFit
   git clone https://github.com/Jammy2211/PyAutoArray
   git clone https://github.com/Jammy2211/PyAutoGalaxy
   git clone https://github.com/Jammy2211/PyAutoLens

   pip install -r PyAutoConf/requirements.txt
   pip install -r PyAutoFit/requirements.txt
   pip install -r PyAutoArray/requirements.txt
   pip install -r PyAutoGalaxy/requirements.txt
   pip install -r PyAutoLens/requirements.txt
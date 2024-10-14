# TSCPC Data Simulation Package
Wecome to this package for TSCPC histogram data generation. This package is intended to allow the simulation of single and mixed fluorophore fluoroscence data.

This package will allow you to simulate the histogram photon counts of any fluorophore given:
- The lifetime (ns) is known
- The emission spectra is known

The package also offers the user to control the temporal and spatial resolutions of the histogram, meaning:
- The total length of time of the histogram can be adjusted
- The time bin size can be adjusted
- The spectral range to which the histogram capture can be adjusted.

These setting can be configured in the probe-config.py file.

The package will also simulate properties of the device:
-  PDE (Photon Dectector Efficency), recreated from Erdogan et al 2019 Figure 15.2. This option will reduce the number of photons simulated and aggretated in the histogram by a scalor taken from the PDE function.
- IRF, we simulate the IRF using a Guassian function, allowing the user to provied a wavelength dependent $/mu$ and $/sigma$

This package allows for PDE (Photon Dectector Efficency), recreated from Erdogan et al 2019 Figure 15.2, to be generated and a Poission noise to be added to the samples.


## Set-up requirements
I recommend that the user sets up a conda environment to handle all of the packages required, installation instructions can be found [here](https://docs.anaconda.com/miniconda/miniconda-install/):


Next lets create a new conda environment and install the necessary packages:

```sh
conda create -n fluoro-data-sim python=3.11
conda install numpy scipy matplotlib jupyter seaborn tqdm
conda install -c conda-forge ipympl
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install memory_profiler
conda activate fluoro-data-sim
```

Ensure to activate the conda environment before running any of the code to generate data with: 
```sh
conda activate fluoro-data-sim
```

And to whenever you are finished with the data-generation you can stop using your conda environment with:

```sh
conda deactivate
```

## Quick Start
In the scripts directory you can find three example scripts:
- "mixture-simulation.ipynb" : This jupyter notebook will guide you through the process of creating multiple Tissue Fluorophore objects with different emission distributions and lifetimes, and then simulating the histogram data of the mixture of these fluorophores.

- "single_random_fluoro.ipynb" : This jupyter notebook will guide you through the process of creating a single Tissue Fluorophore object with a random emission distribution and lifetime, and then simulating the histogram data of this fluorophore.

- "single_random_fluoro_gen.py" : This python script will generate 10_000 samples of random fluorophore data (meaning randomly generated emission distribution and lifetime) and saving the data in the /data/sythetic-data directory, with a manifest file to keep track of emission distributions, lifetime and histogram values.

### Running the data generation python script
When running the data generation script I recommend using [tmux](https://github.com/tmux/tmux/wiki) or (screen)[https://linuxize.com/post/how-to-use-linux-screen/] to allow the script to run in the background. Meaning if the terminal is accidentally closed the script will continue to run.

#### Tmux

If using tmux, you can create a new tmux session with:
```sh
tmux new -s fluoro-data-sim
```

Then you can run the data generation script with:
```sh
conda activate fluoro-data-sim
python scripts/single_random_fluoro_gen.py
```

You can then detach from the tmux session with (meaning ctrl+b then d):
```sh
ctrl+b d
```

And you can reattach to the tmux session with:
```sh
tmux attach -t fluoro-data-sim
```

#### Screen
If using screen, you can create a new screen session with:
```sh
screen -S fluoro-data-sim
```

Then you can run the data generation script with:
```sh
conda activate fluoro-data-sim
python scripts/single_random_fluoro_gen.py
```

You can then detach from the screen session with (meaning ctrl+a then d):
```sh
ctrl+a d
```

And you can reattach to the screen session with:
```sh
screen -r fluoro-data-sim
```



## Package Structure
The package is structured as follows:

- data: This directory contains generally contains synthetic data generated by the package, and the manifest file that keeps track of the data.
- data/simulated-requirements: This directory contains the requirements for the simulation, such as the images of PDE and reference fluorophores emissions spectra.
- data/endogenous-fluoro-emissions: This directory contains the emission spectra of some endogenous fluorophores.
- scripts: This directory contains the example scripts for the package.
- src: This directory contains the source code for the package.

## Package Limitations
- This package is limited as the peak intensity of photon counts cannot be numerically calculated, therefore a more analytical approach is needed to calculate the peak intensity of the photon counts.




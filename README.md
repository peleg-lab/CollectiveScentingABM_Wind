# Gone With the Wind: Honey Bee Collective Scenting in the Presence of External Wind
# Agent-Based Model

This repo stores the code for the agent-based model for the manuscript "Gone With the Wind: Honey Bee Collective Scenting in the Presence of External Wind" to be presented atACM Collective Intelligence Conference, November 2023 (CI ‘23).  

Code for the base original agent-based model with detailed information of the model and visualization of output: https://github.com/peleg-lab/CollectiveScentingABM. This repo's code adds the extra feature of the wind (magnitude and condition) to the simulation environment. 

## Requirements
The complete list of required packages provided in *requirements.txt*, which you can install in your environment with the command `pip install -r requirements.txt`. Setting up a Python virtual environment, such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), is highly recommended.

## Model usage:
Default parameters are already set up. Parameters can be changed in run_simulation.py in the `config_opts` variable. Parameters related to the wind condition:
- `ww`: Wind magnitude (0 indicates no wind)
- `wwx`: x component of wind direction vector
- `wwy`: y component of wind direction vector

**`python run_simulation.py`** runs a simulation with the provided parameters.

### Output:
In the folder *experiments*, a subfolder will be created for this particular simulation with the name format *N{num_workers}_T{worker_threshold}_wb{worker_bias_scalar}_seed{random seed}*. A *cfg* file is created in this folder to record the model parameters for this simulation. The file *bee_hist.h5* contains time-series data for the position, state, scenting direction, distance from the queen of all bees. The file *envir_hist.h5* contains the time-series data for the environmental maps of pheromone diffusion and decay created by the collective scenting of the bees.

Reference: Nguyen DMT, Fard Gharooni G, Iuzzolino ML, Peleg O (2023). Gone With the Wind: Honey Bee Collective Scenting in the Presence of External Wind. Collective Intelligence Conference 2023.
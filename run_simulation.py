import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import modules.Bees as Bees
import modules.BeeKeeper as BeeKeeper
import modules.Environment as Environment

#---------------------------------------------#
#--------------- Make config file ---------------#
#---------------------------------------------#
def write_file(path, cfg_opts):
    with open(path, "w") as outfile:
        for key, val in cfg_opts.items():
            if key == "hidden_layers":
                write_str = f"--{key} "
                for val_i in val:
                    write_str += f"{val_i} "
                write_str += "\n"
                outfile.write(write_str)
            else:
                outfile.write(f"--{key} {val}\n")

#---------------------------------------------#
#--------------- Setup.py here ---------------#
#---------------------------------------------#
def directory(config):
    # Show params on folder title
    Q = config["queen_initial_concentration"]
    W = config["worker_initial_concentration"]
    D = config["diffusion_coefficient"]
    T = config["worker_threshold"]
    wb = config["worker_bias_scalar"]
    decay = config["decay"]
    ww = config["ww"]
    seed = config["random_seed"]
    N = config["num_workers"]

    params_name = f"T{T}_wb{wb}_ww{ww}_N{N}_seed{seed}"
    model_dir = os.path.join(config["base_dir"], f"{params_name}")

    # Make folder for this set of params
    os.makedirs(model_dir, exist_ok=True)

    return model_dir

def world_parameters(cfg, model_dir):
    bee_keeper_params = {
        "bee_path"         : os.path.join(model_dir, "bee_hist.h5"),
        "environment_path" : os.path.join(model_dir, "envir_hist.h5"),
        "src_path"         : os.path.join(model_dir, "src_hist.npy"),
        "sleeping"         : not cfg["measurements_on"],
        "save_concentration_maps" : cfg["save_concentration_maps"]
    }

    environment_params = {
        "x_min"      : cfg["x_min"],
        "x_max"      : cfg["x_max"],
        "dx"         : cfg["dx"],
        "t_min"      : cfg["t_min"],
        "t_max"      : cfg["t_max"],
        "dt"         : cfg["dt"],
        "D"          : cfg["diffusion_coefficient"],
        "decay_rate" : cfg["decay"],
        "culling_threshold" : cfg["culling_threshold"],
        "ww"         : cfg["ww"],
        "wwx"        : cfg["wwx"],
        "wwy"        : cfg["wwy"]
    }

    queen_params  = {
        "num"                : -1,
        "x"                  : cfg["queen_x"],
        "y"                  : cfg["queen_y"],
        "A"                  : cfg["queen_initial_concentration"],
        "wb"                 : cfg["queen_bias_scalar"],
        "emission_frequency" : cfg["queen_emission_frequency"],
    }

    bee_params = {
        "x_min"            : cfg["x_min"],
        "x_max"            : cfg["x_max"],
        "init_stddev"      : cfg["space_constraint"],
        "A"                : cfg["worker_initial_concentration"],
        "threshold"        : cfg["worker_threshold"],
        "wb"               : cfg["worker_bias_scalar"],
        "wait_period"      : cfg["worker_wait_period"],
        "step_size"        : cfg["worker_step_size"],
        "probabilistic"    : cfg["enable_probabilistic"],
        "trans_prob"       : cfg["worker_trans_prob"],
        "sensitivity_mode" : cfg["sensitivity_mode"]
    }

    world_params = {
        "bee_keeper" : bee_keeper_params,
        "environment" : environment_params,
        "queen" : queen_params,
        "worker" : bee_params
    }

    return world_params

def convert_index_to_xy(idx, idx_min=0, idx_max=600, xy_min=-3, xy_max=3):
    xy = np.interp(idx, [idx_min, idx_max], [xy_min, xy_max])
    return xy

def generate_points_with_min_distance(num_bees, shape, min_dist):
    # Compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = (np.sqrt(num_bees / width_ratio)) + 1
    num_x = (num_bees / num_y) + 1

    # Create regularly spaced points
    x = np.linspace(0., shape[1], num_x)[1:-1]
    y = np.linspace(0., shape[0], num_y)[1:-1]
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # Compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # Perturb points
    max_movement = (init_dist - min_dist) / 2
    noise = np.random.uniform(low=-max_movement,
                            high=max_movement,
                            size=(len(coords), 2))
    coords += noise
    return coords

def create_bees(coords, dim, cfg, bee_params):
    np.random.shuffle(coords)
    bees = []
    for bee_i in range(cfg["num_workers"]):
        bee_params = bee_params
        bee_params["num"] = bee_i
        bee = Bees.Worker(bee_params)

        # Give bee initial xy position
        bee_x_idx = coords[bee_i][0]
        bee_y_idx = coords[bee_i][1]
        bee_x = convert_index_to_xy(bee_x_idx, idx_min=0, idx_max=dim, xy_min=cfg["x_min"], xy_max=cfg["x_max"])
        bee_y = convert_index_to_xy(bee_y_idx, idx_min=0, idx_max=dim, xy_min=cfg["x_min"], xy_max=cfg["x_max"])
        bee.x = bee_x
        bee.y = bee_y

        # Append to list of bee objects
        bees.append(bee)
    return bees

def make_world_objects(cfg_options, world_params):
    dim = len(np.arange(cfg_options["x_min"], cfg_options["x_max"], cfg_options["dx"]))+1

    environment = Environment.Environment(world_params["environment"])
    queen_bee = Bees.Queen(world_params["queen"])
    bee_keeper = BeeKeeper.BeeKeeper(world_params["bee_keeper"])

    # Worker bee objects
    coords = generate_points_with_min_distance(cfg_options["num_workers"]*2, shape=(dim, dim), min_dist=10)
    bees = create_bees(coords, dim, cfg_options, world_params["worker"])

    world_objs = {
        "environment" : environment,
        "queen_bee" : queen_bee,
        "bees" : bees,
        "bee_keeper" : bee_keeper
    }

    return world_objs

#---------------------------------------------#
#--------------- Main.py here ---------------#
#---------------------------------------------#
def run(cfg_options, environment, queen_bee, bees, bee_keeper):
    try:
        for global_i, t_i in enumerate(environment):

            if cfg_options['verbose']:
                sys.stdout.write(f"\rTimestep: {global_i+1}/{environment.t_grid.shape[0]}")
                sys.stdout.flush()

            # Step 1: Check for and build sources list for current timestep
            # ----------------------------------------------------
            # Update pheromone list from queen bee
            environment.update_pheromone_sources(queen_bee, t_i)

            # Update pheromone list from worker bees
            for bee_i, bee in enumerate(bees):
                environment.update_pheromone_sources(bee, t_i)

            environment.cull_pheromone_sources(t_i)
            # ----------------------------------------------------

            # Step 2: Build Concentration map and get gradients
            # ----------------------------------------------------
            # Init concentration map for current timestep to 0's
            # environment.init_concentration_map()
            # Iterate through pheromone sources and build concentration maps
            # -- for each pheromone source, calculate gradient for each bee
            for pheromone_src in environment.pheromone_sources:
                # Update concentration map with x, y, A, dt, etc.
                pheromone_src_C = environment.update_concentration_map(t_i, pheromone_src)

                # Iterate through list of active bees and calculate gradient
                for bee in bees:
                    # If bee is inactive don't need to sense
                    if bee.state != 'inactive':
                        bee.sense_environment(t_i, environment, pheromone_src, pheromone_src_C)
            # ----------------------------------------------------

            # Step 3: Update bees & environment
            # ----------------------------------------------------
            queen_bee.update()

            for bee_i, bee in enumerate(bees):
                dist_to_queen = bee_keeper.compute_dist_to_queen(bee, queen_bee)
                bee.update(dist_to_queen)

                # Measure and store bee info
                bee_keeper.measure_bees(bee, queen_bee, global_i)

            # Store concentration maps
            bee_keeper.measure_environment(environment)
            # ----------------------------------------------------

            # Take steps (update movement, clear grads, etc)
            queen_bee.step()

            np.random.shuffle(bees)
            bee_positions = {f'bee_{bee.num}' : [bee.x, bee.y] for bee_i, bee in enumerate(bees)}

            # Add queen to list to not overlap her
            bee_positions['queen'] = [0,0]

            for bee in bees:
                bee.step(environment, bee_positions)

            if global_i % 50 == 0:
                bee_keeper.log_data_to_handy_dandy_notebook()

        # Save data to h5's
        bee_keeper.log_data_to_handy_dandy_notebook()

    except KeyboardInterrupt:
        print("\nEnding early.")
        bee_keeper.log_data_to_handy_dandy_notebook()


#---------------------------------------------#
#--------------- Run simulation ---------------#
#---------------------------------------------#
config_opts = {
        "verbose"     : True,
        "random_seed" : 0,

        ### ENVIRONMENT PARAMS ###
        "x_min" : -1, # -3,
        "x_max" : 1, # 3,
        "dx" : 0.01,
        "t_min" : 0,
        "t_max" : 10000 * (0.05/10),  
        "dt" : 0.05 / 10,
        "decay" : 18.0*6,
        "diffusion_coefficient" : 0.6,
        "ww": 0,
        "wwx": 0.1,
        "wwy": 0,
        # Wind directions (towards):
        # Down: wwx=0, wwy=0.1
        # Up: wwx=0, wwy=-0.1
        # Right: wwx=0.1, wwy=0
        # Left: wwx=-0.1, wwy=0

        ### QUEEN PARAMS ###
        "queen_x" : 0,
        "queen_y" : 0,
        "queen_bias_scalar" : 0.0,
        "queen_emission_frequency" : 80,
        "queen_initial_concentration" : 0.0575,

        ### WORKER PARAMS ###
        "num_workers" : 100,              
        "worker_wait_period" : 80,
        "worker_step_size" : 0.1,
        "worker_initial_concentration" : 0.0575,
        "worker_trans_prob" : 0.5,
        "enable_probabilistic" : True,
        "sensitivity_mode"  : "none",

        "culling_threshold" : 1e-3,
        "space_constraint" : 0.85,
        "t_threshold" : 100,

        "measurements_on" : True,
        "save_concentration_maps" : True,
        "base_dir" : f"experiments/ww000_n100", 
    }
    

def main(threshold, wb):
    config_opts["worker_threshold"] = threshold
    config_opts["worker_bias_scalar"] = wb     

    model_dir = directory(config_opts)
    write_file(f'{model_dir}/exp.cfg', config_opts)
    world_params = world_parameters(config_opts, model_dir)
    world_objects = make_world_objects(config_opts, world_params)

    run(config_opts, **world_objects)

if __name__ == '__main__':
    main()

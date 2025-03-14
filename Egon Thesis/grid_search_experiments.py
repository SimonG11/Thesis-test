"""
grid_search_experiments.py

This file performs grid search for three algorithms:
 - MOHHO (using MOHHO_with_progress)
 - PSO
 - MOACO (using MOACO_improved)

For each algorithm, every hyperparameter used in its instantiation is
searched over a candidate list. For each parameter combination, the experiment
is run 5 times (for statistical robustness). The absolute hypervolume is computed
(based on the union of all archives for that combination) and the combination with the
highest average absolute hypervolume is selected. A summary is saved as a JSON file.

A fixed time limit is enforced per run (e.g., 300 seconds), and a multi-level
progress bar structure is used to track progress across hyperparameter combinations,
runs, and (if needed) iterations.
"""

import json
import logging
import time
import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Import necessary modules (assumed available in your project)
from rcpsp_model import RCPSPModel
from tasks import get_default_tasks, generate_random_tasks
from algorithms import MOHHO_with_progress, PSO, MOACO_improved
from objectives import multi_objective
import utils
from metrics import absolute_hypervolume_fixed

# ------------------------
# Grid Search Functions
# ------------------------

def grid_search_mohho(param_grid: Dict[str, List[Any]], runs: int = 5,
                      use_random_instance: bool = False, base_num_tasks: int = 20,
                      time_limit: float = None) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    Grid search for MOHHO. Expected grid parameters:
      - "population": candidate values for search agents count.
      - "iter": candidate values for number of iterations.
      
    Each combination is run 'runs' times with a fixed time limit (if provided).
    Nested progress bars are used:
      - Outer bar: hyperparameter combinations (position=0)
      - Inner bar: runs for each combination (position=1, leave=False)
    """
    results_summary = {}  # key: parameter combination (as JSON string), value: average absolute hypervolume
    combinations = list(product(*param_grid.values()))
    
    # Outer progress bar for hyperparameter combinations (always visible)
    for combination in tqdm(combinations, desc="MOHHO: Combinations", position=0, leave=False):
        params = dict(zip(param_grid.keys(), combination))
        hv_values = []
        archives_list = []
        num_tasks = base_num_tasks
        # Inner progress bar for runs (this bar clears when done)
        for run in tqdm(range(runs), desc="Runs", position=1, leave=True):
            utils.initialize_seed(14 + run)
            # Set up tasks, workers, and model (as in experiment.py)
            workers = {"Developer": 10, "Manager": 2, "Tester": 3}
            worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
            tasks = generate_random_tasks(num_tasks, workers) if use_random_instance else get_default_tasks()
            num_tasks += 10
            model = RCPSPModel(tasks, workers, worker_cost)
            dim = len(model.tasks)
            lb_current = np.array([task["min"] for task in model.tasks])
            ub_current = np.array([task["max"] for task in model.tasks])
            
            # Extract grid search parameters for MOHHO
            population = params["population"]
            iters = params["iter"]
            
            # Run MOHHO with the time limit (its own progress bar for iterations is assumed to use position=2 if enabled)
            archive, _ = MOHHO_with_progress(lambda x: multi_objective(x, model),
                                             lb_current, ub_current, dim,
                                             population, iters,
                                             time_limit=time_limit)
            archives_list.append(archive)
        
        # Combine archives from all runs for this parameter combination
        union_archive = []
        for arch in archives_list:
            union_archive.extend(arch)
        # Compute fixed reference and global lower bound based on the union of archives
        ref_archives = {"dummy": [union_archive]}
        fixed_ref = utils.compute_fixed_reference(ref_archives)
        global_lower_bound = utils.compute_combined_ideal(ref_archives)
        # Compute absolute hypervolume for each run using the fixed references
        for arch in archives_list:
            hv = absolute_hypervolume_fixed(arch, fixed_ref, global_lower_bound)
            hv_values.append(hv)
        avg_hv = np.mean(hv_values)
        combo_str = json.dumps(params, sort_keys=True)
        results_summary[combo_str] = avg_hv
        tqdm.write(f"MOHHO Grid Search, Params: {combo_str}, Avg Abs HV: {avg_hv:.2f}")
    
    best_params_str, best_hv = max(results_summary.items(), key=lambda item: item[1])
    best_config = json.loads(best_params_str)
    return best_config, best_hv, results_summary


def grid_search_pso(param_grid: Dict[str, List[Any]], runs: int = 5,
                    use_random_instance: bool = False, base_num_tasks: int = 20,
                    time_limit: float = None) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    Grid search for PSO with a similar multi-level progress bar structure.
    """
    results_summary = {}
    combinations = list(product(*param_grid.values()))
    for combination in tqdm(combinations, desc="PSO: Combinations", position=0, leave=True):
        params = dict(zip(param_grid.keys(), combination))
        hv_values = []
        archives_list = []
        num_tasks = base_num_tasks
        for run in tqdm(range(runs), desc="Runs", position=1, leave=False):
            utils.initialize_seed(14 + run)
            workers = {"Developer": 10, "Manager": 2, "Tester": 3}
            worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
            tasks = generate_random_tasks(num_tasks, workers) if use_random_instance else get_default_tasks()
            num_tasks += 10
            model = RCPSPModel(tasks, workers, worker_cost)
            dim = len(model.tasks)
            lb_current = np.array([task["min"] for task in model.tasks])
            ub_current = np.array([task["max"] for task in model.tasks])
            objectives = [lambda x: multi_objective(x, model)]
            optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                            pop=params["pop"],
                            c2=params["c2"],
                            w_max=params["w_max"],
                            w_min=params["w_min"],
                            disturbance_rate_min=params["disturbance_rate_min"],
                            disturbance_rate_max=params["disturbance_rate_max"],
                            jump_interval=params["jump_interval"])
            _ = optimizer.run(max_iter=params["max_iter"], time_limit=time_limit)
            archives_list.append(optimizer.archive)
        union_archive = []
        for arch in archives_list:
            union_archive.extend(arch)
        ref_archives = {"dummy": [union_archive]}
        fixed_ref = utils.compute_fixed_reference(ref_archives)
        global_lower_bound = utils.compute_combined_ideal(ref_archives)
        for arch in archives_list:
            hv = absolute_hypervolume_fixed(arch, fixed_ref, global_lower_bound)
            hv_values.append(hv)
        avg_hv = np.mean(hv_values)
        combo_str = json.dumps(params, sort_keys=True)
        results_summary[combo_str] = avg_hv
        tqdm.write(f"PSO Grid Search, Params: {combo_str}, Avg Abs HV: {avg_hv:.2f}")
    best_params_str, best_hv = max(results_summary.items(), key=lambda item: item[1])
    best_config = json.loads(best_params_str)
    return best_config, best_hv, results_summary


def grid_search_moaco(param_grid: Dict[str, List[Any]], runs: int = 5,
                      use_random_instance: bool = False, base_num_tasks: int = 20,
                      time_limit: float = None) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    Grid search for MOACO with a similar multi-level progress bar structure.
    """
    results_summary = {}
    combinations = list(product(*param_grid.values()))
    for combination in tqdm(combinations, desc="MOACO: Combinations", position=0, leave=True):
        params = dict(zip(param_grid.keys(), combination))
        hv_values = []
        archives_list = []
        num_tasks = base_num_tasks
        for run in tqdm(range(runs), desc="Runs", position=1, leave=False):
            utils.initialize_seed(14 + run)
            workers = {"Developer": 10, "Manager": 2, "Tester": 3}
            worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
            tasks = generate_random_tasks(num_tasks, workers) if use_random_instance else get_default_tasks()
            num_tasks += 10
            model = RCPSPModel(tasks, workers, worker_cost)
            dim = len(model.tasks)
            lb_current = np.array([task["min"] for task in model.tasks])
            ub_current = np.array([task["max"] for task in model.tasks])
            archive, _ = MOACO_improved(lambda x: multi_objective(x, model),
                                        model.tasks, lb_current, ub_current,
                                        params["ant_count"], params["max_iter"],
                                        alpha=params["alpha"],
                                        beta=params["beta"],
                                        evaporation_rate=params["evaporation_rate"],
                                        colony_count=params["colony_count"],
                                        time_limit=time_limit)
            archives_list.append(archive)
        union_archive = []
        for arch in archives_list:
            union_archive.extend(arch)
        ref_archives = {"dummy": [union_archive]}
        fixed_ref = utils.compute_fixed_reference(ref_archives)
        global_lower_bound = utils.compute_combined_ideal(ref_archives)
        for arch in archives_list:
            hv = absolute_hypervolume_fixed(arch, fixed_ref, global_lower_bound)
            hv_values.append(hv)
        avg_hv = np.mean(hv_values)
        combo_str = json.dumps(params, sort_keys=True)
        results_summary[combo_str] = avg_hv
        tqdm.write(f"MOACO Grid Search, Params: {combo_str}, Avg Abs HV: {avg_hv:.2f}")
    best_params_str, best_hv = max(results_summary.items(), key=lambda item: item[1])
    best_config = json.loads(best_params_str)
    return best_config, best_hv, results_summary


# ------------------------
# Main Execution Block
# ------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Settings for grid search experiments
    runs = 5
    use_random_instance = False
    num_tasks = 20
    # Set a fixed time limit per run (e.g., 300 seconds = 5 minutes)
    TIME_LIMIT = 10

    # Hyperparameter Grids for RCPSP Scheduling Problems

    # MOHHO (Multi-Objective Harris Hawks Optimizer):
    #   - Population: 50–100 hawks (Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034)
    #   - Iterations: 200–500 iterations (Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034)
    mohho_grid = {
        "population": [50, 100],
        "iter": [200, 500]
    }
    
    # PSO (Particle Swarm Optimization):
    #   - pop: 50–100 particles (Huang et al., 2016; Eberhart & Shi, 2000)
    #   - c2: around 2.0 (Clerc & Kennedy, 2002: https://doi.org/10.1109/4235.995895)
    #   - w_max: 0.9, w_min: 0.4 (Eberhart & Shi, 2000: https://doi.org/10.1109/4235.995895)
    #   - disturbance_rate: ~5%-10% (Huang et al., 2016)
    #   - jump_interval: 50–100 iterations (Hybrid PSO studies)
    #   - max_iter: 200–500 iterations (Huang et al., 2016)
    pso_grid = {
        "pop": [50, 100],
        "c2": [1.9, 2.0, 2.1],
        "w_max": [0.9],
        "w_min": [0.4],
        "disturbance_rate_min": [0.05, 0.1],
        "disturbance_rate_max": [0.1, 0.15],
        "jump_interval": [50, 100],
        "max_iter": [200, 500]
    }
    
    # MOACO (Multi-Objective Ant Colony Optimization):
    #   - ant_count: 20–50 ants per iteration (Dorigo et al., 2006: https://link.springer.com/book/10.1007/978-3-540-31851-7)
    #   - max_iter: 200–500 iterations (Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034)
    #   - alpha: around 1.0 (Dorigo et al., 2006)
    #   - beta: between 2.0 and 2.5 (Dorigo et al., 2006)
    #   - evaporation_rate: between 0.1 and 0.3 (López-Ibáñez, 2004: https://scholar.google.com/scholar?q=Lopez-Ibanez+2004+ACO)
    #   - colony_count: 1 or 2 colonies (Iredi et al., 2001: https://doi.org/10.1109/CEC.2001.944994)
    moaco_grid = {
        "ant_count": [20, 50],
        "max_iter": [200, 500],
        "alpha": [1.0, 1.5],
        "beta": [2.0, 2.5],
        "evaporation_rate": [0.1, 0.2, 0.3],
        "colony_count": [1, 2]
    }

    # Run grid search for each algorithm.
    best_mohho, best_mohho_hv, mohho_summary = grid_search_mohho(
        mohho_grid, runs=runs,
        use_random_instance=use_random_instance,
        num_tasks=num_tasks,
        time_limit=TIME_LIMIT
    )
    # Uncomment below to run for PSO and MOACO as needed:
    # best_pso, best_pso_hv, pso_summary = grid_search_pso(
    #     pso_grid, runs=runs,
    #     use_random_instance=use_random_instance,
    #     num_tasks=num_tasks,
    #     time_limit=TIME_LIMIT
    # )
    # best_moaco, best_moaco_hv, moaco_summary = grid_search_moaco(
    #     moaco_grid, runs=runs,
    #     use_random_instance=use_random_instance,
    #     num_tasks=num_tasks,
    #     time_limit=TIME_LIMIT
    # )

    # Combine all results into an overall summary.
    overall_summary = {
        "MOHHO": {
            "best_config": best_mohho,
            "best_absolute_hypervolume": best_mohho_hv,
            "all_results": mohho_summary
        }
    #    "PSO": {
    #        "best_config": best_pso,
    #        "best_absolute_hypervolume": best_pso_hv,
    #        "all_results": pso_summary
    #    },
    #    "MOACO": {
    #        "best_config": best_moaco,
    #        "best_absolute_hypervolume": best_moaco_hv,
    #        "all_results": moaco_summary
    #    }
    }

    # Save the summary to a JSON file.
    with open("grid_search_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=4)

    logging.info("Grid search complete. Summary saved to 'grid_search_summary.json'.")

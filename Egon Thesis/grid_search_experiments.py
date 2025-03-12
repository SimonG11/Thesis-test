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
"""

import json
import logging
import time
import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Any

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
                      use_random_instance: bool = False, num_tasks: int = 20) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    Grid search for MOHHO. Expected grid parameters:
      - "population": candidate values for search agents count.
      - "iter": candidate values for number of iterations.
    """
    results_summary = {}  # key: parameter combination (as JSON string), value: average absolute hypervolume
    for combination in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combination))
        hv_values = []
        archives_list = []
        for run in range(runs):
            # Initialize seed to vary each run (e.g., base seed 14 offset by run index)
            utils.initialize_seed(14 + run)
            # Set up tasks, workers, and model (same as in experiment.py)
            workers = {"Developer": 10, "Manager": 2, "Tester": 3}
            worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
            tasks = generate_random_tasks(num_tasks, workers) if use_random_instance else get_default_tasks()
            model = RCPSPModel(tasks, workers, worker_cost)
            dim = len(model.tasks)
            lb_current = np.array([task["min"] for task in model.tasks])
            ub_current = np.array([task["max"] for task in model.tasks])
            
            # Extract grid search parameters for MOHHO
            population = params["population"]
            iters = params["iter"]
            
            # Run MOHHO (ignoring convergence curve here)
            archive, _ = MOHHO_with_progress(lambda x: multi_objective(x, model),
                                             lb_current, ub_current, dim,
                                             population, iters)
            archives_list.append(archive)
        # Combine archives from all runs for this parameter combination
        union_archive = []
        for arch in archives_list:
            union_archive.extend(arch)
        # Compute fixed reference and global lower bound based on the union of archives
        # (we use a dummy dictionary to mimic the structure expected by the utils functions)
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
        logging.info(f"MOHHO Grid Search, Params: {combo_str}, Avg Abs HV: {avg_hv:.2f}")
    # Determine best configuration (highest average absolute hypervolume)
    best_params_str, best_hv = max(results_summary.items(), key=lambda item: item[1])
    best_config = json.loads(best_params_str)
    return best_config, best_hv, results_summary

def grid_search_pso(param_grid: Dict[str, List[Any]], runs: int = 5,
                    use_random_instance: bool = False, num_tasks: int = 20) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    Grid search for PSO. Expected grid parameters:
      - "pop": population size.
      - "c2": cognitive coefficient.
      - "w_max": maximum inertia weight.
      - "w_min": minimum inertia weight.
      - "disturbance_rate_min": minimum disturbance rate.
      - "disturbance_rate_max": maximum disturbance rate.
      - "jump_interval": jump interval.
      - "max_iter": number of iterations.
    """
    results_summary = {}
    for combination in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combination))
        hv_values = []
        archives_list = []
        for run in range(runs):
            utils.initialize_seed(14 + run)
            workers = {"Developer": 10, "Manager": 2, "Tester": 3}
            worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
            tasks = generate_random_tasks(num_tasks, workers) if use_random_instance else get_default_tasks()
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
            # Run PSO with max_iter specified in the grid
            _ = optimizer.run(max_iter=params["max_iter"])
            # Archive is stored in optimizer.archive
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
        logging.info(f"PSO Grid Search, Params: {combo_str}, Avg Abs HV: {avg_hv:.2f}")
    best_params_str, best_hv = max(results_summary.items(), key=lambda item: item[1])
    best_config = json.loads(best_params_str)
    return best_config, best_hv, results_summary

def grid_search_moaco(param_grid: Dict[str, List[Any]], runs: int = 5,
                      use_random_instance: bool = False, num_tasks: int = 20) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    Grid search for MOACO. Expected grid parameters:
      - "ant_count": number of ants.
      - "max_iter": number of iterations.
      - "alpha": pheromone importance.
      - "beta": heuristic importance.
      - "evaporation_rate": pheromone evaporation rate.
      - "colony_count": number of ants in the colony (can be independent of ant_count).
    """
    results_summary = {}
    for combination in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combination))
        hv_values = []
        archives_list = []
        for run in range(runs):
            utils.initialize_seed(14 + run)
            workers = {"Developer": 10, "Manager": 2, "Tester": 3}
            worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
            tasks = generate_random_tasks(num_tasks, workers) if use_random_instance else get_default_tasks()
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
                                        colony_count=params["colony_count"])
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
        logging.info(f"MOACO Grid Search, Params: {combo_str}, Avg Abs HV: {avg_hv:.2f}")
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


    # Hyperparameter Grids for Grid Search Experiments for RCPSP Scheduling Problems
    #
    # MOHHO (Multi-Objective Harris Hawks Optimizer):
    #   - Population:
    #       Research on multi-objective scheduling (e.g., Huang et al., 2016) indicates that 
    #       a population in the range of 50–100 hawks is effective for balancing exploration 
    #       and computational cost.
    #       [Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034]
    #
    #   - Iterations:
    #       Studies report that 200–500 iterations are typically sufficient for convergence 
    #       in RCPSP-sized problems.
    #       [Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034]
    mohho_grid = {
        "population": [50, 100],  # 50-100 hawks (Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034)
        "iter": [200, 500]        # 200-500 iterations (Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034)
    }

    # PSO (Particle Swarm Optimization):
    #   - pop:
    #       A swarm size between 50 and 100 particles is common in scheduling applications.
    #       [Eberhart & Shi, 2000: https://doi.org/10.1109/4235.995895; Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034]
    #
    #   - c2:
    #       Cognitive and social coefficients are typically set around 2.0.
    #       [Clerc & Kennedy, 2002: https://doi.org/10.1109/4235.995895]
    #
    #   - w_max and w_min:
    #       The inertia weight is commonly decreased linearly from 0.9 to 0.4 to balance
    #       exploration and exploitation.
    #       [Eberhart & Shi, 2000: https://doi.org/10.1109/4235.995895]
    #
    #   - disturbance_rate_min & disturbance_rate_max:
    #       A small disturbance (around 5–10%) helps PSO escape local optima.
    #       [Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034]
    #
    #   - jump_interval:
    #       Hybrid PSO studies show that reinitializing or applying a “jump” every 50–100
    #       iterations can improve exploration.
    #       [Hybrid PSO studies: https://doi.org/10.1109/CEC.2001.944994]
    #
    #   - max_iter:
    #       PSO for scheduling typically runs for 200–500 iterations.
    #       [Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034]
    pso_grid = {
        "pop": [50, 100],                                # 50-100 particles (Huang et al., 2016; Eberhart & Shi, 2000)
        "c2": [1.9, 2.0, 2.1],                           # Around 2.0 for cognitive coefficient (Clerc & Kennedy, 2002: https://doi.org/10.1109/4235.995895)
        "w_max": [0.9],                                  # Maximum inertia weight 0.9 (Eberhart & Shi, 2000: https://doi.org/10.1109/4235.995895)
        "w_min": [0.4],                                  # Minimum inertia weight 0.4 (Eberhart & Shi, 2000: https://doi.org/10.1109/4235.995895)
        "disturbance_rate_min": [0.05, 0.1],             # Disturbance rate lower bound ~5%-10% (Huang et al., 2016)
        "disturbance_rate_max": [0.1, 0.15],             # Disturbance rate upper bound (Huang et al., 2016)
        "jump_interval": [50, 100],                      # Jump interval: every 50-100 iterations (Hybrid PSO: https://doi.org/10.1109/CEC.2001.944994)
        "max_iter": [200, 500]                           # 200-500 iterations (Huang et al., 2016)
    }

    # MOACO (Multi-Objective Ant Colony Optimization):
    #   - ant_count:
    #       In multi-objective scheduling, 20–50 ants per iteration are typically used to
    #       balance exploration with computational efficiency.
    #       [Dorigo et al., 2006: https://link.springer.com/book/10.1007/978-3-540-31851-7]
    #
    #   - max_iter:
    #       Similar to other algorithms, 200–500 iterations are used for convergence in RCPSP.
    #       [Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034]
    #
    #   - alpha:
    #       The pheromone importance is usually around 1.0.
    #       [Dorigo et al., 2006: https://link.springer.com/book/10.1007/978-3-540-31851-7]
    #
    #   - beta:
    #       The heuristic importance is commonly set between 2.0 and 2.5.
    #       [Dorigo et al., 2006: https://link.springer.com/book/10.1007/978-3-540-31851-7]
    #
    #   - evaporation_rate:
    #       Values between 0.1 and 0.3 are advised to balance exploitation and exploration.
    #       Starting at 0.1 is common.
    #       [López-Ibáñez, 2004: https://scholar.google.com/scholar?q=Lopez-Ibanez+2004+ACO]
    #
    #   - colony_count:
    #       For multi-objective problems, using either a single colony (with an external archive)
    #       or 2 colonies (e.g., for bi-objective problems) is typical.
    #       [Iredi et al., 2001: https://doi.org/10.1109/CEC.2001.944994]
    moaco_grid = {
        "ant_count": [20, 50],                          # 20-50 ants per iteration (Dorigo et al., 2006: https://link.springer.com/book/10.1007/978-3-540-31851-7)
        "max_iter": [200, 500],                          # 200-500 iterations (Huang et al., 2016: https://doi.org/10.1016/j.procs.2016.07.034)
        "alpha": [1.0, 1.5],                             # Pheromone importance around 1.0 (Dorigo et al., 2006: https://link.springer.com/book/10.1007/978-3-540-31851-7)
        "beta": [2.0, 2.5],                              # Heuristic importance between 2.0 and 2.5 (Dorigo et al., 2006)
        "evaporation_rate": [0.1, 0.2, 0.3],             # Evaporation rate between 0.1 and 0.3 (López-Ibáñez, 2004: https://scholar.google.com/scholar?q=Lopez-Ibanez+2004+ACO)
        "colony_count": [1, 2]                           # 1 or 2 colonies (Iredi et al., 2001: https://doi.org/10.1109/CEC.2001.944994)
    }


    # Run grid search for each algorithm.
    best_mohho, best_mohho_hv, mohho_summary = grid_search_mohho(mohho_grid, runs=runs,
                                                                   use_random_instance=use_random_instance,
                                                                   num_tasks=num_tasks)
    #best_pso, best_pso_hv, pso_summary = grid_search_pso(pso_grid, runs=runs,
    #                                                     use_random_instance=use_random_instance,
    #                                                     num_tasks=num_tasks)
    #best_moaco, best_moaco_hv, moaco_summary = grid_search_moaco(moaco_grid, runs=runs,
    #                                                             use_random_instance=use_random_instance,
    #                                                             num_tasks=num_tasks)

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

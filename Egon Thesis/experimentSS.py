import json, logging
import numpy as np
from typing import Dict, List, Tuple, Any
from rcpsp_model import RCPSPModel
from tasks import get_default_tasks, generate_random_tasks
from algorithmsS import MOHHO, PSO, MOACO
from metrics import (normalized_hypervolume_fixed, absolute_hypervolume_fixed, 
                     compute_generational_distance, compute_spread, 
                     compute_spread_3d_by_projections, compute_coverage,
)
from visualization import plot_gantt, plot_convergence, plot_all_pareto_graphs
from objectives import multi_objective
from ericsson_tasks import get_ericsson_tasks
import utils
import time
from tqdm import tqdm

# -------------------- NEW: Imports for pymoo --------------------
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination


def run_experiments(runs: int = 1, use_random_instance: bool = False, num_tasks: int = 10,
                    iterrations: int = 30, population: int = 5, time_limit: float = None, ericsson: bool = False
                   ) -> Tuple[Dict[str, Any],
                              Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]],
                              List[Dict[str, Any]],
                              Dict[str, List[List[float]]]]:
    """
    Run experiments for MOHHO, PSO, MOACO, a baseline, and a benchmark using NSGA-II from pymoo.
    Each algorithm is run for a fixed wall-clock time (if time_limit is provided).
    A nested progress bar structure is used:
      - Outer bar (position=0): overall experiment runs.
      - Inner bar (position=1): tracks each algorithm execution within a run.
      - Algorithm-specific progress bars (e.g., in MOHHO_with_progress) use higher positions.
    Returns results, archives, baseline schedules, and convergence curves.
    """
    if ericsson:
        tasks, workers, worker_cost = get_ericsson_tasks()
    else:
        workers = {"Developer": 10, "Manager": 2, "Tester": 3}
        worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
        logging.info("Generating tasks...")
        tasks = generate_random_tasks(num_tasks, workers) if use_random_instance else get_default_tasks()
        logging.info("Tasks generated")
    model = RCPSPModel(tasks, workers, worker_cost)
    dim = len(model.tasks)
    lb_current = np.array([task["min"] for task in model.tasks])
    ub_current = np.array([task["max"] for task in model.tasks])
    
    results = {
        "MOHHO": {"normalized_hypervolume": [], "absolute_hypervolume": [],
                  "spread": [], "multi_objective_spread": [], "coverage": [], "coverage_pso": [],
                  "coverage_aco": [], "coverage_nsga": [], "runtimes": []},
        "PSO": {"normalized_hypervolume": [], "absolute_hypervolume": [],
                "spread": [], "multi_objective_spread": [], "coverage": [], "coverage_hho": [],
                "coverage_aco": [], "coverage_nsga": [], "runtimes": []},
        "MOACO": {"normalized_hypervolume": [], "absolute_hypervolume": [],
                  "spread": [], "multi_objective_spread": [], "coverage": [], "coverage_pso": [],
                  "coverage_hho": [], "coverage_nsga": [], "runtimes": []},
        "NSGAII": {"normalized_hypervolume": [], "absolute_hypervolume": [],
                   "spread": [], "multi_objective_spread": [], "coverage": [], "coverage_pso": [],
                  "coverage_hho": [], "coverage_aco": [], "runtimes": []}
    }
    archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]] = {
        "MOHHO": [], "PSO": [], "MOACO": [], "NSGAII": []
    }
    base_schedules = []
    convergence_curves = {"MOHHO": [], "PSO": [], "MOACO": [], "NSGAII": []}
    base_schedule, _ = model.baseline_allocation()
    base_schedules.append(base_schedule)

    time_limit_formated = utils.seconds_to_hms(time_limit)
    termination = get_termination("time", time_limit_formated)
    # Outer progress bar: overall experiment runs
    for run in tqdm(range(runs), desc="Experiment Runs", position=0, leave=True):
        tqdm.write(f"Starting run {run+1}/{runs}...")
        
        # Create an inner progress bar for the four algorithm executions in this run.
        with tqdm(total=4, desc="Algorithms", position=1, leave=False) as algo_bar:
            
            # -------------------- MOHHO --------------------
            algo_bar.set_description("MOHHO")
            start_time = time.time()
            archive_hho, _ = MOHHO(lambda x: multi_objective(x, model),
                                                        lb_current, ub_current, dim,
                                                        population, iterrations,
                                                        time_limit=time_limit)
            runtime = time.time() - start_time
            results["MOHHO"]["runtimes"].append(runtime)
            archives_all["MOHHO"].append(archive_hho)
            algo_bar.write("MOHHO Done")
            algo_bar.update(1)
            
            # -------------------- PSO --------------------
            algo_bar.set_description("PSO")
            objectives = [lambda x: multi_objective(x, model)]
            optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                            pop=population, c2=2, w_max=0.9, w_min=0.4,
                            disturbance_rate_min=0.1, disturbance_rate_max=0.2, jump_interval=75)
            start_time = time.time()
            _ = optimizer.run(max_iter=iterrations, time_limit=time_limit)
            runtime = time.time() - start_time
            archive_pso = optimizer.archive
            results["PSO"]["runtimes"].append(runtime)
            archives_all["PSO"].append(archive_pso)
            algo_bar.write("PSO Done")
            algo_bar.update(1)
            
            # -------------------- MOACO --------------------
            algo_bar.set_description("MOACO")
            start_time = time.time()
            archive_moaco, _ = MOACO(
                lambda x: multi_objective(x, model),
                model.tasks, lb_current, ub_current, population, iterrations,
                alpha=1.0, beta=2.0, evaporation_rate=0.1,
                colony_count=(2),
                time_limit=time_limit
            )
            runtime = time.time() - start_time
            results["MOACO"]["runtimes"].append(runtime)
            archives_all["MOACO"].append(archive_moaco)
            algo_bar.write("MOACO Done")
            algo_bar.update(1)
            
            # -------------------- NSGA-II via pymoo --------------------
            algo_bar.set_description("NSGA-II")
            # Wrap the RCPSP problem in a pymoo Problem class.
            class RCPSPProblem(Problem):
                def __init__(self, model, lb, ub):
                    self.model = model
                    self.dim = len(model.tasks)
                    sample_obj = multi_objective(np.array([lb[i] for i in range(self.dim)]), self.model)
                    n_obj = len(sample_obj) if isinstance(sample_obj, (list, np.ndarray)) else 1
                    super().__init__(n_var=self.dim,
                                     n_obj=n_obj,
                                     n_constr=0,
                                     xl=lb,
                                     xu=ub)
                def _evaluate(self, X, out, *args, **kwargs):
                    F = []
                    for x in X:
                        F.append(multi_objective(x, self.model))
                    out["F"] = np.array(F)
            
            
            problem = RCPSPProblem(model, lb_current, ub_current)
            algorithm = NSGA2(pop_size=population)
            start_time = time.time()
            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=14,
                           verbose=True)
            nsga_runtime = time.time() - start_time
            archive_nsga = [(sol, obj) for sol, obj in zip(res.X, res.F)]
            archive_nsga = utils.remove_excess_solutions_with_crowding_distance(archive=archive_nsga, max_archive_size=100)
            results["NSGAII"]["runtimes"].append(nsga_runtime)
            archives_all["NSGAII"].append(archive_nsga)
            algo_bar.write("NSGA-II Done")
            algo_bar.update(1)
            
            results["MOACO"]["coverage"].append(compute_coverage(archive_moaco, archive_hho + archive_pso + archive_nsga))
            results["MOHHO"]["coverage"].append(compute_coverage(archive_hho, archive_moaco + archive_pso + archive_nsga))
            results["PSO"]["coverage"].append(compute_coverage(archive_pso, archive_hho + archive_moaco + archive_nsga))
            results["NSGAII"]["coverage"].append(compute_coverage(archive_nsga, archive_hho + archive_moaco + archive_pso))
            
            results["MOACO"]["coverage_hho"].append(compute_coverage(archive_moaco, archive_hho))
            results["MOACO"]["coverage_pso"].append(compute_coverage(archive_moaco, archive_pso))
            results["MOACO"]["coverage_nsga"].append(compute_coverage(archive_moaco, archive_nsga))

            results["MOHHO"]["coverage_aco"].append(compute_coverage(archive_hho, archive_moaco))
            results["MOHHO"]["coverage_pso"].append(compute_coverage(archive_hho, archive_pso))
            results["MOHHO"]["coverage_nsga"].append(compute_coverage(archive_hho, archive_nsga))

            results["PSO"]["coverage_aco"].append(compute_coverage(archive_pso, archive_moaco))
            results["PSO"]["coverage_hho"].append(compute_coverage(archive_pso, archive_hho))
            results["PSO"]["coverage_nsga"].append(compute_coverage(archive_pso, archive_nsga))

            results["NSGAII"]["coverage_hho"].append(compute_coverage(archive_nsga, archive_hho))
            results["NSGAII"]["coverage_pso"].append(compute_coverage(archive_nsga, archive_pso))
            results["NSGAII"]["coverage_aco"].append(compute_coverage(archive_nsga, archive_moaco))
            
        
    fixed_ref = utils.compute_fixed_reference(archives_all)
    global_lower_bound = utils.compute_combined_ideal(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    logging.info(f"Combined ideal (global lower bound): {global_lower_bound}")
    
    all_archives_total = []
    for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]:
        for archive in archives_all[alg]:
            all_archives_total.append(archive)
    extreme_bounds = utils.compute_extremes(all_archives_total)
    for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]:
        for archive in archives_all[alg]:
            norm_hv = normalized_hypervolume_fixed(archive, fixed_ref)
            abs_hv = absolute_hypervolume_fixed(archive, fixed_ref, global_lower_bound)
            sp = compute_spread(archive)
            rsp = compute_spread_3d_by_projections(archive, extreme_bounds)
            results[alg]["normalized_hypervolume"].append(norm_hv)
            results[alg]["absolute_hypervolume"].append(abs_hv)
            results[alg]["spread"].append(sp)
            results[alg]["multi_objective_spread"].append(rsp)

    union_archive = [entry for alg in archives_all for archive in archives_all[alg] for entry in archive]
    all_objs = np.array([obj for (_, obj) in union_archive])
    true_pareto = utils.get_true_pareto_points(all_objs)
    gd_results = {"MOHHO": [], "PSO": [], "MOACO": [], "NSGAII": []}
    for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]:
        for archive in archives_all[alg]:
            logging.info(f"{alg} archive size: {len(archive)}")
            if archive != []:
                gd = compute_generational_distance(archive, true_pareto) if archive and true_pareto.size > 0 else None
            else:
                gd = -1
            gd_results[alg].append(gd)
    results["Generational_Distance"] = gd_results
    
    schedule, _ = model.compute_schedule(archives_all["PSO"][0][0][0])
    plot_gantt(schedule, "random schedule")
    return results, archives_all, base_schedules, convergence_curves


if __name__ == '__main__':
    utils.initialize_seed(69)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    runs = 1
    use_random_instance = True
    num_tasks = 20
    POPULATION = 100
    ITERATIONS = 500  # Maximum iterations (may not be reached if time_limit is hit)
    TIME_LIMIT = 30  # seconds (1 minute per algorithm run)
    ericsson = True

    results, archives_all, base_schedules, convergence_curves = run_experiments(
        runs=runs, use_random_instance=use_random_instance, num_tasks=num_tasks,
        population=POPULATION, iterrations=ITERATIONS, time_limit=TIME_LIMIT, ericsson=ericsson
    )
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    #means, stds = statistical_analysis(results)
    plot_convergence({alg: results[alg]["absolute_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]}, "Absolute Hypervolume (%)")
    plot_convergence({alg: results[alg]["normalized_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]}, "Normalized Hypervolume (%)")
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]}, "Spread (Diversity)")
    plot_convergence({alg: results[alg]["multi_objective_spread"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]}, "Multi objective spread (3D)")
    plot_convergence({alg: results[alg]["coverage"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]}, "Coverage (Algorithm dominates x% of the other algorithms)")
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    plot_convergence({alg: results[alg]["runtimes"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII"]}, "Runtimes")

    fixed_ref = utils.compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    archives = []


    for alg in archives_all:
        temp_archive = []
        for run in archives_all[alg]:
            for sol, obj in run:
                temp_archive = utils.update_archive_with_crowding(temp_archive, (sol, obj))
        archives.append(temp_archive)
    

    plot_all_pareto_graphs(archives, ["MOHHO", "PSO", "MOACO", "NSGAII"], ['o', '^', 's', 'd'], ['blue', 'red', 'green', 'purple'], fixed_ref)
    
    last_baseline = base_schedules[-1]
    plot_gantt(last_baseline, f"Baseline Schedule (Greedy Allocation)")
    
    logging.info("Experiment complete. Results saved to 'experiment_results.json'.")

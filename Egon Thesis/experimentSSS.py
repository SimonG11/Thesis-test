import json, logging
import numpy as np
from typing import Dict, List, Tuple, Any
from rcpsp_model import RCPSPModel
from tasks import get_default_tasks, generate_random_tasks
from algorithmsS import MOHHO, PSO, MOACO
from metrics import (normalized_hypervolume_fixed, absolute_hypervolume_fixed, 
                     compute_generational_distance, compute_spread, 
                     compute_spread_3d_by_projections, compute_coverage,
                     statistical_analysis)
from visualization import plot_gantt, plot_convergence, plot_pareto_2d, plot_all_pareto_graphs, plot_comparative_bar_chart, plot_aggregate_convergence
from scipy.stats import f_oneway
from objectives import objective_makespan, objective_total_cost, objective_neg_utilization, multi_objective
from ericsson_tasks import get_ericsson_tasks
import utilsS as utils
import time
from tqdm import tqdm

# -------------------- NEW: Imports for pymoo --------------------
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA  # Using SMS-EMOA from pymoo
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

# Instead of importing get_reference_directions from pymoo, we define our own:
def get_reference_directions(n_obj, n_partitions):
    """
    Generate reference directions on the unit simplex for NSGA-III using the Das-Dennis approach.
    
    Parameters:
        n_obj (int): Number of objectives.
        n_partitions (int): Number of divisions along each axis.
        
    Returns:
        numpy.ndarray: An array of shape (n_points, n_obj) with reference directions.
    """
    def recursive_build(n, left, current, results):
        if n == 1:
            results.append(current + [left])
        else:
            for i in range(left+1):
                recursive_build(n-1, left-i, current + [i], results)
    results = []
    recursive_build(n_obj, n_partitions, [], results)
    directions = np.array(results, dtype=float) / n_partitions
    return directions

def run_experiments(runs: int = 1, use_random_instance: bool = False, num_tasks: int = 10,
                    iterrations: int = 30, population: int = 5, time_limit: float = None, ericsson: bool = False
                   ) -> Tuple[Dict[str, Any],
                              Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]],
                              List[Dict[str, Any]],
                              Dict[str, List[List[float]]]]:
    """
    Run experiments for MOHHO, PSO, MOACO, a baseline, and several benchmarks using pymoo:
    NSGA-II, NSGA-III, SPEA2, MOEA/D, and SMS-EMOA.
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
    
    # Extend results for new algorithms
    results = {
        "MOHHO": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                  "spread": [], "multi_objective_spread": [], "coverage": [], "coverage_pso": [],
                  "coverage_aco": [], "runtimes": []},
        "PSO": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                "spread": [], "multi_objective_spread": [], "coverage": [], "coverage_hho": [],
                "coverage_aco": [], "runtimes": []},
        "MOACO": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                  "spread": [], "multi_objective_spread": [], "coverage": [], "coverage_pso": [],
                  "coverage_hho": [], "runtimes": []},
        "Baseline": {"makespan": []},
        "NSGAII": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                   "spread": [], "multi_objective_spread": [], "coverage": [], "runtimes": []},
        "NSGAIII": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                    "spread": [], "multi_objective_spread": [], "coverage": [], "runtimes": []},
        "SPEA2": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                  "spread": [], "multi_objective_spread": [], "coverage": [], "runtimes": []},
        "MOEAD": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                  "spread": [], "multi_objective_spread": [], "coverage": [], "runtimes": []},
        "SMSEMOA": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [],
                    "spread": [], "multi_objective_spread": [], "coverage": [], "runtimes": []}
    }
    archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]] = {
        "MOHHO": [], "PSO": [], "MOACO": [], "NSGAII": [],
        "NSGAIII": [], "SPEA2": [], "MOEAD": [], "SMSEMOA": []
    }
    base_schedules = []
    convergence_curves = {
        "MOHHO": [], "PSO": [], "MOACO": [], "NSGAII": [],
        "NSGAIII": [], "SPEA2": [], "MOEAD": [], "SMSEMOA": []
    }

    # Outer progress bar: overall experiment runs
    for run in tqdm(range(runs), desc="Experiment Runs", position=0, leave=True):
        tqdm.write(f"Starting run {run+1}/{runs}...")
        base_schedule, base_ms = model.baseline_allocation()
        results["Baseline"]["makespan"].append(base_ms)
        base_schedules.append(base_schedule)
        
        # Update inner progress bar total to reflect all 8 algorithms
        with tqdm(total=8, desc="Algorithms", position=1, leave=True) as algo_bar:
            
            # -------------------- MOHHO --------------------
            algo_bar.set_description("MOHHO")
            start_time = time.time()
            archive_hho, conv_hho = MOHHO(lambda x: multi_objective(x, model),
                                          lb_current, ub_current, dim,
                                          population, iterrations,
                                          time_limit=time_limit)
            runtime = time.time() - start_time
            results["MOHHO"]["runtimes"].append(runtime)
            best_ms_hho = min(archive_hho, key=lambda entry: entry[1][0])[1][0] if archive_hho else None
            results["MOHHO"]["best_makespan"].append(best_ms_hho)
            archives_all["MOHHO"].append(archive_hho)
            convergence_curves["MOHHO"].append(conv_hho)
            algo_bar.write("MOHHO Done")
            algo_bar.update(1)
            
            # -------------------- PSO --------------------
            algo_bar.set_description("PSO")
            objectives = [lambda x: multi_objective(x, model)]
            optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                            pop=population, c2=2, w_max=0.9, w_min=0.4,
                            disturbance_rate_min=0.1, disturbance_rate_max=0.2, jump_interval=75)
            start_time = time.time()
            conv_pso = optimizer.run(max_iter=iterrations, time_limit=time_limit)
            runtime = time.time() - start_time
            results["PSO"]["runtimes"].append(runtime)
            archive_pso = optimizer.archive
            best_ms_pso = min(archive_pso, key=lambda entry: entry[1][0])[1][0] if archive_pso else None
            results["PSO"]["best_makespan"].append(best_ms_pso)
            archives_all["PSO"].append(archive_pso)
            convergence_curves["PSO"].append(conv_pso)
            algo_bar.write("PSO Done")
            algo_bar.update(1)
            
            # -------------------- MOACO --------------------
            algo_bar.set_description("MOACO")
            start_time = time.time()
            archive_moaco, conv_moaco = MOACO(
                lambda x: multi_objective(x, model),
                model.tasks, lb_current, ub_current, population, iterrations,
                alpha=1.0, beta=2.0, evaporation_rate=0.1,
                colony_count=(2),
                time_limit=time_limit
            )
            runtime = time.time() - start_time
            results["MOACO"]["runtimes"].append(runtime)
            best_ms_moaco = min(archive_moaco, key=lambda entry: entry[1][0])[1][0] if archive_moaco else None
            results["MOACO"]["best_makespan"].append(best_ms_moaco)
            # Coverage metrics among MOACO, MOHHO and PSO
            results["MOACO"]["coverage"].append(compute_coverage(archive_moaco, archive_hho + archive_pso))
            results["MOHHO"]["coverage"].append(compute_coverage(archive_hho, archive_moaco + archive_pso))
            results["PSO"]["coverage"].append(compute_coverage(archive_pso, archive_hho + archive_moaco))
            results["MOACO"]["coverage_hho"].append(compute_coverage(archive_moaco, archive_hho))
            results["MOHHO"]["coverage_aco"].append(compute_coverage(archive_hho, archive_moaco))
            results["PSO"]["coverage_hho"].append(compute_coverage(archive_pso, archive_hho))
            results["MOACO"]["coverage_pso"].append(compute_coverage(archive_moaco, archive_pso))
            results["MOHHO"]["coverage_pso"].append(compute_coverage(archive_hho, archive_pso))
            results["PSO"]["coverage_aco"].append(compute_coverage(archive_pso, archive_moaco))
            archives_all["MOACO"].append(archive_moaco)
            convergence_curves["MOACO"].append(conv_moaco)
            algo_bar.write("MOACO Done")
            algo_bar.update(1)
            
            # -------------------- NSGA-II via pymoo --------------------
            algo_bar.set_description("NSGA-II")
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
                           termination=('n_gen', iterrations),
                           seed=14,
                           verbose=False)
            nsga_runtime = time.time() - start_time
            archive_nsga = [(sol, obj) for sol, obj in zip(res.X, res.F)]
            best_ms_nsga = min(archive_nsga, key=lambda entry: entry[1][0])[1][0] if archive_nsga else None
            results["NSGAII"]["runtimes"].append(nsga_runtime)
            results["NSGAII"]["best_makespan"].append(best_ms_nsga)
            archives_all["NSGAII"].append(archive_nsga)
            convergence_curves["NSGAII"].append([])  # NSGA-II convergence curve not recorded here
            algo_bar.write("NSGA-II Done")
            algo_bar.update(1)
            
            # -------------------- NSGA-III via pymoo --------------------
            algo_bar.set_description("NSGA-III")
            # Generate reference directions using our own helper function
            sample_obj = multi_objective(np.array([lb_current[i] for i in range(dim)]), model)
            n_obj = len(sample_obj) if isinstance(sample_obj, (list, np.ndarray)) else 1
            ref_dirs = get_reference_directions(n_obj, n_partitions=12)
            algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=population)
            start_time = time.time()
            res = minimize(problem,
                           algorithm,
                           termination=('n_gen', iterrations),
                           seed=14,
                           verbose=False)
            nsga3_runtime = time.time() - start_time
            archive_nsga3 = [(sol, obj) for sol, obj in zip(res.X, res.F)]
            best_ms_nsga3 = min(archive_nsga3, key=lambda entry: entry[1][0])[1][0] if archive_nsga3 else None
            results["NSGAIII"]["runtimes"].append(nsga3_runtime)
            results["NSGAIII"]["best_makespan"].append(best_ms_nsga3)
            archives_all["NSGAIII"].append(archive_nsga3)
            convergence_curves["NSGAIII"].append([])
            algo_bar.write("NSGA-III Done")
            algo_bar.update(1)
            
            # -------------------- SPEA2 via pymoo --------------------
            algo_bar.set_description("SPEA2")
            algorithm = SPEA2(pop_size=population)
            start_time = time.time()
            res = minimize(problem,
                           algorithm,
                           termination=('n_gen', iterrations),
                           seed=14,
                           verbose=False)
            spea2_runtime = time.time() - start_time
            archive_spea2 = [(sol, obj) for sol, obj in zip(res.X, res.F)]
            best_ms_spea2 = min(archive_spea2, key=lambda entry: entry[1][0])[1][0] if archive_spea2 else None
            results["SPEA2"]["runtimes"].append(spea2_runtime)
            results["SPEA2"]["best_makespan"].append(best_ms_spea2)
            archives_all["SPEA2"].append(archive_spea2)
            convergence_curves["SPEA2"].append([])
            algo_bar.write("SPEA2 Done")
            algo_bar.update(1)

            # -------------------- MOEA/D via pymoo --------------------
            algo_bar.set_description("MOEA/D")
            # Generate reference directions for MOEAD
            ref_dirs_moead = get_reference_directions(n_obj, n_partitions=12)
            # Remove pop_size parameter, since MOEAD calculates it from ref_dirs
            algorithm = MOEAD(ref_dirs=ref_dirs_moead)
            start_time = time.time()
            res = minimize(problem,
                           algorithm,
                           termination=('n_gen', iterrations),
                           seed=14,
                           verbose=False)
            moead_runtime = time.time() - start_time
            archive_moead = [(sol, obj) for sol, obj in zip(res.X, res.F)]
            best_ms_moead = min(archive_moead, key=lambda entry: entry[1][0])[1][0] if archive_moead else None
            results["MOEAD"]["runtimes"].append(moead_runtime)
            results["MOEAD"]["best_makespan"].append(best_ms_moead)
            archives_all["MOEAD"].append(archive_moead)
            convergence_curves["MOEAD"].append([])
            algo_bar.write("MOEA/D Done")
            algo_bar.update(1)


            
            # -------------------- SMS-EMOA via pymoo --------------------
            algo_bar.set_description("SMS-EMOA")
            algorithm = SMSEMOA(pop_size=population)
            start_time = time.time()
            res = minimize(problem,
                           algorithm,
                           termination=('n_gen', iterrations),
                           seed=14,
                           verbose=False)
            smsemoa_runtime = time.time() - start_time
            archive_smsemoa = [(sol, obj) for sol, obj in zip(res.X, res.F)]
            best_ms_smsemoa = min(archive_smsemoa, key=lambda entry: entry[1][0])[1][0] if archive_smsemoa else None
            results["SMSEMOA"]["runtimes"].append(smsemoa_runtime)
            results["SMSEMOA"]["best_makespan"].append(best_ms_smsemoa)
            archives_all["SMSEMOA"].append(archive_smsemoa)
            convergence_curves["SMSEMOA"].append([])
            algo_bar.write("SMS-EMOA Done")
            algo_bar.update(1)
        
    # Compute fixed reference and global lower bound based on all archives from all algorithms
    fixed_ref = utils.compute_fixed_reference(archives_all)
    global_lower_bound = utils.compute_combined_ideal(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    logging.info(f"Combined ideal (global lower bound): {global_lower_bound}")
    
    all_archives_total = []
    for alg in archives_all:
        for archive in archives_all[alg]:
            all_archives_total.append(archive)
    extreme_bounds = utils.compute_extremes(all_archives_total)
    for alg in archives_all:
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
    gd_results = {"MOHHO": [], "PSO": [], "MOACO": [], "NSGAII": [], "NSGAIII": [], "SPEA2": [], "MOEAD": [], "SMSEMOA": []}
    for alg in archives_all:
        for archive in archives_all[alg]:
            logging.info(f"{alg} archive size: {len(archive)}")
            gd = compute_generational_distance(archive, true_pareto) if archive and true_pareto.size > 0 else None
            gd_results[alg].append(gd)
    results["Generational_Distance"] = gd_results
    
    schedule, _ = model.compute_schedule(archives_all["PSO"][0][0][0])
    plot_gantt(schedule, "random schedule")
    return results, archives_all, base_schedules, convergence_curves

if __name__ == '__main__':
    utils.initialize_seed(14)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    runs = 1
    use_random_instance = True
    num_tasks = 50
    POPULATION = 100
    ITERATIONS = 200  # Maximum iterations (may not be reached if time_limit is hit)
    TIME_LIMIT = 180  # seconds (1 minute per algorithm run)
    ericsson = False

    results, archives_all, base_schedules, convergence_curves = run_experiments(
        runs=runs, use_random_instance=use_random_instance, num_tasks=num_tasks,
        population=POPULATION, iterrations=ITERATIONS, time_limit=TIME_LIMIT, ericsson=ericsson
    )
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    means, stds = statistical_analysis(results)
    plot_convergence({alg: results[alg]["best_makespan"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]}, "Best Makespan (hours)")
    plot_convergence({alg: results[alg]["absolute_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]}, "Absolute Hypervolume (%)")
    plot_convergence({alg: results[alg]["normalized_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]}, "Normalized Hypervolume (%)")
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]}, "Spread (Diversity)")
    plot_convergence({alg: results[alg]["multi_objective_spread"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]}, "Multi objective spread (3D)")
    plot_convergence({alg: results[alg]["coverage"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]}, "Coverage (Algorithm dominates x% of the other algorithms)")
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    plot_convergence({alg: results[alg]["runtimes"] for alg in ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]}, "Runtimes")

    fixed_ref = utils.compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    archives = []
    for alg in archives_all:
        temp_archive = []
        for run in archives_all[alg]:
            for sol, obj in run:
                temp_archive = utils.update_archive_with_crowding(temp_archive, (sol, obj))
        archives.append(temp_archive)
    # Update markers and colors for the 8 algorithms
    markers = ['o', '^', 's', 'd', 'v', 'p', 'h', '*']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    algo_names = ["MOHHO", "PSO", "MOACO", "NSGAII", "NSGAIII", "SPEA2", "MOEAD", "SMSEMOA"]
    plot_pareto_2d(archives, algo_names, markers, colors, ref_point=fixed_ref)
    plot_all_pareto_graphs(archives, algo_names, markers, colors, fixed_ref)
    
    last_baseline = base_schedules[-1]
    last_makespan = results["Baseline"]["makespan"][-1]
    plot_gantt(last_baseline, f"Baseline Schedule (Greedy Allocation)\nMakespan: {last_makespan:.2f} hrs")
    
    plot_aggregate_convergence(convergence_curves, "Aggregate Convergence Curves for All Algorithms")
    
    logging.info("Experiment complete. Results saved to 'experiment_results.json'.")

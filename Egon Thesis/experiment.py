# experiment.py
import json, logging
import numpy as np
from typing import Dict, List, Tuple, Any
from rcpsp_model import RCPSPModel
from tasks import get_default_tasks, generate_random_tasks
from algorithms import MOHHO_with_progress, PSO, MOACO_improved
from metrics import (compute_fixed_reference, compute_combined_ideal, 
                     normalized_hypervolume_fixed, absolute_hypervolume_fixed, 
                     compute_generational_distance, compute_spread, 
                     compute_spread_3d_by_projections, compute_coverage)
from visualization import plot_gantt, plot_convergence, plot_pareto_2d, plot_all_pareto_graphs, plot_comparative_bar_chart, plot_aggregate_convergence
from scipy.stats import f_oneway
from objectives import objective_makespan, objective_total_cost, objective_neg_utilization, multi_objective
import utils


def run_experiments(runs: int = 1, use_random_instance: bool = False, num_tasks: int = 10, iterrations: int = 30, population: int = 5
                   ) -> Tuple[Dict[str, Any], Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]], List[Dict[str, Any]], Dict[str, List[List[float]]]]:
    """
    Run experiments for MOHHO, PSO, MOACO, and a baseline.
    Returns results, archives, baseline schedules, and convergence curves.
    """
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
        "MOHHO": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [], "spread": [], "multi_objective_spread": [], "coverage" : [], "coverage_pso": [], "coverage_aco": []},
        "PSO": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [], "spread": [], "multi_objective_spread": [], "coverage" : [], "coverage_hho": [], "coverage_aco": []},
        "MOACO": {"best_makespan": [], "normalized_hypervolume": [], "absolute_hypervolume": [], "spread": [], "multi_objective_spread": [], "coverage" : [], "coverage_pso": [], "coverage_hho": []},
        "Baseline": {"makespan": []}
    }
    archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]] = {"MOHHO": [], "PSO": [], "MOACO": []}
    base_schedules = []
    convergence_curves = {"MOHHO": [], "PSO": [], "MOACO": []}

    for run in range(runs):
        logging.info(f"Run {run+1}/{runs}...")
        base_schedule, base_ms = model.baseline_allocation()
        results["Baseline"]["makespan"].append(base_ms)
        base_schedules.append(base_schedule)

        logging.info(f"Initializing MOHHO run {run+1}...")
        hho_iter = iterrations
        search_agents_no = population
        archive_hho, conv_hho = MOHHO_with_progress(lambda x: multi_objective(x, model), lb_current, ub_current, dim, search_agents_no, hho_iter)
        best_ms_hho = min(archive_hho, key=lambda entry: entry[1][0])[1][0] if archive_hho else None
        results["MOHHO"]["best_makespan"].append(best_ms_hho)
        archives_all["MOHHO"].append(archive_hho)
        convergence_curves["MOHHO"].append(conv_hho)
        logging.info(f"MOHHO Done")
        logging.info(f"Initializing MOPSO run {run+1}...")
        objectives = [lambda x: objective_makespan(x, model),
                      lambda x: objective_total_cost(x, model),
                      lambda x: objective_neg_utilization(x, model)]
        optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                        pop=population, c2=1.05, w_max=0.9, w_min=0.4,
                        disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
        conv_pso = optimizer.run(max_iter=iterrations)
        archive_pso = optimizer.archive
        best_ms_pso = min(archive_pso, key=lambda entry: entry[1][0])[1][0] if archive_pso else None
        results["PSO"]["best_makespan"].append(best_ms_pso)
        archives_all["PSO"].append(archive_pso)
        convergence_curves["PSO"].append(conv_pso)

        logging.info(f"MOPSO Done")
        logging.info(f"Initializing MOACO run {run+1}...")
        ant_count = population
        moaco_iter = iterrations
        archive_moaco, conv_moaco = MOACO_improved(lambda x: multi_objective(x, model), model.tasks, workers,
                                                   lb_current, ub_current, ant_count, moaco_iter,
                                                   alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0)
        best_ms_moaco = min(archive_moaco, key=lambda entry: entry[1][0])[1][0] if archive_moaco else None
        results["MOACO"]["best_makespan"].append(best_ms_moaco)
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
        logging.info(f"MOACO Done")

    fixed_ref = compute_fixed_reference(archives_all)
    global_lower_bound = compute_combined_ideal(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    logging.info(f"Combined ideal (global lower bound): {global_lower_bound}")
    all_archives_total = []
    algs = ["MOHHO", "PSO", "MOACO"]
    for alg in algs:
        for archive in archives_all[alg]:
            all_archives_total.append(archive)
    extreme_bounds = utils.compute_extremes(all_archives_total)
    for alg in algs:
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
    # Extract all objective vectors from the union.
    all_objs = np.array([obj for (_, obj) in union_archive])
    
    # Compute the true Pareto front (non-dominated set) using the helper function.
    true_pareto = utils.get_true_pareto_points(all_objs)
    gd_results = {"MOHHO": [], "PSO": [], "MOACO": []}
    for alg in ["MOHHO", "PSO", "MOACO"]:
        for archive in archives_all[alg]:
            gd = compute_generational_distance(archive, true_pareto) if archive and true_pareto.size > 0 else None
            gd_results[alg].append(gd)
    results["Generational_Distance"] = gd_results

    return results, archives_all, base_schedules, convergence_curves


def statistical_analysis(results: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    algos = ["MOHHO", "PSO", "MOACO", "Baseline"]
    means, stds, data = {}, {}, {}
    data["Baseline"] = results["Baseline"]["makespan"]
    for algo in ["MOHHO", "PSO", "MOACO"]:
        data[algo] = results[algo]["best_makespan"]
    for algo in algos:
        arr = np.array(data[algo])
        means[algo] = np.mean(arr)
        stds[algo] = np.std(arr)
        logging.info(f"{algo}: Mean = {means[algo]:.2f}, Std = {stds[algo]:.2f}")
    if all(len(data[algo]) > 1 for algo in algos):
        F_stat, p_value = f_oneway(data["Baseline"], data["MOHHO"], data["PSO"], data["MOACO"])
        logging.info(f"ANOVA: F = {F_stat:.2f}, p = {p_value:.4f}")
    else:
        logging.warning("Not enough data for ANOVA.")
    return means, stds


def grid_search_pso_population(pop_sizes: List[int], runs_per_config: int = 3, model: RCPSPModel = None,
                               lb: np.ndarray = None, ub: np.ndarray = None, dim: int = None) -> Dict[int, Tuple[float, float]]:
    results_grid = {}
    for pop in pop_sizes:
        best_makespans = []
        for _ in range(runs_per_config):
            objectives = [lambda x: objective_makespan(x, model),
                          lambda x: objective_total_cost(x, model),
                          lambda x: objective_neg_utilization(x, model)]
            optimizer = PSO(dim=dim, lb=lb, ub=ub, obj_funcs=objectives,
                            pop=pop, c2=1.05, w_max=0.9, w_min=0.4,
                            disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
            _ = optimizer.run(max_iter=30)
            archive = optimizer.archive
            if archive:
                best = min(archive, key=lambda entry: entry[1][0])[1][0]
                best_makespans.append(best)
        if best_makespans:
            avg = np.mean(best_makespans)
            std = np.std(best_makespans)
            results_grid[pop] = (avg, std)
            logging.info(f"PSO pop size {pop}: Avg best makespan = {avg:.2f}, Std = {std:.2f}")
    return results_grid


if __name__ == '__main__':
    utils.initialize_seed(4)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    runs = 5
    use_random_instance = False
    num_tasks = 10
    POPULATION = 10
    ITERATIONS = 30
    tasks_for_exp = generate_random_tasks(num_tasks, {"Developer": 10, "Manager": 2, "Tester": 3}) if use_random_instance else get_default_tasks()

    results, archives_all, base_schedules, convergence_curves = run_experiments(runs=runs, use_random_instance=use_random_instance, num_tasks=num_tasks, population=POPULATION, iterrations=ITERATIONS)
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    means, stds = statistical_analysis(results)
    plot_convergence({alg: results[alg]["best_makespan"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Best Makespan (hours)")
    plot_convergence({alg: results[alg]["absolute_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Absolute Hypervolume (%)")
    plot_convergence({alg: results[alg]["normalized_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Normalized Hypervolume (%)")
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Spread (Diversity)")
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    
    fixed_ref = compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    last_archives = [archives_all[alg][-1] for alg in ["MOHHO", "PSO", "MOACO"]]
    plot_pareto_2d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'], ref_point=fixed_ref)
    plot_all_pareto_graphs(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'], fixed_ref)
    
    last_baseline = base_schedules[-1]
    last_makespan = results["Baseline"]["makespan"][-1]
    plot_gantt(last_baseline, f"Baseline Schedule (Greedy Allocation)\nMakespan: {last_makespan:.2f} hrs")
    
    plot_aggregate_convergence(convergence_curves, "Aggregate Convergence Curves for All Algorithms")
    
    pop_sizes = [10, 20, 30]
    workers = {"Developer": 10, "Manager": 2, "Tester": 3}
    default_tasks = get_default_tasks()
    model_for_grid = RCPSPModel(default_tasks, workers, {"Developer": 50, "Manager": 75, "Tester": 40})
    lb_array = np.array([task["min"] for task in default_tasks])
    ub_array = np.array([task["max"] for task in default_tasks])
    grid_results = grid_search_pso_population(pop_sizes, runs_per_config=3, model=model_for_grid,
                                              lb=lb_array, ub=ub_array, dim=len(default_tasks))
    
    logging.info("Experiment complete. Results saved to 'experiment_results.json'.")

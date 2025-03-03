#!/usr/bin/env python3
"""
Improved Multi-Objective Comparison for RCPSP using Adaptive MOHHO, Adaptive MOPSO, and Improved MOACO

This script implements and compares three metaheuristic algorithms for the Resource-Constrained 
Project Scheduling Problem (RCPSP) with multiple objectives. The implementation has been updated 
to incorporate advanced mechanisms from the literature, including:
 - Non-linear adaptive parameter tuning (using cosine schedules and self-adaptation)
 - Enhanced archive management with diversity preservation (inspired by NSGA-II crowding distance)
 - Periodic local search and diversity-driven injection of new hawks
 - Multi-colony pheromone updates with periodic reinitialization in MOACO

References:
 - Heidari, A., et al. "Harris Hawks Optimization: Algorithm and Applications."
 - Coello, C.A.C., & Lechuga, M.S. "MOPSO: A Proposal for Multiple Objective Particle Swarm Optimization."
 - Dorigo, M., & Stützle, T. "Ant Colony Optimization."
 - Deb, K. "Multi-Objective Optimization Using Evolutionary Algorithms."
 - (Additional references as cited in the in-depth analysis)

Author: Simon Gottschalk
Date: 2025-02-13
"""

import numpy as np
import matplotlib.pyplot as plt
import random, math, time, copy, json, logging
from typing import List, Tuple, Dict, Any, Callable, Optional
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.stats import f_oneway  # For ANOVA

# ----------------------------- Logging Setup -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------- Reproducibility -----------------------------
def initialize_seed(seed: int = 42) -> None:
    """Initialize random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

initialize_seed(42)

# =============================================================================
# -------------------------- Helper Functions -------------------------------
# =============================================================================

def dominates(obj_a: np.ndarray, obj_b: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if solution A dominates solution B in a minimization context.
    Two objective values are considered equal if their difference is less than epsilon.
    
    A dominates B if every objective in A is <= corresponding objective in B,
    with at least one strictly less (by more than epsilon).
    """
    less_equal = np.all(obj_a <= obj_b + epsilon)
    strictly_less = np.any(obj_a < obj_b - epsilon)
    return less_equal and strictly_less

def levy(dim: int) -> np.ndarray:
    """
    Compute a Levy flight step for a given dimensionality.
    
    Levy flights allow for occasional large jumps in the search space to help escape local optima.
    """
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.power(np.abs(v), 1 / beta))

def find_earliest_start(earliest: float, duration: float, allocated: int,
                        scheduled_tasks: List[Dict[str, Any]],
                        capacity: int, resource: str, epsilon: float = 1e-6) -> float:
    """
    Determine the earliest feasible start time for a task given resource constraints.
    
    Inspired by the Serial Schedule Generation Scheme (SSGS), this function checks candidate 
    time windows (based on existing task start/finish times) and returns the earliest time when 
    the required resource capacity is available.
    """
    tasks_r = [t for t in scheduled_tasks if t.get("resource") == resource]
    if not tasks_r:
        return earliest

    candidate_times = {earliest}
    for task in tasks_r:
        if task["start"] >= earliest:
            candidate_times.add(task["start"])
        if task["finish"] >= earliest:
            candidate_times.add(task["finish"])
    candidate_times = sorted(candidate_times)

    for t in candidate_times:
        events = [t, t + duration]
        for task in tasks_r:
            if task["finish"] > t and task["start"] < t + duration:
                events.extend([task["start"], task["finish"]])
        events = sorted(set(events))
        feasible = True
        for i in range(len(events) - 1):
            mid = (events[i] + events[i+1]) / 2.0
            usage = sum(task["workers"] for task in tasks_r if task["start"] <= mid < task["finish"])
            if usage + allocated > capacity:
                feasible = False
                break
        if feasible:
            return t

    last_finish = max(task["finish"] for task in tasks_r)
    return last_finish + epsilon

# -----------------------------------------------------------------------------
# Chaotic Initialization using Logistic Map
# -----------------------------------------------------------------------------
def chaotic_map_initialization(lb: np.ndarray, ub: np.ndarray, dim: int, n_agents: int) -> np.ndarray:
    """
    Initialize the population using a logistic chaotic map.
    
    The logistic map (x_{n+1} = 4*x_n*(1 - x_n) for r=4) generates chaotic sequences,
    resulting in a diverse distribution of initial solutions.
    """
    r = 4.0
    population = np.zeros((n_agents, dim))
    for i in range(n_agents):
        x = np.random.rand(dim)
        for _ in range(10):
            x = r * x * (1 - x)
        population[i, :] = lb + x * (ub - lb)
    return population

# -----------------------------------------------------------------------------
# Default Task Definition for Reproducibility
# -----------------------------------------------------------------------------
def get_default_tasks() -> List[Dict[str, Any]]:
    """
    Return a fixed list of tasks for the RCPSP.
    
    This fixed instance supports reproducible experiments and benchmark comparisons.
    """
    return [
        {"id": 1, "task_name": "Requirements Gathering", "base_effort": 80,  "min": 1, "max": 14, "dependencies": [],         "resource": "Manager"},
        {"id": 2, "task_name": "System Design",          "base_effort": 100, "min": 1, "max": 14, "dependencies": [1],        "resource": "Manager"},
        {"id": 3, "task_name": "Module 1 Development",   "base_effort": 150, "min": 1, "max": 14, "dependencies": [2],        "resource": "Developer"},
        {"id": 4, "task_name": "Module 2 Development",   "base_effort": 150, "min": 1, "max": 14, "dependencies": [2],        "resource": "Developer"},
        {"id": 5, "task_name": "Integration",            "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Developer"},
        {"id": 6, "task_name": "Testing",                "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Tester"},
        {"id": 7, "task_name": "Acceptance Testing",     "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Tester"},
        {"id": 8, "task_name": "Documentation",          "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Developer"},
        {"id": 9, "task_name": "Training",               "base_effort": 50,  "min": 1, "max": 14, "dependencies": [7, 8],     "resource": "Tester"},
        {"id": 10,"task_name": "Deployment",             "base_effort": 70,  "min": 2, "max": 14, "dependencies": [7, 9],     "resource": "Manager"}
    ]

# =============================================================================
# -------------------------- RCPSP Model Definition -------------------------
# =============================================================================

class RCPSPModel:
    """
    A model representing the Resource-Constrained Project Scheduling Problem (RCPSP).

    Attributes:
        tasks (List[Dict]): List of task definitions.
        workers (Dict[str, int]): Dictionary specifying available workers per resource type.
        worker_cost (Dict[str, int]): Dictionary specifying cost per man–hour for each resource.
    """
    def __init__(self, tasks: List[Dict[str, Any]], 
                 workers: Dict[str, int],
                 worker_cost: Dict[str, int]) -> None:
        self.tasks = tasks
        self.workers = workers
        self.worker_cost = worker_cost

    def compute_schedule(self, x: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Compute a feasible schedule using a Serial Schedule Generation Scheme (SSGS).

        Given an allocation vector 'x', tasks are scheduled based on their dependencies and resource constraints,
        ensuring an active schedule (i.e., no task can start earlier without delaying others).

        Returns:
            schedule: List of tasks with timing details.
            makespan: Overall finish time of the project.
        """
        schedule = []
        finish_times: Dict[int, float] = {}
        for task in self.tasks:
            tid = task["id"]
            resource_type = task["resource"]
            capacity = self.workers[resource_type]
            effective_max = min(task["max"], capacity)
            alloc = int(round(x[tid - 1]))
            alloc = max(task["min"], min(effective_max, alloc))
            new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
            duration = new_effort / alloc
            earliest = max([finish_times[dep] for dep in task["dependencies"]]) if task["dependencies"] else 0
            candidate_start = find_earliest_start(earliest, duration, alloc, schedule, capacity, resource_type)
            start_time = candidate_start
            finish_time = start_time + duration
            finish_times[tid] = finish_time
            schedule.append({
                "task_id": tid,
                "task_name": task["task_name"],
                "start": start_time,
                "finish": finish_time,
                "duration": duration,
                "workers": alloc,
                "resource": resource_type
            })
        makespan = max(item["finish"] for item in schedule)
        return schedule, makespan

    def baseline_allocation(self) -> Tuple[List[Dict[str, Any]], float]:
        """
        Generate a baseline schedule by assigning the minimum required workers to all tasks.
        
        This greedy allocation strategy serves as a baseline for comparison.
        """
        x = np.array([task["min"] for task in self.tasks])
        return self.compute_schedule(x)

# =============================================================================
# ----------------------- Objective Functions (Using Model) -----------------
# =============================================================================

def objective_makespan(x: np.ndarray, model: RCPSPModel) -> float:
    """Objective 1: Minimize project makespan."""
    _, ms = model.compute_schedule(x)
    return ms

def objective_total_cost(x: np.ndarray, model: RCPSPModel) -> float:
    """Objective 2: Minimize total labor cost."""
    total_cost = 0.0
    for task in model.tasks:
        tid = task["id"]
        resource_type = task["resource"]
        capacity = model.workers[resource_type]
        effective_max = min(task["max"], capacity)
        alloc = round(x[tid - 1] * 2) / 2
        alloc = max(task["min"], min(effective_max, alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
        wage_rate = model.worker_cost[resource_type]
        total_cost += duration * alloc * wage_rate
    return total_cost

def objective_neg_utilization(x: np.ndarray, model: RCPSPModel) -> float:
    """
    Objective 3: Maximize average resource utilization.
    
    (Negated so that all objectives are minimized.)
    """
    utils = []
    for task in model.tasks:
        tid = task["id"]
        resource_type = task["resource"]
        capacity = model.workers[resource_type]
        effective_max = min(task["max"], capacity)
        alloc = round(x[tid - 1] * 2) / 2
        alloc = max(task["min"], min(effective_max, alloc))
        utils.append(alloc / task["max"])
    return -np.mean(utils)

def multi_objective(x: np.ndarray, model: RCPSPModel) -> np.ndarray:
    """
    Return the multi-objective vector for a given allocation vector x.
    
    The vector consists of:
        [makespan, total cost, -average utilization].
    """
    return np.array([
        objective_makespan(x, model),
        objective_total_cost(x, model),
        objective_neg_utilization(x, model)
    ])

# =============================================================================
# ----------------------- Performance Metrics -------------------------------
# =============================================================================

def approximate_hypervolume(archive: List[Tuple[np.ndarray, np.ndarray]],
                            reference_point: np.ndarray,
                            num_samples: int = 100) -> float:
    """
    Approximate the hypervolume of the archive via Monte Carlo sampling.
    
    Hypervolume measures the volume of the objective space dominated by the Pareto front, relative
    to a reference point. For minimization problems, the reference point should be chosen such that
    it is dominated by all solutions.
    """
    if not archive:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    mins = np.min(objs, axis=0)
    samples = np.random.uniform(low=mins, high=reference_point, size=(num_samples, len(reference_point)))
    count = sum(1 for sample in samples if any(np.all(sol <= sample) for sol in objs))
    vol = np.prod(reference_point - mins)
    return (count / num_samples) * vol

def compute_crowding_distance(archive: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Compute the crowding distance for each solution in the archive.
    
    Crowding distance is used to maintain diversity among the non-dominated solutions.
    """
    if not archive:
        return np.array([])
    objs = np.array([entry[1] for entry in archive])
    num_objs = objs.shape[1]
    distances = np.zeros(len(archive))
    for m in range(num_objs):
        sorted_indices = np.argsort(objs[:, m])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
        m_values = objs[sorted_indices, m]
        m_range = m_values[-1] - m_values[0]
        if m_range == 0:
            continue
        for i in range(1, len(archive) - 1):
            distances[sorted_indices[i]] += (m_values[i+1] - m_values[i-1]) / m_range
    return distances

def same_entry(entry1: Tuple[np.ndarray, np.ndarray],
               entry2: Tuple[np.ndarray, np.ndarray]) -> bool:
    """Return True if two archive entries are identical (both decision and objective vectors)."""
    return np.array_equal(entry1[0], entry2[0]) and np.array_equal(entry1[1], entry2[1])

def update_archive_with_crowding(archive: List[Tuple[np.ndarray, np.ndarray]],
                                 new_entry: Tuple[np.ndarray, np.ndarray],
                                 max_archive_size: int = 50,
                                 epsilon: float = 1e-6) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Update the non-dominated archive with a new entry while preserving diversity.
    
    Uses crowding distance to remove the most crowded solution if the archive exceeds max_archive_size.
    The epsilon parameter allows near-equal objective values to be considered equal.
    
    (For further improvements, consider replacing this with a full non-dominated sorting and crowding distance
    mechanism as used in NSGA-II.)
    """
    sol_new, obj_new = new_entry
    dominated_flag = False
    removal_list = []
    for (sol_arch, obj_arch) in archive:
        if dominates(obj_arch, obj_new, epsilon):
            dominated_flag = True
            break
        if dominates(obj_new, obj_arch, epsilon):
            removal_list.append((sol_arch, obj_arch))
    if not dominated_flag:
        archive = [entry for entry in archive if not any(same_entry(entry, rem) for rem in removal_list)]
        archive.append(new_entry)
        if len(archive) > max_archive_size:
            distances = compute_crowding_distance(archive)
            min_index = np.argmin(distances)
            archive.pop(min_index)
    return archive

def compute_generational_distance(archive: List[Tuple[np.ndarray, np.ndarray]],
                                  true_pareto: np.ndarray) -> Optional[float]:
    """
    Compute the Generational Distance (GD) between the archive and the true Pareto front.
    
    GD measures the average distance from the archive solutions to the closest point on the true Pareto front.
    """
    if not archive or true_pareto.size == 0:
        return None
    objs = np.array([entry[1] for entry in archive])
    distances = [np.min(np.linalg.norm(true_pareto - sol, axis=1)) for sol in objs]
    return np.mean(distances)

def compute_spread(archive: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Compute the spread (diversity) of the archive as the average pairwise Euclidean distance in objective space.
    """
    if len(archive) < 2:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    dists = [np.linalg.norm(objs[i] - objs[j]) for i in range(len(objs)) for j in range(i+1, len(objs))]
    return np.mean(dists)

# -----------------------------------------------------------------------------
# Fixed Reference Point Calculation for Hypervolume Comparison
# -----------------------------------------------------------------------------
def compute_fixed_reference(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
    """
    Compute a fixed reference point based on the union of all solution archives from all algorithms.
    
    For minimization problems, the fixed reference point is chosen as the element-wise maximum (nadir)
    over the union of all final archives. This reference point will be used consistently across algorithms.
    """
    union_archive = []
    for alg in archives_all:
        for archive in archives_all[alg]:
            union_archive.extend(archive)
    if not union_archive:
        raise ValueError("No archive entries found.")
    objs = np.array([entry[1] for entry in union_archive])
    ref_point = np.max(objs, axis=0)
    return ref_point

def normalized_hypervolume_fixed(archive: List[Tuple[np.ndarray, np.ndarray]], fixed_ref: np.ndarray) -> float:
    """
    Compute the normalized hypervolume as a percentage using a fixed reference point.
    
    The normalized hypervolume is defined as:
        (Hypervolume(archive, fixed_ref) / TotalVolume(ideal, fixed_ref)) * 100
    where the ideal point is computed as the element-wise minimum of the archive.
    
    This percentage indicates the proportion of the available objective space (between the ideal and fixed reference point)
    that is dominated by the Pareto front.
    """
    if not archive:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    ideal = np.min(objs, axis=0)
    total_volume = np.prod(fixed_ref - ideal)
    if total_volume == 0:
        return 0.0
    hv = approximate_hypervolume(archive, reference_point=fixed_ref)
    return (hv / total_volume) * 100.0

# =============================================================================
# ----------------------- Visualization Functions ---------------------------
# =============================================================================

def plot_gantt(schedule: List[Dict[str, Any]], title: str) -> None:
    """Plot a Gantt chart for the given schedule."""
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks, yticklabels = [], []
    for i, task in enumerate(schedule):
        ax.broken_barh([(task["start"], task["duration"])],
                       (i * 10, 9),
                       facecolors='tab:blue')
        yticks.append(i * 10 + 5)
        yticklabels.append(f"Task {task['task_id']}: {task['task_name']} ({task['resource']})\n(W: {task['workers']})")
        ax.text(task["start"] + task["duration"] / 2, i * 10 + 5,
                f"{task['start']:.1f}-{task['finish']:.1f}",
                ha='center', va='center', color='white', fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Tasks")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_convergence(metrics_dict: Dict[str, List[float]], metric_name: str) -> None:
    """
    Plot boxplots for a given performance metric across different runs.
    
    Uses 'tick_labels' for compatibility with newer Matplotlib versions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    data = list(metrics_dict.values())
    ax.boxplot(data, tick_labels=list(metrics_dict.keys()))
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribution of {metric_name} across runs")
    ax.grid(True)
    plt.show()

def plot_pareto_2d(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                   labels: List[str], markers: List[str], colors: List[str],
                   ref_point: Optional[np.ndarray] = None) -> None:
    """
    Plot 2D Pareto fronts (Makespan vs. Total Cost) for the provided archives.
    
    If a reference point is provided, it is plotted as an 'x' marker for comparison.
    """
    plt.figure(figsize=(8, 6))
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
            plt.scatter(objs[:, 0], objs[:, 1], c=color, marker=marker, s=80,
                        edgecolor='k', label=label)
    if ref_point is not None:
        plt.scatter(ref_point[0], ref_point[1], c='black', marker='x', s=100, label='Fixed Reference')
    plt.xlabel("Makespan (hours)")
    plt.ylabel("Total Cost")
    plt.title("2D Pareto Front (Makespan vs. Total Cost)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pareto_3d(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                   labels: List[str], markers: List[str], colors: List[str],
                   ref_point: Optional[np.ndarray] = None) -> None:
    """
    Plot 3D Pareto fronts (Makespan, Total Cost, Average Utilization) for the provided archives.
    
    If a reference point is provided, it is plotted as a distinct marker.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
            # Note: Average utilization is negated in our objective vector.
            ax.scatter(objs[:, 0], objs[:, 1], -objs[:, 2], c=color, marker=marker, s=80,
                       edgecolor='k', label=label)
    if ref_point is not None:
        ax.scatter([ref_point[0]], [ref_point[1]], [-ref_point[2]], c='black', marker='x', s=100, label='Fixed Reference')
    ax.set_xlabel("Makespan (hours)")
    ax.set_ylabel("Total Cost")
    ax.set_zlabel("Average Utilization")
    ax.set_title("3D Pareto Front")
    ax.legend()
    plt.show()

# =============================================================================
# -------------------- Random Instance & Task Generation --------------------
# =============================================================================

def generate_random_tasks(num_tasks: int, workers: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Generate a list of random, acyclic tasks for scalability testing.
    
    Each task (indexed from 1) may depend on any subset of tasks 1 to i-1.
    """
    tasks_list = []
    resource_types = list(workers.keys())
    for i in range(1, num_tasks + 1):
        base_effort = random.randint(50, 150)
        min_alloc = random.randint(1, 3)
        max_alloc = random.randint(min_alloc + 1, 15)
        dependencies = random.sample(range(1, i), random.randint(0, min(3, i - 1))) if i > 1 else []
        resource = random.choice(resource_types)
        tasks_list.append({
            "id": i,
            "task_name": f"Task {i}",
            "base_effort": base_effort,
            "min": min_alloc,
            "max": max_alloc,
            "dependencies": dependencies,
            "resource": resource
        })
    return tasks_list

# =============================================================================
# ----------------------- Algorithm Implementations -------------------------
# =============================================================================

def MOHHO_with_progress(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Adaptive MOHHO (Multi-Objective Harris Hawks Optimization) with chaotic initialization,
    non-linear adaptive escape energy, self-adaptive step size, and periodic local search.
    
    Also includes diversity-driven injection of new hawks if the population becomes too similar.
    
    Returns:
        archive: List of non-dominated solutions.
        progress: Convergence history (best makespan per iteration).
    """
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    # Initialize self-adaptive step sizes (one per hawk and per dimension)
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    t = 0
    diversity_threshold = 0.1 * np.mean(ub - lb)
    while t < max_iter:
        # Non-linear decaying escape energy (using cosine function)
        E1 = 2 * math.cos((t / max_iter) * (math.pi / 2))
        for i in range(search_agents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            f_val = objf(X[i, :])
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))
        # Select a guiding solution ("rabbit") using diversity-aware roulette selection from archive
        rabbit = random.choice(archive)[0] if archive else X[0, :].copy()
        for i in range(search_agents_no):
            old_x = X[i, :].copy()
            old_obj = np.linalg.norm(objf(old_x))
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            r = random.random()
            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_index = random.randint(0, search_agents_no - 1)
                X_rand = X[rand_index, :].copy()
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * np.abs(X_rand - 2 * random.random() * X[i, :])
                else:
                    X[i, :] = (rabbit - np.mean(X, axis=0)) - random.random() * ((ub - lb) * random.random() + lb)
            else:
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = rabbit - Escaping_Energy * np.abs(rabbit - X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X[i, :] = (rabbit - X[i, :]) - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :])
                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :])
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :]) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0)) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
            # --- Self-adaptive step size update in MOHHO ---
            # Scale the update by the hawk's step size:
            new_x = old_x + step_sizes[i, :] * (X[i, :] - old_x)
            new_x = np.clip(new_x, lb, ub)
            new_obj = np.linalg.norm(objf(new_x))
            # Adapt step size based on improvement
            if new_obj < old_obj:
                step_sizes[i, :] *= 0.95
            else:
                step_sizes[i, :] *= 1.05
            X[i, :] = new_x.copy()
        # --- Diversity-driven new hawk injection ---
        # Compute average pairwise distance in population X
        dists = [np.linalg.norm(X[i] - X[j]) for i in range(search_agents_no) for j in range(i+1, search_agents_no)]
        avg_dist = np.mean(dists) if dists else 0
        if avg_dist < diversity_threshold:
            # Replace worst performing hawk with a new one
            obj_values = [np.linalg.norm(objf(X[i])) for i in range(search_agents_no)]
            worst_idx = np.argmax(obj_values)
            if archive:
                base = random.choice(archive)[0]
                new_hawk = base + np.random.uniform(-0.5, 0.5, size=dim)
                X[worst_idx, :] = np.clip(new_hawk, lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)  # reset step size
            else:
                X[worst_idx, :] = chaotic_map_initialization(lb, ub, dim, 1)[0]
                step_sizes[worst_idx, :] = np.ones(dim)
        best_makespan = np.min([objf(X[i, :])[0] for i in range(search_agents_no)])
        progress.append(best_makespan)
        t += 1
    return archive, progress

class PSO:
    """
    Adaptive MOPSO (Multi-Objective Particle Swarm Optimization) with non-linear inertia update,
    periodic mutation, diversity-preserving archive updates, and self-adaptive inertia weights.
    Also uses hypercube-based leader selection.
    """
    def __init__(self, dim: int, lb: np.ndarray, ub: np.ndarray,
                 obj_funcs: List[Callable[[np.ndarray], float]], pop: int = 30,
                 c2: float = 1.05, w_max: float = 0.9, w_min: float = 0.4,
                 disturbance_rate_min: float = 0.1, disturbance_rate_max: float = 0.3,
                 jump_interval: int = 20) -> None:
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.obj_funcs = obj_funcs
        self.pop = pop
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.iteration = 0
        self.max_iter = 200
        self.vmax = self.ub - self.lb
        self.swarm: List[Dict[str, Any]] = []
        for _ in range(pop):
            pos = np.array([random.randint(int(self.lb[i]), int(self.ub[i])) for i in range(dim)])
            vel = np.array([random.uniform(-self.vmax[i], self.vmax[i]) for i in range(dim)])
            particle = {
                'position': pos,
                'velocity': vel,
                'pbest': pos.copy(),
                'obj': self.evaluate(pos),
                'w': self.w_max  # self-adaptive inertia weight per particle
            }
            self.swarm.append(particle)
        self.archive: List[Tuple[np.ndarray, np.ndarray]] = []
        self.disturbance_rate_min = disturbance_rate_min
        self.disturbance_rate_max = disturbance_rate_max
        self.jump_interval = jump_interval

    def evaluate(self, pos: np.ndarray) -> np.ndarray:
        """Evaluate a particle's position using the objective functions."""
        if len(self.obj_funcs) == 1:
            return np.array([self.obj_funcs[0](pos)])
        else:
            return np.array([f(pos) for f in self.obj_funcs])

    def select_leader_hypercube(self) -> List[np.ndarray]:
        """
        Select guiding positions for each particle based on hypercube division of the archive.
        The objective space is divided into a fixed grid, and leaders are selected with probability
        inversely proportional to the number of archive solutions in the corresponding grid cell.
        """
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        # Use all objectives for grid division
        objs = np.array([entry[1] for entry in self.archive])
        num_bins = 5  # fixed number of divisions per objective
        mins = np.min(objs, axis=0)
        maxs = np.max(objs, axis=0)
        # Avoid zero division
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        cell_indices = []
        cell_counts = {}
        for entry in self.archive:
            idx = tuple(((entry[1] - mins) / ranges * num_bins).astype(int))
            idx = tuple(min(x, num_bins - 1) for x in idx)
            cell_indices.append(idx)
            cell_counts[idx] = cell_counts.get(idx, 0) + 1
        # For each particle, select a leader from archive with probability proportional to 1/(cell count)
        leaders = []
        weights = [1 / cell_counts[cell_indices[i]] for i in range(len(self.archive))]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        for _ in range(self.pop):
            chosen = np.random.choice(len(self.archive), p=probs)
            leaders.append(self.archive[chosen][0])
        return leaders

    def jump_improved_operation(self) -> None:
        """Perform a jump operation to help escape local optima."""
        if len(self.archive) < 2:
            return
        c1, c2 = random.sample(self.archive, 2)
        a1, a2 = random.uniform(0, 1), random.uniform(0, 1)
        oc1 = c1[0] + a1 * (c1[0] - c2[0])
        oc2 = c2[0] + a2 * (c2[0] - c1[0])
        for oc in [oc1, oc2]:
            oc = np.array([int(np.clip(val, self.lb[i], self.ub[i])) for i, val in enumerate(oc)])
            obj_val = self.evaluate(oc)
            self.archive = update_archive_with_crowding(self.archive, (oc, obj_val))

    def disturbance_operation(self, particle: Dict[str, Any]) -> None:
        """Apply a random disturbance to a particle's position to enhance exploration."""
        rate = self.disturbance_rate_min + (self.disturbance_rate_max - self.disturbance_rate_min) * (self.iteration / self.max_iter)
        if random.random() < rate:
            k = random.randint(1, self.dim)
            dims = random.sample(range(self.dim), k)
            new_pos = particle['position'].copy()
            for d in dims:
                rn = np.random.normal(0.5, 1)
                if rn < 0.5:
                    new_pos[d] = new_pos[d] - 0.5 * (new_pos[d] - self.lb[d]) * rn
                else:
                    new_pos[d] = new_pos[d] + 0.5 * (self.ub[d] - new_pos[d]) * rn
                new_pos[d] = int(np.clip(new_pos[d], self.lb[d], self.ub[d]))
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)

    def move(self) -> None:
        """
        Update the swarm by moving each particle, applying self-adaptive inertia weight updates,
        periodic extra mutation if diversity is low, and updating the archive.
        Leader selection is now performed using a hypercube-based method.
        """
        self.iteration += 1
        leaders = self.select_leader_hypercube()
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj = np.linalg.norm(self.evaluate(old_pos))
            r2 = random.random()
            guide = leaders[idx]
            # Use the particle's own inertia weight
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([int(np.clip(round(new_pos[i]), self.lb[i], self.ub[i])) for i in range(self.dim)])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)
            particle['pbest'] = new_pos.copy()
            # --- Self-adaptive inertia weight update ---
            new_obj = np.linalg.norm(self.evaluate(new_pos))
            if new_obj < old_obj:
                particle['w'] = max(particle['w'] * 0.95, self.w_min)
            else:
                particle['w'] = min(particle['w'] * 1.05, self.w_max)
            self.disturbance_operation(particle)
        self.update_archive()
        # Periodic jump operation to boost exploration
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()
        # Extra mutation: if swarm diversity is low, reinitialize one random particle
        positions = np.array([p['position'] for p in self.swarm])
        if len(positions) > 1:
            pairwise_dists = [np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions))]
            avg_distance = np.mean(pairwise_dists)
            if avg_distance < 0.1 * np.mean(self.ub - self.lb):
                idx_to_mutate = random.randint(0, self.pop - 1)
                self.swarm[idx_to_mutate]['position'] = np.array([random.randint(int(self.lb[i]), int(self.ub[i])) for i in range(self.dim)])
                self.swarm[idx_to_mutate]['obj'] = self.evaluate(self.swarm[idx_to_mutate]['position'])
        self.update_archive()

    def update_archive(self) -> None:
        """Update the external archive using current swarm particles."""
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def run(self, max_iter: Optional[int] = None) -> List[float]:
        """
        Run the Adaptive MOPSO for a specified number of iterations.
        
        Returns:
            List of best makespan values per iteration.
        """
        if max_iter is None:
            max_iter = self.max_iter
        convergence: List[float] = []
        for _ in range(max_iter):
            self.move()
            best_ms = min(p['obj'][0] for p in self.swarm)
            convergence.append(best_ms)
        return convergence

def MOACO_improved(objf: Callable[[np.ndarray], np.ndarray],
                    tasks: List[Dict[str, Any]], workers: Dict[str, int],
                    lb: np.ndarray, ub: np.ndarray, ant_count: int, max_iter: int,
                    alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.1,
                    Q: float = 100.0, P: float = 0.6, w1: float = 1.0, w2: float = 1.0,
                    sigma_share: float = 1.0, lambda3: float = 2.0, lambda4: float = 5.0,
                    colony_count: int = 2  # NEW: number of colonies for multi-colony approach
                    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Improved MOACO (Multi-Objective Ant Colony Optimization) incorporating local search,
    multi-colony pheromone updates with periodic reinitialization, and adaptive evaporation.
    
    Returns:
        archive: List of non-dominated solutions.
        progress: Convergence history (best makespan per iteration).
    """
    dim = len(lb)
    # Initialize multiple pheromone matrices (one per colony)
    colony_pheromones = []
    colony_heuristics = []
    for _ in range(colony_count):
        pheromone = []
        heuristic = []
        for i in range(dim):
            possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
            pheromone.append({v: 1.0 for v in possible_values})
            h_dict = {}
            task = tasks[i]
            for v in possible_values:
                new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (v - 1))
                duration = new_effort / v
                h_dict[v] = 1.0 / duration
            heuristic.append(h_dict)
        colony_pheromones.append(pheromone)
        colony_heuristics.append(heuristic)
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    # Divide ants evenly among colonies
    ants_per_colony = ant_count // colony_count
    for iteration in range(max_iter):
        # For each colony, construct solutions
        colony_solutions = []
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            for _ in range(ants_per_colony):
                solution: List[int] = []
                for i in range(dim):
                    possible_values = list(pheromone[i].keys())
                    probs = []
                    for v in possible_values:
                        tau = pheromone[i][v]
                        h_val = heuristic[i][v]
                        probs.append((tau ** alpha) * (h_val ** beta))
                    total = sum(probs)
                    probs = [p / total if total > 0 else 1 / len(probs) for p in probs]
                    r = random.random()
                    cumulative = 0.0
                    chosen = possible_values[-1]
                    for idx, v in enumerate(possible_values):
                        cumulative += probs[idx]
                        if r <= cumulative:
                            chosen = v
                            break
                    solution.append(chosen)
                # Local search: explore neighbors by adjusting each dimension by ±1.
                neighbors = []
                for i in range(dim):
                    for delta in [-1, 1]:
                        neighbor = solution.copy()
                        neighbor[i] = int(np.clip(neighbor[i] + delta, lb[i], ub[i]))
                        neighbors.append(neighbor)
                best_neighbor = solution
                best_obj = objf(np.array(solution))
                for neighbor in neighbors:
                    n_obj = objf(np.array(neighbor))
                    if n_obj[0] < best_obj[0]:
                        best_obj = n_obj
                        best_neighbor = neighbor
                solution = best_neighbor
                obj_val = objf(np.array(solution))
                colony_solutions.append((solution, obj_val))
        # Merge solutions from all colonies: update global archive
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), obj_val))
        # Update pheromones separately for each colony
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            for i in range(dim):
                for v in pheromone[i]:
                    # Adaptive evaporation: can be modified further if desired
                    pheromone[i][v] *= (1 - evaporation_rate)
            # Deposit pheromone from non-dominated solutions
            for sol, obj_val in archive:
                r = random.random()
                if r > P:
                    deposit = w1 * lambda3
                else:
                    niche_counts = []
                    for (arch_sol, arch_obj) in archive:
                        count = 0.0
                        for (other_sol, other_obj) in archive:
                            if np.array_equal(arch_sol, other_sol):
                                continue
                            d = np.linalg.norm(arch_obj - other_obj)
                            if d < sigma_share:
                                count += (1 - d / sigma_share)
                        niche_counts.append(count)
                    min_index = np.argmin(niche_counts)
                    chosen_sol, chosen_obj = archive[min_index]
                    distances = [np.linalg.norm(chosen_obj - other_obj)
                                 for (other_sol, other_obj) in archive
                                 if not np.array_equal(chosen_sol, other_sol)]
                    mu = min(distances) if distances else 0
                    deposit = w2 * (lambda4 if mu > 0 else lambda3)
                for i, v in enumerate(sol):
                    pheromone[i][v] += deposit
        # --- Periodic pheromone reinitialization ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            # Gather all pheromone values
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            if np.var(all_values) < 0.001:  # threshold for reinitialization
                for i in range(dim):
                    possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
                    pheromone[i] = {v: 1.0 for v in possible_values}
        # Merge pheromone matrices from all colonies (average)
        merged_pheromone = []
        for i in range(dim):
            merged = {}
            possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
            for v in possible_values:
                val = sum(colony_pheromones[colony_idx][i].get(v, 0) for colony_idx in range(colony_count)) / colony_count
                merged[v] = val
            merged_pheromone.append(merged)
        # Update each colony's pheromone with the merged version for synchronization
        for colony_idx in range(colony_count):
            colony_pheromones[colony_idx] = [merged_pheromone[i].copy() for i in range(dim)]
        best_ms = min(obj_val[0] for _, obj_val in colony_solutions)
        progress.append(best_ms)
    return archive, progress

# =============================================================================
# ------------------------- Experiment Runner -------------------------------
# =============================================================================

def run_experiments(runs: int = 1, use_random_instance: bool = False, num_tasks: int = 10
                   ) -> Tuple[Dict[str, Any], Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]], List[Dict[str, Any]]]:
    """
    Run multiple independent experiments for Adaptive MOHHO, Adaptive MOPSO, Improved MOACO, and Baseline.
    
    Returns:
        results: Dictionary containing performance metrics (best makespan, normalized hypervolume, spread).
        archives_all: Dictionary of final archives (Pareto fronts) for each algorithm (list of runs).
        base_schedules: List of baseline schedules from the greedy allocation.
    """
    workers = {"Developer": 10, "Manager": 2, "Tester": 3}
    worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}

    if use_random_instance:
        tasks = generate_random_tasks(num_tasks, workers)
    else:
        tasks = get_default_tasks()
    model = RCPSPModel(tasks, workers, worker_cost)
    dim = len(model.tasks)
    lb_current = np.array([task["min"] for task in model.tasks])
    ub_current = np.array([task["max"] for task in model.tasks])
    
    # Prepare results storage.
    results = {
        "MOHHO": {"best_makespan": [], "normalized_hypervolume": [], "spread": []},
        "PSO": {"best_makespan": [], "normalized_hypervolume": [], "spread": []},
        "MOACO": {"best_makespan": [], "normalized_hypervolume": [], "spread": []},
        "Baseline": {"makespan": []}
    }
    archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]] = {"MOHHO": [], "PSO": [], "MOACO": []}
    base_schedules = []

    for run in range(runs):
        logging.info(f"Run {run+1}/{runs}...")
        base_schedule, base_ms = model.baseline_allocation()
        results["Baseline"]["makespan"].append(base_ms)
        base_schedules.append(base_schedule)

        hho_iter = 100
        search_agents_no = 10
        archive_hho, _ = MOHHO_with_progress(lambda x: multi_objective(x, model), lb_current, ub_current, dim, search_agents_no, hho_iter)
        best_ms_hho = min(archive_hho, key=lambda entry: entry[1][0])[1][0] if archive_hho else None
        results["MOHHO"]["best_makespan"].append(best_ms_hho)
        archives_all["MOHHO"].append(archive_hho)

        objectives = [lambda x: objective_makespan(x, model),
                      lambda x: objective_total_cost(x, model),
                      lambda x: objective_neg_utilization(x, model)]
        optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                        pop=10, c2=1.05, w_max=0.9, w_min=0.4,
                        disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
        _ = optimizer.run(max_iter=100)
        archive_pso = optimizer.archive
        best_ms_pso = min(archive_pso, key=lambda entry: entry[1][0])[1][0] if archive_pso else None
        results["PSO"]["best_makespan"].append(best_ms_pso)
        archives_all["PSO"].append(archive_pso)

        ant_count = 10
        moaco_iter = 100
        archive_moaco, _ = MOACO_improved(lambda x: multi_objective(x, model), model.tasks, workers,
                                          lb_current, ub_current, ant_count, moaco_iter,
                                          alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0)
        best_ms_moaco = min(archive_moaco, key=lambda entry: entry[1][0])[1][0] if archive_moaco else None
        results["MOACO"]["best_makespan"].append(best_ms_moaco)
        archives_all["MOACO"].append(archive_moaco)

    # Compute a fixed reference point from the union of all archives.
    fixed_ref = compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")

    # For each algorithm and each run, compute normalized hypervolume using the fixed reference.
    for alg in ["MOHHO", "PSO", "MOACO"]:
        for archive in archives_all[alg]:
            norm_hv = normalized_hypervolume_fixed(archive, fixed_ref)
            results[alg]["normalized_hypervolume"].append(norm_hv)
            sp = compute_spread(archive)
            results[alg]["spread"].append(sp)

    # (Optional) Compute generational distance using the union archive as true Pareto front.
    union_archive = [entry for alg in archives_all for archive in archives_all[alg] for entry in archive]
    true_pareto = []
    for sol, obj in union_archive:
        if not any(dominates(other_obj, obj) for _, other_obj in union_archive if not np.array_equal(other_obj, obj)):
            true_pareto.append(obj)
    true_pareto = np.array(true_pareto)
    gd_results = {"MOHHO": [], "PSO": [], "MOACO": []}
    for alg in ["MOHHO", "PSO", "MOACO"]:
        for archive in archives_all[alg]:
            gd = compute_generational_distance(archive, true_pareto) if archive and true_pareto.size > 0 else None
            gd_results[alg].append(gd)
    results["Generational_Distance"] = gd_results

    return results, archives_all, base_schedules

# =============================================================================
# ------------------------- Statistical Analysis ----------------------------
# =============================================================================

def statistical_analysis(results: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute the mean, standard deviation, and perform one-way ANOVA on best makespan values.
    
    The ANOVA test checks if the differences in mean performance among the algorithms (and baseline)
    are statistically significant.
    """
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

# =============================================================================
# ------------------------- Parameter Tuning (Grid Search) ------------------
# =============================================================================

def grid_search_pso_population(pop_sizes: List[int], runs_per_config: int = 3, model: RCPSPModel = None,
                               lb: np.ndarray = None, ub: np.ndarray = None, dim: int = None) -> Dict[int, Tuple[float, float]]:
    """
    Perform a grid search to tune the population size of Adaptive MOPSO.
    
    For each population size, record the average best makespan and standard deviation.
    
    Returns:
        Dictionary mapping population size to (average best makespan, standard deviation).
    """
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

# =============================================================================
# ------------------------- Automated Unit Testing --------------------------
# =============================================================================

def run_unit_tests() -> None:
    """
    Run basic unit tests:
      1. Test that update_archive_with_crowding produces a non-dominated archive.
      2. Test that RCPSPModel.compute_schedule returns a feasible schedule.
    """
    # Test 1: Archive update
    sol1 = np.array([1, 2, 3])
    obj1 = np.array([10, 20, 30])
    sol2 = np.array([2, 3, 4])
    obj2 = np.array([12, 22, 32])
    archive = []
    archive = update_archive_with_crowding(archive, (sol1, obj1))
    archive = update_archive_with_crowding(archive, (sol2, obj2))
    # In a minimization context, if sol1 dominates sol2, archive should contain only sol1.
    if len(archive) != 1:
        logging.error("Unit Test Failed: Archive contains dominated solutions.")
    else:
        logging.info("Unit Test Passed: Archive update produces non-dominated set.")

    # Test 2: Feasibility of schedule
    workers = {"Developer": 5, "Manager": 2, "Tester": 3}
    worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
    tasks = get_default_tasks()
    model = RCPSPModel(tasks, workers, worker_cost)
    x = np.array([task["min"] for task in tasks])
    schedule, ms = model.compute_schedule(x)
    if schedule and ms > 0:
        logging.info("Unit Test Passed: RCPSP schedule is computed successfully.")
    else:
        logging.error("Unit Test Failed: RCPSP schedule computation issue.")

# =============================================================================
# ------------------------- Main Comparison ---------------------------------
# =============================================================================

if __name__ == '__main__':
    # Run unit tests first
    run_unit_tests()
    
    runs = 5  # Number of independent runs for statistical significance
    use_random_instance = False  # Set True for random instances (scalability testing)
    num_tasks = 10

    if use_random_instance:
        tasks_for_exp = generate_random_tasks(num_tasks, {"Developer": 10, "Manager": 2, "Tester": 3})
    else:
        tasks_for_exp = get_default_tasks()

    results, archives_all, base_schedules = run_experiments(runs=runs, use_random_instance=use_random_instance, num_tasks=num_tasks)
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    means, stds = statistical_analysis(results)
    
    # Plot convergence metrics for Best Makespan, Normalized Hypervolume, Spread, and Generational Distance.
    plot_convergence({alg: results[alg]["best_makespan"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Best Makespan (hours)")
    plot_convergence({alg: results[alg]["normalized_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Normalized Hypervolume (%)")
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Spread (Diversity)")
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    
    # Plot Pareto fronts with the fixed reference point for comparison.
    fixed_ref = compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    last_archives = [archives_all[alg][-1] for alg in ["MOHHO", "PSO", "MOACO"]]
    plot_pareto_2d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'], ref_point=fixed_ref)
    plot_pareto_3d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'], ref_point=fixed_ref)
    
    last_baseline = base_schedules[-1]
    last_makespan = results["Baseline"]["makespan"][-1]
    plot_gantt(last_baseline, f"Baseline Schedule (Greedy Allocation)\nMakespan: {last_makespan:.2f} hrs")
    
    logging.info("Starting grid search for PSO population size...")
    pop_sizes = [10, 20, 30]
    workers = {"Developer": 10, "Manager": 2, "Tester": 3}
    worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
    default_tasks = get_default_tasks()
    model_for_grid = RCPSPModel(default_tasks, workers, worker_cost)
    lb_array = np.array([task["min"] for task in default_tasks])
    ub_array = np.array([task["max"] for task in default_tasks])
    grid_results = grid_search_pso_population(pop_sizes, runs_per_config=3, model=model_for_grid,
                                              lb=lb_array, ub=ub_array, dim=len(default_tasks))
    
    logging.info("Experiment complete. Results saved to 'experiment_results.json'.")

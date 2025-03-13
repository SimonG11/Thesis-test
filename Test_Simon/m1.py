#!/usr/bin/env python3
"""
Improved Multi-Objective Comparison for RCPSP using Adaptive MOHHO, Adaptive MOPSO, and Improved MOACO

This script implements and compares three metaheuristic algorithms for the Resource-Constrained 
Project Scheduling Problem (RCPSP) with multiple objectives. All search moves (worker allocations)
are strictly limited to half-step increments (1, 1.5, 2, 2.5, ...), reducing the search space and
speeding up the algorithms.

References:
 - Heidari, A., et al. "Harris Hawks Optimization: Algorithm and Applications." [Original HHO]
 - Coello, C.A.C., & Lechuga, M.S. "MOPSO: A Proposal for Multiple Objective Particle Swarm Optimization." [MOPSO base]
 - Dorigo, M., & Stützle, T. "Ant Colony Optimization." [ACO base]
 - Deb, K. "Multi-Objective Optimization Using Evolutionary Algorithms." [NSGA-II crowding]
 - Sun, Y., et al. "Chaotic Multi-Objective Particle Swarm Optimization Algorithm Incorporating Clone Immunity." [Chaotic init in MOPSO]
 - Yüzgeç, U., & Kuşoğlu, M. "An Improved Multi-Objective Harris Hawk Optimization with Blank Angle Region Enhanced Search." [MOHHO enhancements]

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

# ------------------------- Helper Functions -------------------------

def round_half(x: float) -> float:
    """
    Round a number to the nearest half step (e.g., 1, 1.5, 2, 2.5, ...).
    This function ensures that worker allocations remain in the allowed discrete set.
    """
    return round(x * 2) / 2.0

def clip_round_half(x: float, lb: float, ub: float) -> float:
    """Clip a value between lb and ub and round it to the nearest half step."""
    return round_half(np.clip(x, lb, ub))

def discretize_vector(vec: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Discretize each element of a vector to half steps within given bounds."""
    return np.array([clip_round_half(val, lb[i], ub[i]) for i, val in enumerate(vec)])

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

def find_earliest_start(earliest: float, duration: float, allocated: float,
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


# Named constants for clarity.
DAY_HOURS_FULL = 8    # Full working day in billable hours.
DAY_HOURS_HALF = DAY_HOURS_FULL / 2    # Half working day in billable hours.

def convert_hours_to_billable_days(duration: float) -> float:
    """
    Convert a duration (in hours) to billable days.
    
    Business Rules:
      - Non-positive durations return 0.0 days.
      - Durations ≤ DAY_HOURS_HALF (4 hours) count as a half day (0.5).
      - Durations > DAY_HOURS_HALF and ≤ DAY_HOURS_FULL (8 hours) count as a full day (1.0).
      - For durations > DAY_HOURS_FULL:
            Compute full days = duration // DAY_HOURS_FULL.
            Let remainder = duration - (full_days * DAY_HOURS_FULL).
            If remainder is ≤ DAY_HOURS_HALF, add 0.5 day; otherwise, add 1.0 day.
    
    Examples:
      • 1.25 or 4 hours   → 0.5 day
      • 4.1, 7.9, or 8 hours → 1.0 day
      • 10 hours           → 1 full day + 0.5 day = 1.5 days
      • 13 hours           → 1 full day + 1.0 day = 2.0 days
      • 16 hours           → 2 full days = 2.0 days
    """
    if duration <= 0:
        return 0.0
    full_days = int(duration // DAY_HOURS_FULL)
    remainder = duration - full_days * DAY_HOURS_FULL
    if remainder == 0:
        extra = 0.0
    elif remainder <= DAY_HOURS_HALF:
        extra = 0.5
    else:
        extra = 1.0
    return full_days + extra

def compute_billable_hours(duration: float) -> float:
    """
    Convert a duration (in hours) to billable hours.
    
    Billing Conversion:
      - Each full day is billed as DAY_HOURS_FULL hours.
      - Each half day is billed as DAY_HOURS_HALF hours.
    
    Examples:
      • 3 hours → 0.5 day → 4 billable hours.
      • 4.1 hours → 1.0 day → 8 billable hours.
      • 10 hours → 1.5 days → 12 billable hours.
      • 13 hours → 2.0 days → 16 billable hours.
    """
    billable_days = convert_hours_to_billable_days(duration)
    full_days = int(billable_days)
    half_day = 1 if (billable_days - full_days) >= 0.5 else 0
    return full_days * DAY_HOURS_FULL + half_day * DAY_HOURS_HALF

def compute_billable_cost(duration: float, allocation: float, wage_rate: float) -> float:
    """
    Compute the total labor cost given a task's duration (in hours), worker allocation, and wage rate.
    
    Let F be the computed billable hours.
    For a full worker (allocation = 1): cost = F * wage_rate.
    
    For a pure half worker (allocation < 1):
      - If duration < 2 hours, charge half of the full worker cost.
      - Otherwise, if F ≤ DAY_HOURS_FULL, charge the full worker cost;
        if F > DAY_HOURS_FULL, subtract a discount computed as (DAY_HOURS_HALF * wage_rate) / 2.
    
    For a mixed allocation (allocation ≥ 1), interpret allocation as:
      (integer number of full workers) plus a half worker if the fractional part is ≥ 0.5.
      Then, cost = (number of full workers * full_cost) + (if half worker present, add full_cost/2).
    
    Examples (with wage_rate = 50):
      - Full worker (allocation = 1):
          • 3 hours: F = 4 → cost = 4 * 50 = 200.
          • 10 hours: F = 12 → cost = 12 * 50 = 600.
      - Pure half worker (allocation = 0.5):
          • 1 hour: F = 4, duration < 2 → cost = (4 * 50) / 2 = 100.
          • 3 hours: F = 4, duration ≥ 2 → cost = 4 * 50 = 200.
          • 8 hours: F = 8 → cost = 8 * 50 = 400.
          • 10 hours: F = 12 → cost = (12 * 50) - ((DAY_HOURS_HALF * 50) / 2)
                      = 600 - ( (4*50)/2 ) = 600 - 100 = 500.
          • 18 hours: F = 20 → cost = (20 * 50) - ((DAY_HOURS_HALF * 50) / 2)
                      = 1000 - 100 = 900.
      - Mixed allocation (allocation = 1.5):
          • 3 hours: cost = (1 * 200) + (200/2) = 200 + 100 = 300.
          • 10 hours: cost = (1 * 600) + (600/2) = 600 + 300 = 900.
          • For allocation = 2.5 and 13 hours (F = 16, wage_rate = 40):
              cost = (2 * 16 * 40) + (16 * 40)/2 = 1280 + 320 = 1600.
    """
    F = compute_billable_hours(duration)
    full_cost = F * wage_rate
    if allocation < 1:
        # Pure half worker.
        if duration < 2:
            return full_cost / 2
        else:
            if F <= DAY_HOURS_FULL:
                return full_cost
            else:
                # Discount computed as half the cost of a half-day.
                discount = (DAY_HOURS_HALF * wage_rate) / 2
                return full_cost - discount
    else:
        full_workers = int(allocation)
        fractional = allocation - full_workers
        if fractional >= 0.5:
            return full_workers * full_cost + full_cost / 2
        else:
            return full_workers * full_cost


# ---------------------------------------------------------------------------
# Chaotic Initialization using Logistic Map
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Default Task Definition for Reproducibility
# ---------------------------------------------------------------------------
def get_default_tasks() -> List[Dict[str, Any]]:
    """
    Return a fixed list of tasks for the RCPSP.
    
    This fixed instance supports reproducible experiments and benchmark comparisons.
    """
    return [
        {"id": 1, "task_name": "Requirements Gathering", "base_effort": 80,  "min": 0.5, "max": 14, "dependencies": [],         "resource": "Manager"},
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
            # Use half-step rounding for worker allocation.
            alloc = round_half(x[tid - 1])
            alloc = max(task["min"], min(effective_max, alloc))
            if alloc == 0.5:
                new_effort = task["base_effort"] * 1.2 #Increse effor by 20% if you are only allowed to work 50% with the condition that you work alone
                duration = new_effort
            else:
                new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
                duration = new_effort / alloc
            duration = convert_hours_to_billable_days(duration)
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
# ----------------------- Objective Functions -------------------------------
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
        # Use the round_half function for consistent half-step allocation.
        alloc = round_half(x[tid - 1])
        alloc = max(task["min"], min(effective_max, alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
        wage_rate = model.worker_cost[resource_type]
        task_cost = compute_billable_cost(duration, alloc, wage_rate)
        total_cost += task_cost

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
        alloc = round_half(x[tid - 1])  # Use round_half for half-step allocation.
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

def compute_combined_ideal(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
    """
    Compute the combined ideal point from multiple archives.
    """
    union_archive = []
    for alg in archives_all:
        for archive in archives_all[alg]:
            union_archive.extend(archive)
    if not union_archive:
        raise ValueError("No archive entries found.")
    objs = np.array([entry[1] for entry in union_archive])
    ideal = np.min(objs, axis=0)
    return ideal
# ---------------------------------------------------------------------------
# Fixed Reference Point Calculation for Hypervolume Comparison
# ---------------------------------------------------------------------------
def compute_fixed_reference(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
    """
    Compute a fixed reference point based on the union of all solution archives from all algorithms.
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

def absolute_hypervolume_fixed(archive: List[Tuple[np.ndarray, np.ndarray]],
                               reference_point: np.ndarray,
                               global_lower_bound: np.ndarray,
                               num_samples: int = 1000) -> float:
    """
    Approximate the absolute hypervolume using a fixed global lower bound.
    """
    if not archive:
        return 0.0
    lb = global_lower_bound
    ub = reference_point
    samples = np.random.uniform(low=lb, high=ub, size=(num_samples, len(ub)))
    objs = np.array([entry[1] for entry in archive])
    dominated_count = sum(1 for sample in samples if any(np.all(sol <= sample) for sol in objs))
    return (dominated_count / num_samples) * 100

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
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
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
# --------------------------- MOHHO Algorithm -------------------------
def MOHHO_with_progress(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Adaptive MOHHO_with_progress implements a Multi-Objective Harris Hawks Optimization
    for the RCPSP problem with several enhancements to improve convergence and diversity.
    Decisions are strictly explored in half-step increments.

    Enhancements and Scientific Justifications:
      1. Chaotic Initialization:
         - Uses a logistic chaotic map to initialize the population, thereby enhancing the spread
           and diversity of the initial solutions.
         - Citation: Sun et al. (2019), "Chaotic Multi-Objective Particle Swarm Optimization Algorithm Incorporating Clone Immunity"
           URL: https://doi.org/10.3390/math7020146
         - Also inspired by Yan et al. (2022), "An Improved Multi-Objective Harris Hawk Optimization with Blank Angle Region Enhanced Search"
           URL: https://doi.org/10.3390/sym14050967

      2. Adaptive Step Size Update (Self-adaptation):
         - Dynamically adjusts the step sizes based on improvements between iterations to balance exploration and exploitation.
         - Citation: Adaptive tuning in metaheuristics (e.g., Brest et al. (2006))
           URL: https://doi.org/10.1109/TEVC.2006.872133

      3. Diversity-driven Injection:
         - Monitors population diversity and, if stagnation is detected (average pairwise distance falls below a threshold),
           replaces the worst-performing solution with a new one to avoid premature convergence.
         - Citation: Yüzgeç & Kuşoğlu (2020) propose diversity-driven strategies in multi-objective optimization.

      4. Archive Management via Crowding Distance:
         - Uses a NSGA-II inspired archive update procedure that leverages crowding distance to maintain a diverse set of non-dominated solutions.
         - Citation: Deb et al. (2002), "Multi-Objective Optimization Using Evolutionary Algorithms"
           URL: https://doi.org/10.1109/4235.996017

    Returns:
        archive: A list of non-dominated solutions (each as a tuple of decision and objective vectors).
        progress: A list recording the best makespan value (first objective) per iteration.
    """
     # --- Helper functions for normalization ---
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Min–max scales each column of 'mat' to the [0,1] interval.
        Returns the normalized matrix, along with the per-dimension minima and maxima.
        """
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            if range_val != 0:
                norm[:, d] = (mat[:, d] - mins[d]) / range_val
            else:
                norm[:, d] = 0.5
        return norm, mins, maxs

    def normalize_obj(obj, mins, maxs):
        """
        Normalize a single objective vector using provided minima and maxima.
        """
        obj = np.array(obj, dtype=float)
        norm_obj = np.zeros_like(obj)
        for i in range(len(obj)):
            range_val = maxs[i] - mins[i]
            if range_val != 0:
                norm_obj[i] = (obj[i] - mins[i]) / range_val
            else:
                norm_obj[i] = 0.5
        return norm_obj

    # --- Initialization ---
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    t = 0
    diversity_threshold = 0.1 * np.mean(ub - lb)
    
    while t < max_iter:
        # Update archive with current (feasible) solutions using original objectives.
        for i in range(search_agents_no):
            X[i, :] = discretize_vector(np.clip(X[i, :], lb, ub), lb, ub)
            f_val = objf(X[i, :])
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))
        
        # Compute normalization parameters based on the current population's objectives.
        pop_objs = [objf(X[i, :]) for i in range(search_agents_no)]
        pop_objs_mat = np.array(pop_objs)
        _, pop_mins, pop_maxs = normalize_matrix(pop_objs_mat)
        
        # Choose a "rabbit" (best solution) from the archive.
        rabbit = random.choice(archive)[0] if archive else X[0, :].copy()
        
        # --- Update each hawk agent ---
        for i in range(search_agents_no):
            old_x = X[i, :].copy()
            # Compute normalized objective norm for comparison.
            old_obj = np.linalg.norm(normalize_obj(objf(old_x), pop_mins, pop_maxs))
            E0 = 2 * random.random() - 1
            E1 = 2 * math.cos((t / max_iter) * (math.pi / 2))
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
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :]) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0)) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()
            
            new_x = old_x + step_sizes[i, :] * (X[i, :] - old_x)
            new_x = discretize_vector(np.clip(new_x, lb, ub), lb, ub)
            new_obj = np.linalg.norm(normalize_obj(objf(new_x), pop_mins, pop_maxs))
            if new_obj < old_obj:
                step_sizes[i, :] *= 0.95
            else:
                step_sizes[i, :] *= 1.05
            X[i, :] = new_x.copy()
        
        # --- Diversity injection ---
        dists = [np.linalg.norm(X[i] - X[j]) for i in range(search_agents_no) for j in range(i+1, search_agents_no)]
        avg_dist = np.mean(dists) if dists else 0
        if avg_dist < diversity_threshold:
            obj_values = [np.linalg.norm(normalize_obj(objf(X[i]), pop_mins, pop_maxs)) for i in range(search_agents_no)]
            worst_idx = np.argmax(obj_values)
            if archive:
                base = random.choice(archive)[0]
                new_hawk = base + np.random.uniform(-0.5, 0.5, size=dim)
                X[worst_idx, :] = discretize_vector(new_hawk, lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)
            else:
                X[worst_idx, :] = discretize_vector(chaotic_map_initialization(lb, ub, dim, 1)[0], lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)
        
        # --- Compute balanced progress metric ---
        normalized_objs = [normalize_obj(objf(X[i, :]), pop_mins, pop_maxs) for i in range(search_agents_no)]
        ideal = np.min(np.array(normalized_objs), axis=0)
        # Tchebycheff scalarization: for each agent, take the maximum deviation from the ideal.
        tcheby_values = [max(abs(n_obj - ideal)) for n_obj in normalized_objs]
        progress_metric = min(tcheby_values)
        progress.append(progress_metric)
        
        t += 1
    return archive, progress
# --------------------------- MOPSO Algorithm -------------------------

class PSO:
    """
    Adaptive MOPSO (Multi-Objective Particle Swarm Optimization) for RCPSP with normalization.
    
    Enhancements:
      - Normalization of objectives (using dynamic min–max scaling) so that makespan, cost, and utilization are balanced.
      - Tchebycheff scalarization: each particle is evaluated by the maximum deviation (in normalized space) from the ideal point.
      - Leader selection via hypercube division uses normalized objectives.
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
        # Initialize positions using allowed half-step values.
        for _ in range(pop):
            pos = np.array([random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5)))
                            for i in range(dim)])
            vel = np.array([random.uniform(-self.vmax[i], self.vmax[i]) for i in range(dim)])
            particle = {
                'position': pos,
                'velocity': vel,
                'pbest': pos.copy(),
                'obj': self.evaluate(pos),
                'w': self.w_max  # Start with maximum inertia weight.
            }
            self.swarm.append(particle)
        self.archive: List[Tuple[np.ndarray, np.ndarray]] = []
        self.disturbance_rate_min = disturbance_rate_min
        self.disturbance_rate_max = disturbance_rate_max
        self.jump_interval = jump_interval

    def evaluate(self, pos: np.ndarray) -> np.ndarray:
        """Evaluate a particle's position using the provided objective functions."""
        if len(self.obj_funcs) == 1:
            return np.array([self.obj_funcs[0](pos)])
        else:
            return np.array([f(pos) for f in self.obj_funcs])
    
    @staticmethod
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Min–max scales each column of 'mat' to the [0,1] interval.
        Returns the normalized matrix, and the per-dimension minima and maxima.
        """
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            if range_val != 0:
                norm[:, d] = (mat[:, d] - mins[d]) / range_val
            else:
                norm[:, d] = 0.5
        return norm, mins, maxs

    @staticmethod
    def normalize_obj(obj, mins, maxs):
        """
        Normalize a single objective vector using provided minima and maxima.
        """
        obj = np.array(obj, dtype=float)
        norm_obj = np.zeros_like(obj)
        for i in range(len(obj)):
            range_val = maxs[i] - mins[i]
            if range_val != 0:
                norm_obj[i] = (obj[i] - mins[i]) / range_val
            else:
                norm_obj[i] = 0.5
        return norm_obj

    def select_leader_hypercube(self, norm_mins: np.ndarray, norm_maxs: np.ndarray) -> List[np.ndarray]:
        """
        Select leader particles using hypercube division based on normalized objectives.
        """
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        # Normalize archive objective values using current swarm's normalization parameters.
        objs = np.array([entry[1] for entry in self.archive])
        norm_objs = np.array([self.normalize_obj(obj, norm_mins, norm_maxs) for obj in objs])
        num_bins = 5
        mins = np.min(norm_objs, axis=0)
        maxs = np.max(norm_objs, axis=0)
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        cell_indices = []
        cell_counts = {}
        for norm_obj in norm_objs:
            idx = tuple(((norm_obj - mins) / ranges * num_bins).astype(int))
            idx = tuple(min(x, num_bins - 1) for x in idx)
            cell_indices.append(idx)
            cell_counts[idx] = cell_counts.get(idx, 0) + 1
        weights = [1 / cell_counts[cell_indices[i]] for i in range(len(self.archive))]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        leaders = []
        for _ in range(self.pop):
            chosen = np.random.choice(len(self.archive), p=probs)
            leaders.append(self.archive[chosen][0])
        return leaders

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
                new_pos[d] = clip_round_half(new_pos[d], self.lb[d], self.ub[d])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)

    def move(self) -> None:
        """
        Update the swarm by moving each particle.
        
        The new implementation uses normalization and Tchebycheff scalarization to compute a balanced
        performance measure, which then drives the adaptive inertia weight update.
        """
        self.iteration += 1
        
        # --- Compute normalization parameters from current swarm positions ---
        raw_objs = np.array([self.evaluate(p['position']) for p in self.swarm])
        norm_objs, norm_mins, norm_maxs = self.normalize_matrix(raw_objs)
        # Ideal point (component-wise minimum in normalized space)
        ideal = np.min(norm_objs, axis=0)
        
        # --- Select leaders using normalized hypercube division ---
        leaders = self.select_leader_hypercube(norm_mins, norm_maxs)
        
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj_raw = self.evaluate(old_pos)
            old_norm = self.normalize_obj(old_obj_raw, norm_mins, norm_maxs)
            # Tchebycheff scalarization: maximum deviation from the ideal point.
            old_scalar = np.max(np.abs(old_norm - ideal))
            
            r2 = random.random()
            guide = leaders[idx]
            # Standard PSO velocity update.
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([clip_round_half(new_pos[i], self.lb[i], self.ub[i]) for i in range(self.dim)])
            particle['position'] = new_pos
            new_obj_raw = self.evaluate(new_pos)
            new_norm = self.normalize_obj(new_obj_raw, norm_mins, norm_maxs)
            new_scalar = np.max(np.abs(new_norm - ideal))
            particle['obj'] = new_obj_raw
            particle['pbest'] = new_pos.copy()
            
            # --- Self-adaptive inertia weight update based on normalized improvement ---
            if new_scalar < old_scalar:
                particle['w'] = max(particle['w'] * 0.95, self.w_min)
            else:
                particle['w'] = min(particle['w'] * 1.05, self.w_max)
            
            self.disturbance_operation(particle)
        
        self.update_archive()
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()
        
        positions = np.array([p['position'] for p in self.swarm])
        if len(positions) > 1:
            pairwise_dists = [np.linalg.norm(positions[i] - positions[j])
                              for i in range(len(positions)) for j in range(i+1, len(positions))]
            avg_distance = np.mean(pairwise_dists)
            if avg_distance < 0.1 * np.mean(self.ub - self.lb):
                idx_to_mutate = random.randint(0, self.pop - 1)
                self.swarm[idx_to_mutate]['position'] = np.array(
                    [random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5)))
                     for i in range(self.dim)])
                self.swarm[idx_to_mutate]['obj'] = self.evaluate(self.swarm[idx_to_mutate]['position'])
        
        self.update_archive()

    def jump_improved_operation(self) -> None:
        """Perform a jump operation to escape local optima."""
        if len(self.archive) < 2:
            return
        c1, c2 = random.sample(self.archive, 2)
        a1, a2 = random.uniform(0, 1), random.uniform(0, 1)
        oc1 = c1[0] + a1 * (c1[0] - c2[0])
        oc2 = c2[0] + a2 * (c2[0] - c1[0])
        for oc in [oc1, oc2]:
            oc = np.array([clip_round_half(val, self.lb[i], self.ub[i]) for i, val in enumerate(oc)])
            obj_val = self.evaluate(oc)
            self.archive = update_archive_with_crowding(self.archive, (oc, obj_val))

    def update_archive(self) -> None:
        """Update the external archive using the current swarm particles."""
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def run(self, max_iter: Optional[int] = None) -> List[float]:
        """
        Run the Adaptive MOPSO for a specified number of iterations.
        
        Returns:
            convergence: A list of the best makespan values (first objective) recorded per iteration.
        """
        if max_iter is None:
            max_iter = self.max_iter
        convergence: List[float] = []
        for _ in range(max_iter):
            self.move()
            best_ms = min(p['obj'][0] for p in self.swarm)
            convergence.append(best_ms)
        return convergence


# --------------------------- MOACO Algorithm -------------------------
def MOACO_improved(objf: Callable[[np.ndarray], np.ndarray],
                   tasks: List[Dict[str, Any]], 
                   lb: np.ndarray, ub: np.ndarray, 
                   ant_count: int, max_iter: int,
                   alpha: float = 1.0,
                   beta: float = 2.0,
                   evaporation_rate: float = 0.1,
                   w1: float = 1.0,
                   lambda3: float = 2.0,
                   colony_count: int = 10,
                  ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    MOACO_improved implements a multi-objective Ant Colony Optimization for RCPSP with several enhancements.

    Base algorithm concept from Distributed Optimization by Ant Colonies
    https://www.researchgate.net/publication/216300484_Distributed_Optimization_by_Ant_Colonies

    This version employs:
      1. Tchebycheff Scalarization for Heuristic Ranking:
         - For each task, candidate solutions are evaluated using Tchebycheff scalarization,
           ensuring balanced consideration of all objectives (https://doi.org/10.48550/arXiv.2402.19078).
         - This technique is standard in multi-objective optimization (see Deb et al., 2002:
           "Multi-Objective Optimization Using Evolutionary Algorithms" https://doi.org/10.1109/4235.996017).

      2. Pareto-Based Candidate Selection during Local Search:
         - Instead of aggregating normalized objectives (which can introduce bias), candidates are
           ranked using fast non-dominated sorting and crowding distance.
         - This approach follows the NSGA-II methodology (Deb et al., 2002) and related indicator-based methods 
           (e.g., Zitzler & Künzli, 2004).

      3. Adaptive Pheromone Evaporation and Multi-Colony Pheromone Update:
         - The evaporation rate is adjusted based on pheromone variance to prevent premature convergence.
         - Multi-colony updates are applied to encourage diverse search (Angus & Woodward, 2009,
           https://doi.org/10.1007/s11721-008-0022-4) and adaptive evaporation is inspired by Zhao et al. (2018,
           https://doi.org/10.3390/sym10040104).

      4. Archive Management using Crowding Distance:
         - Archive updates use a NSGA-II inspired crowding distance mechanism (Deb et al., 2002) to
           maintain solution diversity.
           
    """
    # ---------------- Helper Functions ----------------
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Min–max scales each column of 'mat' to the [0,1] interval."""
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            norm[:, d] = (mat[:, d] - mins[d]) / range_val if range_val != 0 else 0.5
        return norm, mins, maxs

    def normalized_crowding_distance(archive):
        """
        Compute the crowding distance for each solution in the archive using normalized objectives.
        This function is inspired by the crowding distance mechanism in NSGA-II (Deb et al., 2002).
        """
        if not archive:
            return np.array([])
        objs = np.array([entry[1] for entry in archive], dtype=float)
        norm_objs, _, _ = normalize_matrix(objs)
        num_objs = norm_objs.shape[1]
        distances = np.zeros(len(archive))
        for m in range(num_objs):
            sorted_indices = np.argsort(norm_objs[:, m])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
            m_values = norm_objs[sorted_indices, m]
            m_range = m_values[-1] - m_values[0]
            if m_range == 0:
                continue
            for i in range(1, len(archive) - 1):
                distances[sorted_indices[i]] += (m_values[i+1] - m_values[i-1]) / m_range
        return distances

    def fast_non_dominated_sort(candidates: List[List[float]]) -> List[int]:
        """
        Perform fast non-dominated sorting on candidate objective vectors.
        Returns a list of ranks (lower is better).
        Based on the sorting procedure used in NSGA-II (Deb et al., 2002).
        """
        n = len(candidates)
        S = [[] for _ in range(n)]
        domination_count = [0] * n
        ranks = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (all(candidates[j][k] <= candidates[i][k] for k in range(len(candidates[i]))) and
                    any(candidates[j][k] < candidates[i][k] for k in range(len(candidates[i])))):
                    domination_count[i] += 1
                elif (all(candidates[i][k] <= candidates[j][k] for k in range(len(candidates[i]))) and
                      any(candidates[i][k] < candidates[j][k] for k in range(len(candidates[i])))):
                    S[i].append(j)
            if domination_count[i] == 0:
                ranks[i] = 1
        current_front = [i for i in range(n) if domination_count[i] == 0]
        front_number = 1
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = front_number + 1
                        next_front.append(j)
            front_number += 1
            current_front = next_front
        return ranks

    # --- Helper: Compute heuristic for each task using Tchebycheff scalarization ---
    def compute_task_heuristic(task_index: int) -> Dict[float, float]:
        """
        For the task at index 'task_index', compute heuristic values for each possible allocation
        using Tchebycheff scalarization. This balances the influence of all objectives.
        Reference: Deb et al. (2002).
        """
        possible_values = list(np.arange(lb[task_index], ub[task_index] + 0.5, 0.5))
        candidate_objs = []
        for v in possible_values:
            candidate = np.array([t["min"] for t in tasks])
            candidate[task_index] = v
            candidate_objs.append(objf(candidate))
        candidate_objs = np.array(candidate_objs)
        # Compute the ideal point (componentwise minimum)
        ideal = np.min(candidate_objs, axis=0)
        # Compute Tchebycheff values with equal weights (balanced contribution)
        tcheby_vals = [max(abs(candidate_objs[j] - ideal)) for j in range(len(possible_values))]
        task_heuristic = {v: 1.0 / (tcheby_vals[j] + 1e-6) for j, v in enumerate(possible_values)}
        return task_heuristic

    # --- Initialization of pheromones and heuristics for each colony ---
    dim = len(lb)
    colony_pheromones = []  # One pheromone matrix per colony
    colony_heuristics = []  # One heuristic matrix per colony
    for colony_idx in range(colony_count):
        pheromone_matrix = []
        heuristic_matrix = []
        for i in range(dim):
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            pheromone_matrix.append({v: 1.0 for v in possible_values})
            heuristic_matrix.append(compute_task_heuristic(i))
        colony_pheromones.append(pheromone_matrix)
        colony_heuristics.append(heuristic_matrix)

    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    best_global = float('inf')
    no_improvement_count = 0
    stagnation_threshold = 10  # iterations before triggering reinitialization (diversity injection)
    eps = 1e-6

    # --- Helper: Pareto-based candidate selection ---
    def select_best_candidate(candidates: List[np.ndarray], cand_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the best candidate based on Pareto dominance and crowding distance.
        This selection mechanism uses fast non-dominated sorting and is inspired by the NSGA-II approach (Deb et al., 2002).
        """
        ranks = fast_non_dominated_sort(cand_objs.tolist())
        first_front_indices = [i for i, rank in enumerate(ranks) if rank == 1]
        if len(first_front_indices) == 1:
            best_idx = first_front_indices[0]
        else:
            # Compute crowding distances for candidates in the first front
            front_objs = np.array([cand_objs[i] for i in first_front_indices])
            cd = np.zeros(len(front_objs))
            num_objs = front_objs.shape[1]
            for m in range(num_objs):
                sorted_indices = np.argsort(front_objs[:, m])
                cd[sorted_indices[0]] = cd[sorted_indices[-1]] = float('inf')
                m_range = front_objs[sorted_indices[-1], m] - front_objs[sorted_indices[0], m]
                if m_range == 0:
                    continue
                for j in range(1, len(front_objs) - 1):
                    cd[sorted_indices[j]] += (front_objs[sorted_indices[j+1], m] - front_objs[sorted_indices[j-1], m]) / m_range
            best_in_front = np.argmax(cd)
            best_idx = first_front_indices[best_in_front]
        return candidates[best_idx], cand_objs[best_idx]

    # --- Main Iteration Loop ---
    for iteration in range(max_iter):
        colony_solutions = []  # store solutions from all colonies
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            for _ in range(ants_per_colony):
                solution = []
                # Construct solution: for each task, select allocation based on pheromone and heuristic
                for i in range(dim):
                    possible_values = list(pheromone[i].keys())
                    probs = []
                    for v in possible_values:
                        tau = pheromone[i][v]
                        h_val = heuristic[i][v]
                        probs.append((tau ** alpha) * (h_val ** beta))
                    total = sum(probs)
                    if not np.isfinite(total) or total <= 0:
                        probs = [1.0 / len(probs)] * len(probs)
                    else:
                        probs = [p / total for p in probs]
                    r = random.random()
                    cumulative = 0.0
                    chosen = possible_values[-1]
                    for idx, v in enumerate(possible_values):
                        cumulative += probs[idx]
                        if r <= cumulative:
                            chosen = v
                            break
                    solution.append(chosen)
                # Local search: perturb each task’s allocation by ±0.5
                candidates = [solution]
                for i in range(dim):
                    for delta in [-0.5, 0.5]:
                        neighbor = solution.copy()
                        neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
                        candidates.append(neighbor)
                candidates = [np.array(c) for c in candidates]
                cand_objs = np.array([objf(c) for c in candidates], dtype=float)
                best_candidate, best_obj = select_best_candidate(candidates, cand_objs)
                colony_solutions.append((best_candidate.tolist(), best_obj.tolist()))
        # --- Archive update ---
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), np.array(obj_val)))
        # --- Pheromone Evaporation ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.array(all_values)
            var_pheromone = np.var(all_values)
            # Adaptive evaporation based on variance (prevents premature convergence)
            # Reference: Zhao et al. (2018)
            current_evap_rate = evaporation_rate * 1.5 if var_pheromone < 0.001 else evaporation_rate
            for i in range(dim):
                for v in pheromone[i]:
                    pheromone[i][v] *= (1 - current_evap_rate)
        # --- Pheromone Deposit Update ---
        crowding = normalized_crowding_distance(archive)
        max_cd = np.max(crowding) if len(crowding) > 0 else 1.0
        if not np.isfinite(max_cd) or max_cd <= 0:
            max_cd = 1.0
        decay_factor = 1.0 - (iteration / max_iter)
        for idx, (sol, obj_val) in enumerate(archive):
            deposit = w1 * lambda3 * (crowding[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i, v in enumerate(sol):
                    colony_pheromones[colony_idx][i][v] += deposit
        # --- Multi-Colony Pheromone Reinitialization and Merge ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.array(all_values)
            if np.var(all_values) < 0.001:
                for i in range(dim):
                    possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
                    pheromone[i] = {v: 1.0 for v in possible_values}
        merged_pheromone = []
        for i in range(dim):
            merged = {}
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            for v in possible_values:
                val = sum(colony_pheromones[colony_idx][i].get(v, 0) for colony_idx in range(colony_count)) / colony_count
                merged[v] = val
            merged_pheromone.append(merged)
        for colony_idx in range(colony_count):
            colony_pheromones[colony_idx] = [merged_pheromone[i].copy() for i in range(dim)]
        # --- Record Progress ---
        if archive:
            objs = np.array([entry[1] for entry in archive])
            ideal = np.min(objs, axis=0)
            # Use Tchebycheff scalarization to compute a balanced score
            tcheby_scores = [max(abs(entry[1] - ideal)) for entry in archive]
            current_best = min(tcheby_scores)
        else:
            current_best = float('inf')
        progress.append(current_best)
        # --- Stagnation Handling ---
        if iteration > 0 and progress[-1] >= progress[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if no_improvement_count >= stagnation_threshold:
            for colony_idx in range(colony_count):
                num_to_reinit = max(1, ants_per_colony // 10)
                for _ in range(num_to_reinit):
                    new_solution = [random.choice(list(np.arange(lb[i], ub[i] + 0.5, 0.5))) for i in range(dim)]
                    archive = update_archive_with_crowding(archive, (np.array(new_solution), objf(np.array(new_solution))))
            no_improvement_count = 0
    return archive, progress


# =============================================================================
# ------------------------- Experiment Runner -------------------------------
# =============================================================================

def run_experiments(POP, ITER, runs: int = 1, use_random_instance: bool = False, num_tasks: int = 10
                   ) -> Tuple[Dict[str, Any], Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]], List[Dict[str, Any]]]:
    """
    Run multiple independent experiments for Adaptive MOHHO, Adaptive MOPSO, Improved MOACO, and Baseline.
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
    
    results = {
        "MOHHO": {"best_makespan": [], "absolute_hypervolume": [], "spread": []},
        "PSO": {"best_makespan": [], "absolute_hypervolume": [], "spread": []},
        "MOACO": {"best_makespan": [], "absolute_hypervolume": [], "spread": []},
        "Baseline": {"makespan": []}
    }

    archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]] = {"MOHHO": [], "PSO": [], "MOACO": []}
    base_schedules = []

    for run in range(runs):
        logging.info(f"Run {run+1}/{runs}...")
        base_schedule, base_ms = model.baseline_allocation()
        results["Baseline"]["makespan"].append(base_ms)
        base_schedules.append(base_schedule)

        hho_iter = ITER
        search_agents_no = POP
        archive_hho, _ = MOHHO_with_progress(lambda x: multi_objective(x, model), lb_current, ub_current, dim, search_agents_no, hho_iter)
        best_ms_hho = min(archive_hho, key=lambda entry: entry[1][0])[1][0] if archive_hho else None
        results["MOHHO"]["best_makespan"].append(best_ms_hho)
        archives_all["MOHHO"].append(archive_hho)

        objectives = [lambda x: objective_makespan(x, model),
                      lambda x: objective_total_cost(x, model),
                      lambda x: objective_neg_utilization(x, model)]
        optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                        pop=POP, c2=1.05, w_max=0.9, w_min=0.4,
                        disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
        _ = optimizer.run(max_iter=ITER)
        archive_pso = optimizer.archive
        best_ms_pso = min(archive_pso, key=lambda entry: entry[1][0])[1][0] if archive_pso else None
        results["PSO"]["best_makespan"].append(best_ms_pso)
        archives_all["PSO"].append(archive_pso)

        ant_count = POP
        moaco_iter = ITER
        archive_moaco, _ = MOACO_improved(
            lambda x: multi_objective(x, model),
            model.tasks, lb_current, ub_current, ant_count, moaco_iter,
            alpha=1.0, beta=2.0, evaporation_rate=0.1,
            colony_count= (ant_count//2))

        best_ms_moaco = min(archive_moaco, key=lambda entry: entry[1][0])[1][0] if archive_moaco else None
        results["MOACO"]["best_makespan"].append(best_ms_moaco)
        archives_all["MOACO"].append(archive_moaco)

    fixed_ref = compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")

    global_lower_bound = compute_combined_ideal(archives_all)

    for alg in ["MOHHO", "PSO", "MOACO"]:
        for archive in archives_all[alg]:
            abs_hv = absolute_hypervolume_fixed(archive, fixed_ref, global_lower_bound)
            results[alg]["absolute_hypervolume"].append(abs_hv)
            sp = compute_spread(archive)
            results[alg]["spread"].append(sp)

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
# ------------------------- Main Comparison ----------------------------------
# =============================================================================

if __name__ == '__main__':
    runs = 1 # Number of independent runs for statistical significance
    use_random_instance = False  # Set True for random instances
    num_tasks = 100
    POP = 50
    ITER = 300

    if use_random_instance:
        tasks_for_exp = generate_random_tasks(num_tasks, {"Developer": 10, "Manager": 2, "Tester": 3})
    else:
        tasks_for_exp = get_default_tasks()

    results, archives_all, base_schedules = run_experiments(POP, ITER, runs=runs, use_random_instance=use_random_instance, num_tasks=num_tasks)
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    means, stds = statistical_analysis(results)
    
    
    plot_convergence({alg: results[alg]["best_makespan"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Best Makespan (hours)")
    plot_convergence({alg: results[alg]["absolute_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Normalized Hypervolume (%)")
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Spread (Diversity)")
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    
    fixed_ref = compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")
    last_archives = [archives_all[alg][-1] for alg in ["MOHHO", "PSO", "MOACO"]]
    plot_pareto_2d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'], ref_point=fixed_ref)
    plot_pareto_3d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'], ref_point=fixed_ref)
    
    last_baseline = base_schedules[-1]
    last_makespan = results["Baseline"]["makespan"][-1]
    #plot_gantt(last_baseline, f"Baseline Schedule (Greedy Allocation)\nMakespan: {last_makespan:.2f} hrs")
    
    logging.info("Experiment complete. Results saved to 'experiment_results.json'.")

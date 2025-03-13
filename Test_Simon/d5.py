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
    # Enhanced initialization using chaotic map
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    # Initialize self-adaptive step sizes for each hawk and dimension.
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    t = 0
    diversity_threshold = 0.1 * np.mean(ub - lb)
    while t < max_iter:
        # Non-linear decaying escape energy (using cosine schedule)
        E1 = 2 * math.cos((t / max_iter) * (math.pi / 2))
        for i in range(search_agents_no):
            X[i, :] = discretize_vector(np.clip(X[i, :], lb, ub), lb, ub)
            f_val = objf(X[i, :])
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))
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
            new_x = old_x + step_sizes[i, :] * (X[i, :] - old_x)
            new_x = discretize_vector(np.clip(new_x, lb, ub), lb, ub)
            new_obj = np.linalg.norm(objf(new_x))
            if new_obj < old_obj:
                step_sizes[i, :] *= 0.95
            else:
                step_sizes[i, :] *= 1.05
            X[i, :] = new_x.copy()
        dists = [np.linalg.norm(X[i] - X[j]) for i in range(search_agents_no) for j in range(i+1, search_agents_no)]
        avg_dist = np.mean(dists) if dists else 0
        if avg_dist < diversity_threshold:
            obj_values = [np.linalg.norm(objf(X[i])) for i in range(search_agents_no)]
            worst_idx = np.argmax(obj_values)
            if archive:
                base = random.choice(archive)[0]
                new_hawk = base + np.random.uniform(-0.5, 0.5, size=dim)
                X[worst_idx, :] = discretize_vector(new_hawk, lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)
            else:
                X[worst_idx, :] = discretize_vector(chaotic_map_initialization(lb, ub, dim, 1)[0], lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)
        best_makespan = np.min([objf(X[i, :])[0] for i in range(search_agents_no)])
        progress.append(best_makespan)
        t += 1
    return archive, progress


class PSO:
    """
    Adaptive MOPSO (Multi-Objective Particle Swarm Optimization) for RCPSP with several enhancements.
    
    Enhancements and Scientific Justifications:
      1. Self-adaptive Inertia Weight Update:
         - Dynamically adjusts the inertia weight based on performance improvements to balance exploration and exploitation.
         - Citation: Zhang et al. (2018), Adaptive MOPSO approaches.
         - URL: https://doi.org/10.1007/s11761-018-0231-7

      2. Periodic Mutation/Disturbance:
         - Introduces random disturbances (mutation) in the particle positions to prevent premature convergence.
         - Citation: Sun et al. (2019), "Chaotic Multi-Objective Particle Swarm Optimization Algorithm Incorporating Clone Immunity"
         - URL: https://doi.org/10.3390/math7020146

      3. Archive Update via Crowding Distance:
         - Maintains an external archive of non-dominated solutions using a NSGA-II style crowding distance measure to preserve diversity.
         - Citation: Deb et al. (2002), "Multi-Objective Optimization Using Evolutionary Algorithms"
         - URL: https://doi.org/10.1109/4235.996017

      4. Hypercube-Based Leader Selection:
         - Divides the objective space into hypercubes and selects leaders based on the inverse density of solutions in each cell,
           promoting diverse search directions.
         - Citation: Coello Coello et al. (2004)
         - URL: https://doi.org/10.1080/03052150410001647966

    This class provides methods to initialize the swarm, update velocities and positions, manage the archive,
    and run the optimization for a specified number of iterations.
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
            pos = np.array([random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5))) for i in range(dim)])
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
        
    def select_leader_hypercube(self) -> List[np.ndarray]:
        """
        Select leader particles using hypercube division of the archive.
        """
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        objs = np.array([entry[1] for entry in self.archive])
        num_bins = 5
        mins = np.min(objs, axis=0)
        maxs = np.max(objs, axis=0)
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        cell_indices = []
        cell_counts = {}
        for entry in self.archive:
            idx = tuple(((entry[1] - mins) / ranges * num_bins).astype(int))
            idx = tuple(min(x, num_bins - 1) for x in idx)
            cell_indices.append(idx)
            cell_counts[idx] = cell_counts.get(idx, 0) + 1
        leaders = []
        weights = [1 / cell_counts[cell_indices[i]] for i in range(len(self.archive))]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        for _ in range(self.pop):
            chosen = np.random.choice(len(self.archive), p=probs)
            leaders.append(self.archive[chosen][0])
        return leaders

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
        Update the swarm by moving each particle, applying self-adaptive inertia weight updates,
        and periodic disturbance operations.
        """
        self.iteration += 1
        leaders = self.select_leader_hypercube()
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj = np.linalg.norm(self.evaluate(old_pos))
            r2 = random.random()
            guide = leaders[idx]
            # Standard PSO velocity and position update.
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([clip_round_half(new_pos[i], self.lb[i], self.ub[i]) for i in range(self.dim)])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)
            particle['pbest'] = new_pos.copy()
            # Update inertia weight based on performance.
            new_obj = np.linalg.norm(self.evaluate(new_pos))
            if new_obj < old_obj:
                particle['w'] = max(particle['w'] * 0.95, self.w_min)
            else:
                particle['w'] = min(particle['w'] * 1.05, self.w_max)
            self.disturbance_operation(particle)
        self.update_archive()
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()
        positions = np.array([p['position'] for p in self.swarm])
        if len(positions) > 1:
            pairwise_dists = [np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions))]
            avg_distance = np.mean(pairwise_dists)
            if avg_distance < 0.1 * np.mean(self.ub - self.lb):
                idx_to_mutate = random.randint(0, self.pop - 1)
                self.swarm[idx_to_mutate]['position'] = np.array([random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5))) for i in range(self.dim)])
                self.swarm[idx_to_mutate]['obj'] = self.evaluate(self.swarm[idx_to_mutate]['position'])
        self.update_archive()

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
            convergence: A list of the best makespan values recorded per iteration.
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
                    alpha: float = 1.0,   # Influence of pheromone trails. (Typical range: 1.0 or more)
                    beta: float = 2.0,    # Influence of heuristic information. (Typical range: 2.0 to 3.0)
                    evaporation_rate: float = 0.1,  # Rate at which pheromone evaporates (0 to 1, lower means slower evaporation)
                    Q: float = 100.0,     # Constant used for pheromone deposit scaling.
                    P: float = 0.6,       # (Unused in this version, but can be used for probabilistic selection adjustments)
                    w1: float = 1.0,      # Weighting for deposit update in pheromone trails.
                    w2: float = 1.0,      # (Unused here; can be used for additional weighting factors)
                    sigma_share: float = 1.0,  # (Unused here; might be used for diversity sharing)
                    lambda3: float = 2.0, # Scaling factor for pheromone deposit.
                    lambda4: float = 5.0, # (Unused in this snippet; can be used for further deposit adjustments)
                    colony_count: int = 10,  # Number of colonies to divide the ant population into.
                    # Optional heuristic function: if provided, must accept (task, candidate_value, task_index)
                    heuristic_obj: Optional[Callable[[Dict[str, Any], float, int], List[float]]] = None,
                    worker_cost: Optional[Dict[str, int]] = None
                   ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    MOACO_improved implements a multi-objective Ant Colony Optimization algorithm for RCPSP.
    
    Overall procedure:
      1. **Heuristic Initialization (per task):**
         - For each task, possible allocation values (from lb to ub, in half-step increments) are evaluated.
         - A full solution is formed by setting all tasks to their minimum allocation except the one under evaluation.
         - The full objective vector is computed via the provided objf.
         - The candidate objective values are normalized (min–max scaling) and ranked using non-dominated sorting.
         - The heuristic value for each candidate allocation is set to 1/(rank + ε), so that lower-ranked (better) candidates have higher heuristic values.
      
      2. **Pheromone Initialization:**
         - For each colony and each task, a pheromone dictionary is initialized.
         - Initially, every possible allocation gets the same pheromone value (set to 1.0).

      3. **Main Iteration Loop:**
         - For each colony, ants construct solutions based on the pheromone and heuristic values.
         - **Solution Construction:**
             - For each task, candidate allocation values are selected probabilistically.
             - The probability for each allocation is computed as (pheromone^alpha * heuristic^beta).
         - **Extended Local Search:**
             - Once a candidate solution is built, local search is performed by generating neighbors (by perturbing each allocation ±0.5).
             - The full objective vector for each neighbor is computed, and normalization is applied.
             - The candidate with the best (lowest mean normalized) objective is chosen.
             - If no improvement is observed, extended neighbors (with larger perturbations, e.g., ±2) are evaluated.
         - **Archive Update:**
             - The non-dominated archive is updated using the new candidate solutions.
         - **Adaptive Evaporation:**
             - The pheromone values are evaporated by a rate that can be adjusted based on the variance of pheromone values.
         - **Deposit Update:**
             - Pheromone deposit is computed based on the normalized crowding distance of archive solutions.
             - The deposit is scaled by lambda3 and a decay factor (based on iteration progress).
         - **Multi-Colony Pheromone Reinitialization:**
             - If pheromone variance becomes too low (indicating potential stagnation), reinitialization is performed.
             - Then, the pheromone matrices from all colonies are merged and synchronized.
         - **Progress Update:**
             - The mean normalized objective value for the current iteration is computed and stored.
         - **Stagnation Handling:**
             - If no improvement is detected over a number of iterations, some ants are reinitialized.
    
    Returns:
      archive: A list of non-dominated complete solutions (each as a tuple (decision vector, objective vector)).
      progress: A list recording the best aggregated (mean-normalized) objective value per iteration.
    """
    import numpy as np
    import random

    # ---------------- Helper: Global Normalization ----------------
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Min–max scale each column (objective) of mat (rows are candidate solutions) to the [0,1] interval.
        If an objective column has zero range, all entries are set to 0.5.
        """
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            if range_val == 0:
                norm[:, d] = 0.5  # Neutral value if no variation exists
            else:
                norm[:, d] = (mat[:, d] - mins[d]) / range_val
        return norm, mins, maxs

    # ---------------- Helper: Global Archive Crowding Distance ----------------
    def normalized_crowding_distance(archive):
        """
        Compute the crowding distance for each solution in the archive, using normalized objectives.
        This helps in preserving diversity among solutions.
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

    # ---------------- Helper: Fast Non-Dominated Sorting ----------------
    def fast_non_dominated_sort(candidates: List[List[float]]) -> List[int]:
        """
        Perform fast non-dominated sorting on the candidate objective vectors.
        Return the rank (Pareto front index) for each candidate.
        Lower rank means a better (more non-dominated) candidate.
        """
        n = len(candidates)
        S = [[] for _ in range(n)]
        domination_count = [0] * n
        ranks = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if candidate j dominates candidate i.
                if (all(candidates[j][k] <= candidates[i][k] for k in range(len(candidates[i]))) and
                    any(candidates[j][k] < candidates[i][k] for k in range(len(candidates[i])))):
                    domination_count[i] += 1
                elif (all(candidates[i][k] <= candidates[j][k] for k in range(len(candidates[i]))) and
                      any(candidates[i][k] < candidates[j][k] for k in range(len(candidates[i])))):
                    S[i].append(j)
            if domination_count[i] == 0:
                ranks[i] = 1  # First Pareto front.
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

    # ---------------- Default Heuristic Function ----------------
    # If no heuristic function is provided, define one that evaluates the full objective vector
    # by setting all tasks to their minimum allocation except the one under evaluation.
    if heuristic_obj is None:
        def default_full_objective_heuristic(task: Dict[str, Any], v: float, i: int) -> List[float]:
            # Start with a baseline: every task gets its minimum allocation.
            x = np.array([t["min"] for t in tasks])
            # For task i, assign the candidate value v.
            x[i] = v
            # Return the full objective vector for this solution.
            return objf(x).tolist()
        heuristic_obj = default_full_objective_heuristic

    dim = len(lb)
    colony_pheromones = []  # One pheromone matrix per colony (list of lists: each task gets a dict of pheromone values).
    colony_heuristics = []  # One heuristic matrix per colony.
    
    # ---------------- Heuristic Initialization (per task) ----------------
    # For each task, calculate heuristic values for all possible allocation values.
    for i in range(dim):
        possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
        candidate_objs = []
        for v in possible_values:
            candidate = np.array([t["min"] for t in tasks])
            candidate[i] = v  # Only change the allocation for task i.
            candidate_objs.append(objf(candidate).tolist())
        # Normalize the candidate objective vectors.
        norm_candidates, _, _ = normalize_matrix(np.array(candidate_objs))
        # Use fast non-dominated sorting to rank the candidates.
        ranks = fast_non_dominated_sort(norm_candidates.tolist())
        # Set the heuristic for each candidate as the inverse of its rank.
        task_heuristic = {v: 1.0 / (ranks[idx] + 1e-6) for idx, v in enumerate(possible_values)}
        # Initialize the same heuristic and pheromone structure for each colony.
        for colony_idx in range(colony_count):
            if colony_idx >= len(colony_pheromones):
                colony_pheromones.append([])
                colony_heuristics.append([])
            colony_pheromones[colony_idx].append({v: 1.0 for v in possible_values})  # Initial pheromone: 1.0 for each value.
            colony_heuristics[colony_idx].append(task_heuristic)

    # ---------------- Initialize Archive and Progress Tracking ----------------
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    best_global = float('inf')
    no_improvement_count = 0
    stagnation_threshold = 10  # Number of iterations to wait before reinitializing some ants.
    eps = 1e-6  # Small epsilon to avoid division by zero.

    # ---------------- Main Iteration Loop ----------------
    for iteration in range(max_iter):
        colony_solutions = []  # Store solutions constructed by ants in all colonies.
        
        # ---- Solution Construction (for each colony) ----
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            for _ in range(ants_per_colony):
                solution = []
                # For each task, choose an allocation based on pheromone and heuristic.
                for i in range(dim):
                    possible_values = list(pheromone[i].keys())
                    probs = []
                    for v in possible_values:
                        tau = np.nan_to_num(pheromone[i][v], nan=0.0)
                        h_val = np.nan_to_num(heuristic[i][v], nan=0.0)
                        # Compute probability proportional to (tau^alpha * h_val^beta)
                        probs.append((tau ** alpha) * (h_val ** beta))
                    total = sum(probs)
                    # If total probability is zero or non-finite, assign uniform probability.
                    if not np.isfinite(total) or total <= 0:
                        probs = [1.0 / len(probs)] * len(probs)
                    else:
                        probs = [p / total for p in probs]
                    # Roulette wheel selection based on cumulative probabilities.
                    r = random.random()
                    cumulative = 0.0
                    chosen = possible_values[-1]  # Default selection if loop does not break.
                    for idx, v in enumerate(possible_values):
                        cumulative += probs[idx]
                        if r <= cumulative:
                            chosen = v
                            break
                    solution.append(chosen)
                # ---- Extended Local Search: Generate neighbors by perturbing each task's allocation ----
                candidates = [solution]
                for i in range(dim):
                    for delta in [-0.5, 0.5]:
                        neighbor = solution.copy()
                        # Ensure the new value is within bounds and in half-step increments.
                        neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
                        candidates.append(neighbor)
                # Evaluate all candidates using the full objective function.
                cand_objs = [objf(np.array(cand)) for cand in candidates]
                cand_objs = np.array(cand_objs, dtype=float)
                # If the archive exists, combine candidate objectives with archive objectives for normalization.
                if archive:
                    arch_objs = np.array([entry[1] for entry in archive])
                    union_objs = np.vstack([cand_objs, arch_objs])
                else:
                    union_objs = cand_objs
                norm_union, _, _ = normalize_matrix(union_objs)
                # Use only the candidate portion for selection.
                norm_cand = norm_union[:len(candidates), :]
                # Compute the mean normalized objective value for each candidate.
                mean_norm = norm_cand.mean(axis=1)
                best_index = np.argmin(mean_norm)
                best_neighbor = candidates[best_index]
                best_obj = cand_objs[best_index]
                # If the best candidate is the original solution (no improvement),
                # try extended perturbations (±2 steps) for further exploration.
                if best_index == 0:
                    extended_candidates = [solution]
                    for i in range(dim):
                        for delta in [-2.0, 2.0]:
                            neighbor = solution.copy()
                            neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
                            extended_candidates.append(neighbor)
                    ext_objs = [objf(np.array(cand)) for cand in extended_candidates]
                    ext_objs = np.array(ext_objs, dtype=float)
                    if archive:
                        union_ext = np.vstack([ext_objs, arch_objs])
                    else:
                        union_ext = ext_objs
                    norm_ext, _, _ = normalize_matrix(union_ext)
                    norm_ext = norm_ext[:len(extended_candidates), :]
                    mean_ext = norm_ext.mean(axis=1)
                    ext_best_index = np.argmin(mean_ext)
                    best_neighbor = extended_candidates[ext_best_index]
                    best_obj = ext_objs[ext_best_index]
                solution = best_neighbor
                # Store the constructed solution and its objective value.
                colony_solutions.append((solution, best_obj))
        
        # ---- Archive Update: Update the global archive using non-dominated sorting and crowding distance.
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), obj_val))
        
        # ---- Adaptive Evaporation: Adjust pheromone evaporation based on pheromone variance.
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.nan_to_num(np.array(all_values), nan=0.0)
            # If variance is low, increase the evaporation rate to encourage exploration.
            var_pheromone = np.var(all_values)
            evap_rate_current = min(0.9, evaporation_rate * 1.5) if var_pheromone < 0.001 else evaporation_rate
            for i in range(dim):
                for v in pheromone[i]:
                    pheromone[i][v] *= (1 - evap_rate_current)
        
        # ---- Deposit Update: Increase pheromone on good solutions using normalized crowding distance.
        crowding = normalized_crowding_distance(archive)
        max_cd = np.max(crowding) if len(crowding) > 0 else 1.0
        if not np.isfinite(max_cd) or max_cd <= 0:
            max_cd = 1.0
        decay_factor = 1.0 - (iteration / max_iter)  # Decays over time to reduce pheromone deposit.
        for idx, (sol, obj_val) in enumerate(archive):
            deposit = w1 * lambda3 * (crowding[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i, v in enumerate(sol):
                    colony_pheromones[colony_idx][i][v] += deposit
        
        # ---- Multi-Colony Pheromone Reinitialization: Merge and synchronize pheromone matrices across colonies.
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.nan_to_num(np.array(all_values), nan=0.0)
            if np.var(all_values) < 0.001:
                # If variance is too low, reinitialize a fraction of pheromones.
                for i in range(dim):
                    possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
                    pheromone[i] = {v: 1.0 for v in possible_values}
        # Merge pheromone information from all colonies for consistency.
        merged_pheromone = []
        for i in range(dim):
            merged = {}
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            for v in possible_values:
                # Average the pheromone values for each possible allocation across colonies.
                val = sum(colony_pheromones[colony_idx][i].get(v, 0) for colony_idx in range(colony_count)) / colony_count
                merged[v] = val
            merged_pheromone.append(merged)
        for colony_idx in range(colony_count):
            colony_pheromones[colony_idx] = [merged_pheromone[i].copy() for i in range(dim)]
        
        # ---- Progress Update: Record the mean normalized objective value for the iteration.
        iter_objs = np.array([obj_val for (_, obj_val) in colony_solutions], dtype=float)
        norm_iter, _, _ = normalize_matrix(iter_objs)
        mean_iter = norm_iter.mean(axis=1)
        current_best = mean_iter.min()
        progress.append(current_best)
        if current_best < best_global:
            best_global = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        # ---- Stagnation Handling: If no improvement for several iterations, reinitialize some ants.
        if no_improvement_count >= stagnation_threshold:
            for colony_idx in range(colony_count):
                num_to_reinit = max(1, ants_per_colony // 10)
                for _ in range(num_to_reinit):
                    new_solution = [random.choice(list(np.arange(lb[i], ub[i] + 0.5, 0.5))) for i in range(dim)]
                    archive = update_archive_with_crowding(archive, (np.array(new_solution), objf(np.array(new_solution))))
            no_improvement_count = 0
    return archive, progress


# =============================================================================
# ------------------------- Grid search -------------------------------
# =============================================================================

def grid_search():
    """
    Grid search over algorithm parameters to tune multi-objective performance.
    
    For each algorithm (MOHHO, MOPSO (PSO), MOACO):
      - Population sizes: 100, 300, 500, 700, 1000
      - Iteration counts: 300, 500, 750, 1000, 2000
    Additionally for MOACO:
      - Colony count as a percentage of the ant population: 5%, 10%, 20%, 30%
    
    Each combination is run 5 times (to mitigate stochastic effects).
    Performance is evaluated using a combination of best makespan, absolute hypervolume,
    spread (diversity) and generational distance.
    
    The results are stored in a JSON file for further analysis.
    """

    # Define grid search ranges
    populations = [100, 300, 500, 700, 1000]
    iterations_list = [300, 500, 750, 1000, 2000]
    colony_percentages = [40 ,45, 50, 55, 60]  # Only for MOACO, 50% best so far
    runs = 1  # Independent runs per combination

    # Fixed RCPSP instance (using default tasks)
    workers = {"Developer": 10, "Manager": 2, "Tester": 3}
    worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
    tasks = get_default_tasks()  # Fixed instance for reproducibility

    # Dictionary to store grid search results per algorithm
    results_grid = {"MOHHO": [], "PSO": [], "MOACO": []}

    # Grid search for each algorithm
    for algorithm in ["MOACO"]:
        print("ny algorithm", algorithm)
        for pop in populations:
            print("ny pop:", pop)
            for iters in iterations_list:
                print("ny iter", iters)
                if algorithm == "MOACO":
                    # Loop over colony percentages (as a percentage of ant_count = pop)
                    for col_pct in colony_percentages:
                        print("Ny koloni", col_pct)
                        colony_count = max(1, int(pop * (col_pct / 100)))
                        metrics = {
                            "pop": pop,
                            "iters": iters,
                            "colony_percentage": col_pct,
                            "colony_count": colony_count,
                            "makespan": [],
                            "hypervolume": [],
                            "spread": [],
                            "generational_distance": []
                        }
                        for r in range(runs):
                            print("Ny run", r)
                            # Create RCPSP model instance
                            model = RCPSPModel(tasks, workers, worker_cost)
                            dim = len(model.tasks)
                            lb_current = np.array([task["min"] for task in tasks])
                            ub_current = np.array([task["max"] for task in tasks])
                            
                            # Run MOACO_improved with current grid parameters.
                            archive, _ = MOACO_improved(
                                lambda x: multi_objective(x, model),
                                tasks, workers, lb_current, ub_current,
                                ant_count=pop, max_iter=iters,
                                alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0,
                                colony_count=colony_count
                            )
                            # Extract performance metrics from the archive
                            if archive:
                                best_ms = min(entry[1][0] for entry in archive)
                                # For hypervolume and spread, define fixed reference and global lower bound based on archive.
                                objs = np.array([entry[1] for entry in archive])
                                fixed_ref = np.max(objs, axis=0)
                                global_lower_bound = np.min(objs, axis=0)
                                hv = absolute_hypervolume_fixed(archive, fixed_ref, global_lower_bound)
                                spread_val = compute_spread(archive)
                                gd = compute_generational_distance(archive, objs)
                            else:
                                best_ms, hv, spread_val, gd = None, None, None, None
                            
                            metrics["makespan"].append(best_ms)
                            metrics["hypervolume"].append(hv)
                            metrics["spread"].append(spread_val)
                            metrics["generational_distance"].append(gd)
                        
                        # Compute average metrics over runs
                        metrics["avg_makespan"] = np.mean([m for m in metrics["makespan"] if m is not None])
                        metrics["avg_hv"] = np.mean([h for h in metrics["hypervolume"] if h is not None])
                        metrics["avg_spread"] = np.mean([s for s in metrics["spread"] if s is not None])
                        metrics["avg_gd"] = np.mean([g for g in metrics["generational_distance"] if g is not None])
                        results_grid["MOACO"].append(metrics)
                else:
                    # For MOHHO and PSO
                    metrics = {
                        "pop": pop,
                        "iters": iters,
                        "makespan": [],
                        "hypervolume": [],
                        "spread": [],
                        "generational_distance": []
                    }
                    for r in range(runs):
                        model = RCPSPModel(tasks, workers, worker_cost)
                        dim = len(model.tasks)
                        lb_current = np.array([task["min"] for task in tasks])
                        ub_current = np.array([task["max"] for task in tasks])
                        
                        if algorithm == "MOHHO":
                            archive, _ = MOHHO_with_progress(
                                lambda x: multi_objective(x, model),
                                lb_current, ub_current, dim, pop, iters
                            )
                        elif algorithm == "PSO":
                            objectives = [
                                lambda x: objective_makespan(x, model),
                                lambda x: objective_total_cost(x, model),
                                lambda x: objective_neg_utilization(x, model)
                            ]
                            optimizer = PSO(
                                dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                                pop=pop, c2=1.05, w_max=0.9, w_min=0.4,
                                disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20
                            )
                            _ = optimizer.run(max_iter=iters)
                            archive = optimizer.archive
                        
                        if archive:
                            best_ms = min(entry[1][0] for entry in archive)
                            objs = np.array([entry[1] for entry in archive])
                            fixed_ref = np.max(objs, axis=0)
                            global_lower_bound = np.min(objs, axis=0)
                            hv = absolute_hypervolume_fixed(archive, fixed_ref, global_lower_bound)
                            spread_val = compute_spread(archive)
                            gd = compute_generational_distance(archive, objs)
                        else:
                            best_ms, hv, spread_val, gd = None, None, None, None

                        metrics["makespan"].append(best_ms)
                        metrics["hypervolume"].append(hv)
                        metrics["spread"].append(spread_val)
                        metrics["generational_distance"].append(gd)
                    
                    metrics["avg_makespan"] = np.mean([m for m in metrics["makespan"] if m is not None])
                    metrics["avg_hv"] = np.mean([h for h in metrics["hypervolume"] if h is not None])
                    metrics["avg_spread"] = np.mean([s for s in metrics["spread"] if s is not None])
                    metrics["avg_gd"] = np.mean([g for g in metrics["generational_distance"] if g is not None])
                    results_grid[algorithm].append(metrics)
    
    # Save the grid search results to a JSON file for further analysis.
    with open("grid_search_results.json", "w") as f:
        json.dump(results_grid, f, indent=4)
    print("Grid search complete. Results saved to grid_search_results.json.")

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
        archive_moaco, _ = MOACO_improved(lambda x: multi_objective(x, model), model.tasks, workers,
                                          lb_current, ub_current, ant_count, moaco_iter,
                                          alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0)
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
# ------------------------- Automated Unit Testing --------------------------
# =============================================================================

def run_unit_tests() -> None:
    """
    Run basic unit tests:
      1. Test that update_archive_with_crowding produces a non-dominated archive.
      2. Test that RCPSPModel.compute_schedule returns a feasible schedule.
    """
    sol1 = np.array([1, 2, 3])
    obj1 = np.array([10, 20, 30])
    sol2 = np.array([2, 3, 4])
    obj2 = np.array([12, 22, 32])
    archive = []
    archive = update_archive_with_crowding(archive, (sol1, obj1))
    archive = update_archive_with_crowding(archive, (sol2, obj2))
    if len(archive) != 1:
        logging.error("Unit Test Failed: Archive contains dominated solutions.")
    else:
        logging.info("Unit Test Passed: Archive update produces non-dominated set.")

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

def run_unit_tests() -> None:
    """
    Run basic unit tests:
      1. Test that update_archive_with_crowding produces a non-dominated archive.
      2. Test that RCPSPModel.compute_schedule returns a feasible schedule.
      3. Test that the new conversion functions handle edge cases and typical durations.
    """
    # ---------------- Archive and Schedule Tests ----------------
    sol1 = np.array([1, 2, 3])
    obj1 = np.array([10, 20, 30])
    sol2 = np.array([2, 3, 4])
    obj2 = np.array([12, 22, 32])
    archive = []
    archive = update_archive_with_crowding(archive, (sol1, obj1))
    archive = update_archive_with_crowding(archive, (sol2, obj2))
    if len(archive) != 1:
        logging.error("Unit Test Failed: Archive contains dominated solutions.")
    else:
        logging.info("Unit Test Passed: Archive update produces non-dominated set.")

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

    # ---------------- Conversion Function Tests ----------------
    # Test convert_hours_to_billable_days:
    # - For non-positive durations, expect 0.0 billable days.
    if convert_hours_to_billable_days(0) != 0.0 or convert_hours_to_billable_days(-5) != 0.0:
        logging.error("Unit Test Failed: convert_hours_to_billable_days did not return 0.0 for non-positive durations.")
    else:
        logging.info("Unit Test Passed: convert_hours_to_billable_days handles non-positive durations correctly.")

    # - For durations ≤ 4 hours, count as half day (0.5).
    if convert_hours_to_billable_days(1.25) != 0.5 or convert_hours_to_billable_days(4) != 0.5:
        logging.error("Unit Test Failed: convert_hours_to_billable_days did not count durations ≤4 hours as 0.5 day.")
    else:
        logging.info("Unit Test Passed: convert_hours_to_billable_days correctly counts durations ≤4 hours as half day.")

    # - For durations > 4 and ≤8 hours, count as full day (1.0).
    if convert_hours_to_billable_days(4.1) != 1.0 or convert_hours_to_billable_days(7.9) != 1.0 or convert_hours_to_billable_days(8) != 1.0:
        logging.error("Unit Test Failed: convert_hours_to_billable_days did not count durations >4 and ≤8 hours as 1 day.")
    else:
        logging.info("Unit Test Passed: convert_hours_to_billable_days correctly counts durations >4 and ≤8 hours as full day.")

    # - For durations >8 hours, ensure proper division into full days and remainder.
    #   10 hours → 8 + 2 hours → 1 full day + 0.5 day = 1.5 days.
    if convert_hours_to_billable_days(10) != 1.5:
        logging.error("Unit Test Failed: convert_hours_to_billable_days did not correctly count 10 hours as 1.5 days.")
    else:
        logging.info("Unit Test Passed: convert_hours_to_billable_days correctly converts 10 hours to 1.5 days.")
    #   13 hours → 8 + 5 hours → 1 full day + 1 full day (since 5 > 4) = 2.0 days.
    if convert_hours_to_billable_days(13) != 2.0:
        logging.error("Unit Test Failed: convert_hours_to_billable_days did not correctly count 13 hours as 2.0 days.")
    else:
        logging.info("Unit Test Passed: convert_hours_to_billable_days correctly converts 13 hours to 2.0 days.")
    #   16 hours → exactly 2 full days = 2.0 days.
    if convert_hours_to_billable_days(16) != 2.0:
        logging.error("Unit Test Failed: convert_hours_to_billable_days did not correctly count 16 hours as 2.0 days.")
    else:
        logging.info("Unit Test Passed: convert_hours_to_billable_days correctly converts 16 hours to 2.0 days.")

    # Test compute_billable_hours:
    # - A half day should be billed as 4 hours and a full day as 8 hours.
    #   3 hours (0.5 day) → 4 billable hours.
    if compute_billable_hours(3) != 4:
        logging.error("Unit Test Failed: compute_billable_hours did not convert 3 hours to 4 billable hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_hours correctly converts 3 hours to 4 billable hours.")
    #   4.1 hours (1 day) → 8 billable hours.
    if compute_billable_hours(4.1) != 8:
        logging.error("Unit Test Failed: compute_billable_hours did not convert 4.1 hours to 8 billable hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_hours correctly converts 4.1 hours to 8 billable hours.")
    #   10 hours (1.5 days) → 12 billable hours.
    if compute_billable_hours(10) != 12:
        logging.error("Unit Test Failed: compute_billable_hours did not convert 10 hours to 12 billable hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_hours correctly converts 10 hours to 12 billable hours.")
    #   13 hours (2 days) → 16 billable hours.
    if compute_billable_hours(13) != 16:
        logging.error("Unit Test Failed: compute_billable_hours did not convert 13 hours to 16 billable hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_hours correctly converts 13 hours to 16 billable hours.")

    # Test compute_billable_cost:
    # For a full worker (allocation = 1)
    #   - 3 hours → 4 billable hours, cost = 4 * wage_rate.
    if compute_billable_cost(3, 1, 50) != 200:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for full worker with 3 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for full worker with 3 hours.")
    #   - 10 hours → 12 billable hours, cost = 12 * wage_rate.
    if compute_billable_cost(10, 1, 50) != 600:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for full worker with 10 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for full worker with 10 hours.")

    # For a half worker (allocation = 0.5)
    #   - 3 hours: 4 billable hours → half worker billed hours = 4 / 2 = 2; cost = 2 * wage_rate.
    if compute_billable_cost(1, 0.5, 50) != 100:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for half worker with 1 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for half worker with 1 hours.")
    if compute_billable_cost(3, 0.5, 50) != 200:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for half worker with 3 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for half worker with 3 hours.")
    if compute_billable_cost(8, 0.5, 50) != 400:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for half worker with 8 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for half worker with 8 hours.")

    if compute_billable_cost(10, 0.5, 50) != 500:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for half worker with 10 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for half worker with 10 hours.")

    if compute_billable_cost(18, 0.5, 50) != 900:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for half worker with 18 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for half worker with 18 hours.")

    # For a mixed allocation (e.g., 1.5, meaning one full worker + one half worker):
    #   - 3 hours: full worker cost = 4 * 50 = 200; half worker cost = (4 / 2) * 50 = 100; total = 300.
    if compute_billable_cost(3, 1.5, 50) != 300:

        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for mixed allocation (1.5) with 3 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for mixed allocation (1.5) with 3 hours.")
    #   - 10 hours: full worker cost = 12 * 50 = 600; half worker cost = (12 / 2) * 50 = 300; total = 900.
    if compute_billable_cost(10, 1.5, 50) != 900:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for mixed allocation (1.5) with 10 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for mixed allocation (1.5) with 10 hours.")
    #   - 13 hours (16 billable hours) with allocation = 2.5:
    #       Full workers: 2 → 2 * 16 = 32; half worker: 1 → 16 / 2 = 8; total = 40; wage_rate = 40; cost = 40 * 40 = 1600.
    if compute_billable_cost(13, 2.5, 40) != 1600:
        logging.error("Unit Test Failed: compute_billable_cost did not compute cost correctly for mixed allocation (2.5) with 13 hours.")
    else:
        logging.info("Unit Test Passed: compute_billable_cost correctly computes cost for mixed allocation (2.5) with 13 hours.")


# =============================================================================
# ------------------------- Main Comparison ----------------------------------
# =============================================================================

if __name__ == '__main__':
    """
    In the benchmark experiments,we set 
    the maximum iteration number to be 500, 
    the number of search agents to be 100, and 
    the  maximum  archive  size  to be 100.
    To  obtain  the statistical results, the MOHHO and others are run 10 times.
    Yüzgeç & Kuşoğlu (2020)
    """
    #run_unit_tests()
    #grid_search()
    runs = 1 # Number of independent runs for statistical significance
    use_random_instance = False  # Set True for random instances
    num_tasks = 10
    POP = 80
    ITER = 200

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

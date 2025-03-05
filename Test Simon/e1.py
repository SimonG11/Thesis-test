#!/usr/bin/env python3
"""
Improved Multi-Objective Comparison for RCPSP using Adaptive MOHHO, Adaptive MOPSO, and Improved MOACO
with Extended Real-World Constraints

This script implements and compares three metaheuristic algorithms for the Resource-Constrained 
Project Scheduling Problem (RCPSP) with multiple objectives. The extended model includes:
  - Deadlines, priorities, and various delay factors (vendor, compliance, quality, material, etc.)
  - Risk multipliers (country risk, complexity) and cost risk/energy multipliers to model financial constraints.
  - Workload balance as an objective.
  
The multi-objective vector consists of:
  [Makespan, Total Cost, -Average Utilization, Weighted Tardiness, Workload Balance]

Author: Simon Gottschalk
Date: 2025-02-13
"""

import numpy as np
import matplotlib.pyplot as plt
import random, math, time, copy, json, logging
from typing import List, Tuple, Dict, Any, Callable, Optional
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import f_oneway

# ----------------------------- Logging Setup -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------- Reproducibility -----------------------------
def initialize_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

initialize_seed(42)

# =============================================================================
# -------------------------- Helper Functions -------------------------------
# =============================================================================
def dominates(obj_a: np.ndarray, obj_b: np.ndarray, epsilon: float = 1e-6) -> bool:
    less_equal = np.all(obj_a <= obj_b + epsilon)
    strictly_less = np.any(obj_a < obj_b - epsilon)
    return less_equal and strictly_less

def levy(dim: int) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.power(np.abs(v), 1 / beta))

def find_earliest_start(earliest: float, duration: float, allocated: int,
                        scheduled_tasks: List[Dict[str, Any]],
                        capacity: int, resource: str, epsilon: float = 1e-6) -> float:
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

def chaotic_map_initialization(lb: np.ndarray, ub: np.ndarray, dim: int, n_agents: int) -> np.ndarray:
    r = 4.0
    population = np.zeros((n_agents, dim))
    for i in range(n_agents):
        x = np.random.rand(dim)
        for _ in range(10):
            x = r * x * (1 - x)
        population[i, :] = lb + x * (ub - lb)
    return population

# =============================================================================
# ----------------------- Extended Task Generation --------------------------
# =============================================================================
def get_extended_tasks() -> List[Dict[str, Any]]:
    """
    Returns a set of tasks with additional fields for various constraints.
    For each task, attributes (with defaults if not provided) include:
      - deadline, priority
      - vendor_delay, compliance_delay, quality_assurance_delay, skill_gap_delay,
        material_delay, equipment_delay, contingency_time, scope_creep_delay,
        communication_delay, env_delay
      - country_risk_factor, complexity_factor
      - cost_risk_factor, energy_cost_multiplier
    """
    return [
        {"id": 1, "task_name": "Requirements Gathering", "base_effort": 80,  "min": 1, "max": 14,
         "dependencies": [], "resource": "Manager", "deadline": 20, "priority": 1,
         "vendor_delay": 0, "compliance_delay": 0, "quality_assurance_delay": 0,
         "skill_gap_delay": 0, "material_delay": 0, "equipment_delay": 0,
         "contingency_time": 1, "scope_creep_delay": 0, "communication_delay": 0, "env_delay": 0,
         "country_risk_factor": 1, "complexity_factor": 1, "cost_risk_factor": 0, "energy_cost_multiplier": 0},
        {"id": 2, "task_name": "System Design", "base_effort": 100, "min": 1, "max": 14,
         "dependencies": [1], "resource": "Manager", "deadline": 40, "priority": 1,
         "vendor_delay": 0, "compliance_delay": 2, "quality_assurance_delay": 0,
         "skill_gap_delay": 0, "material_delay": 0, "equipment_delay": 0,
         "contingency_time": 1, "scope_creep_delay": 1, "communication_delay": 0, "env_delay": 0,
         "country_risk_factor": 1.05, "complexity_factor": 1.1, "cost_risk_factor": 0.02, "energy_cost_multiplier": 0},
        {"id": 3, "task_name": "Module 1 Development", "base_effort": 150, "min": 1, "max": 14,
         "dependencies": [2], "resource": "Developer", "deadline": 80, "priority": 2,
         "vendor_delay": 1, "compliance_delay": 0, "quality_assurance_delay": 0,
         "skill_gap_delay": 1, "material_delay": 0, "equipment_delay": 0,
         "contingency_time": 2, "scope_creep_delay": 0, "communication_delay": 0, "env_delay": 0,
         "country_risk_factor": 1.1, "complexity_factor": 1.2, "cost_risk_factor": 0.03, "energy_cost_multiplier": 0},
        {"id": 4, "task_name": "Module 2 Development", "base_effort": 150, "min": 1, "max": 14,
         "dependencies": [2], "resource": "Developer", "deadline": 80, "priority": 2,
         "vendor_delay": 1, "compliance_delay": 0, "quality_assurance_delay": 0,
         "skill_gap_delay": 1, "material_delay": 0, "equipment_delay": 0,
         "contingency_time": 2, "scope_creep_delay": 0, "communication_delay": 0, "env_delay": 0,
         "country_risk_factor": 1.1, "complexity_factor": 1.2, "cost_risk_factor": 0.03, "energy_cost_multiplier": 0},
        {"id": 5, "task_name": "Integration", "base_effort": 100, "min": 1, "max": 14,
         "dependencies": [4], "resource": "Developer", "deadline": 100, "priority": 3,
         "vendor_delay": 0, "compliance_delay": 0, "quality_assurance_delay": 0,
         "skill_gap_delay": 0, "material_delay": 1, "equipment_delay": 0,
         "contingency_time": 1, "scope_creep_delay": 1, "communication_delay": 0, "env_delay": 0,
         "country_risk_factor": 1, "complexity_factor": 1.1, "cost_risk_factor": 0.01, "energy_cost_multiplier": 0},
        {"id": 6, "task_name": "Testing", "base_effort": 100, "min": 1, "max": 14,
         "dependencies": [4], "resource": "Tester", "deadline": 110, "priority": 2,
         "vendor_delay": 0, "compliance_delay": 0, "quality_assurance_delay": 2,
         "skill_gap_delay": 0, "material_delay": 0, "equipment_delay": 1,
         "contingency_time": 1, "scope_creep_delay": 0, "communication_delay": 0, "env_delay": 0,
         "country_risk_factor": 1, "complexity_factor": 1, "cost_risk_factor": 0.02, "energy_cost_multiplier": 0},
        {"id": 7, "task_name": "Acceptance Testing", "base_effort": 100, "min": 1, "max": 14,
         "dependencies": [4], "resource": "Tester", "deadline": 110, "priority": 2,
         "vendor_delay": 0, "compliance_delay": 0, "quality_assurance_delay": 2,
         "skill_gap_delay": 0, "material_delay": 0, "equipment_delay": 1,
         "contingency_time": 1, "scope_creep_delay": 0, "communication_delay": 0, "env_delay": 0,
         "country_risk_factor": 1, "complexity_factor": 1, "cost_risk_factor": 0.02, "energy_cost_multiplier": 0},
        {"id": 8, "task_name": "Documentation", "base_effort": 100, "min": 1, "max": 14,
         "dependencies": [4], "resource": "Developer", "deadline": 90, "priority": 4,
         "vendor_delay": 0, "compliance_delay": 0, "quality_assurance_delay": 0,
         "skill_gap_delay": 0, "material_delay": 0, "equipment_delay": 0,
         "contingency_time": 1, "scope_creep_delay": 2, "communication_delay": 1, "env_delay": 0,
         "country_risk_factor": 1, "complexity_factor": 1, "cost_risk_factor": 0.01, "energy_cost_multiplier": 0},
        {"id": 9, "task_name": "Training", "base_effort": 50, "min": 1, "max": 14,
         "dependencies": [7, 8], "resource": "Tester", "deadline": 120, "priority": 3,
         "vendor_delay": 0, "compliance_delay": 0, "quality_assurance_delay": 0,
         "skill_gap_delay": 0, "material_delay": 0, "equipment_delay": 0,
         "contingency_time": 1, "scope_creep_delay": 0, "communication_delay": 1, "env_delay": 0,
         "country_risk_factor": 1, "complexity_factor": 1, "cost_risk_factor": 0.01, "energy_cost_multiplier": 0},
        {"id": 10, "task_name": "Deployment", "base_effort": 70, "min": 2, "max": 14,
         "dependencies": [7, 9], "resource": "Manager", "deadline": 130, "priority": 1,
         "vendor_delay": 0, "compliance_delay": 0, "quality_assurance_delay": 0,
         "skill_gap_delay": 0, "material_delay": 0, "equipment_delay": 0,
         "contingency_time": 2, "scope_creep_delay": 0, "communication_delay": 0, "env_delay": 1,
         "country_risk_factor": 1.05, "complexity_factor": 1.1, "cost_risk_factor": 0.02, "energy_cost_multiplier": 0.05}
    ]

# =============================================================================
# ------------------ Extended RCPSP Model Definition ------------------------
# =============================================================================
class ExtendedRCPSPModel:
    """
    Extended RCPSP model incorporating additional delays and risk factors.
    The effective duration for each task is computed as:
    
      effective_duration = (base_effort_adjusted / allocation * complexity_factor * country_risk_factor)
                           + (sum of extra delays)
                           
    Extra delays include vendor_delay, compliance_delay, quality_assurance_delay,
    skill_gap_delay, material_delay, equipment_delay, contingency_time, scope_creep_delay,
    communication_delay, and env_delay.
    
    For cost, the effective cost is:
    
      effective_cost = effective_duration * allocation * wage_rate * (1 + cost_risk_factor + energy_cost_multiplier)
    """
    def __init__(self, tasks: List[Dict[str, Any]], 
                 workers: Dict[str, int],
                 worker_cost: Dict[str, int]) -> None:
        self.tasks = tasks
        self.workers = workers
        self.worker_cost = worker_cost

    def compute_schedule(self, x: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        schedule = []
        finish_times: Dict[int, float] = {}
        for task in self.tasks:
            tid = task["id"]
            resource = task["resource"]
            capacity = self.workers[resource]
            effective_max = min(task["max"], capacity)
            alloc = int(round(x[tid - 1]))
            alloc = max(task["min"], min(effective_max, alloc))
            # Basic duration calculation
            duration_base = (task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))) / alloc
            # Extra delays from various constraints
            extra_delay = (task.get("vendor_delay", 0) + task.get("compliance_delay", 0) +
                           task.get("quality_assurance_delay", 0) + task.get("skill_gap_delay", 0) +
                           task.get("material_delay", 0) + task.get("equipment_delay", 0) +
                           task.get("contingency_time", 0) + task.get("scope_creep_delay", 0) +
                           task.get("communication_delay", 0) + task.get("env_delay", 0))
            complexity = task.get("complexity_factor", 1)
            country_factor = task.get("country_risk_factor", 1)
            effective_duration = (duration_base * complexity * country_factor) + extra_delay
            # Respect dependencies
            earliest = max([finish_times[dep] for dep in task["dependencies"]]) if task["dependencies"] else 0
            start_time = find_earliest_start(earliest, effective_duration, alloc, schedule, capacity, resource)
            finish_time = start_time + effective_duration
            finish_times[tid] = finish_time
            # Tardiness if deadline is set
            deadline = task.get("deadline")
            tardiness = max(0.0, finish_time - deadline) if deadline is not None else 0.0
            schedule.append({
                "task_id": tid,
                "task_name": task["task_name"],
                "start": start_time,
                "finish": finish_time,
                "duration": effective_duration,
                "workers": alloc,
                "resource": resource,
                "deadline": deadline,
                "priority": task.get("priority", 1),
                "tardiness": tardiness
            })
        makespan = max(item["finish"] for item in schedule)
        return schedule, makespan

    def baseline_allocation(self) -> Tuple[List[Dict[str, Any]], float]:
        x = np.array([task["min"] for task in self.tasks])
        return self.compute_schedule(x)

# =============================================================================
# ------------------- Extended Objective Functions --------------------------
# =============================================================================
def objective_makespan(x: np.ndarray, model: ExtendedRCPSPModel) -> float:
    _, ms = model.compute_schedule(x)
    return ms

def objective_total_cost(x: np.ndarray, model: ExtendedRCPSPModel) -> float:
    total_cost = 0.0
    for task in model.tasks:
        tid = task["id"]
        resource = task["resource"]
        capacity = model.workers[resource]
        effective_max = min(task["max"], capacity)
        alloc = round(x[tid - 1] * 2) / 2
        alloc = max(task["min"], min(effective_max, alloc))
        duration_base = (task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))) / alloc
        extra_delay = (task.get("vendor_delay", 0) + task.get("compliance_delay", 0) +
                       task.get("quality_assurance_delay", 0) + task.get("skill_gap_delay", 0) +
                       task.get("material_delay", 0) + task.get("equipment_delay", 0) +
                       task.get("contingency_time", 0) + task.get("scope_creep_delay", 0) +
                       task.get("communication_delay", 0) + task.get("env_delay", 0))
        complexity = task.get("complexity_factor", 1)
        country_factor = task.get("country_risk_factor", 1)
        effective_duration = (duration_base * complexity * country_factor) + extra_delay
        cost_multiplier = 1 + task.get("cost_risk_factor", 0) + task.get("energy_cost_multiplier", 0)
        wage_rate = model.worker_cost[resource]
        total_cost += effective_duration * alloc * wage_rate * cost_multiplier
    return total_cost

def objective_neg_utilization(x: np.ndarray, model: ExtendedRCPSPModel) -> float:
    utils = []
    for task in model.tasks:
        tid = task["id"]
        resource = task["resource"]
        capacity = model.workers[resource]
        effective_max = min(task["max"], capacity)
        alloc = round(x[tid - 1] * 2) / 2
        alloc = max(task["min"], min(effective_max, alloc))
        utils.append(alloc / task["max"])
    return -np.mean(utils)

def objective_weighted_tardiness(x: np.ndarray, model: ExtendedRCPSPModel) -> float:
    schedule, _ = model.compute_schedule(x)
    total_weighted_tardiness = 0.0
    for task in schedule:
        if task.get("deadline") is not None:
            total_weighted_tardiness += task["tardiness"] * task.get("priority", 1)
    return total_weighted_tardiness

def objective_workload_balance(x: np.ndarray, model: ExtendedRCPSPModel) -> float:
    schedule, _ = model.compute_schedule(x)
    resource_usage = {}
    for task in schedule:
        res = task["resource"]
        usage = task["duration"] * task["workers"]
        resource_usage[res] = resource_usage.get(res, 0) + usage
    usage_values = list(resource_usage.values())
    if len(usage_values) <= 1:
        return 0.0
    return np.std(usage_values)

def multi_objective_extended(x: np.ndarray, model: ExtendedRCPSPModel) -> np.ndarray:
    return np.array([
        objective_makespan(x, model),
        objective_total_cost(x, model),
        objective_neg_utilization(x, model),
        objective_weighted_tardiness(x, model),
        objective_workload_balance(x, model)
    ])

# =============================================================================
# ----------------------- Performance Metrics -------------------------------
# =============================================================================
def approximate_hypervolume(archive: List[Tuple[np.ndarray, np.ndarray]],
                            reference_point: np.ndarray,
                            num_samples: int = 100) -> float:
    if not archive:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    mins = np.min(objs, axis=0)
    samples = np.random.uniform(low=mins, high=reference_point, size=(num_samples, len(reference_point)))
    count = sum(1 for sample in samples if any(np.all(sol <= sample) for sol in objs))
    vol = np.prod(reference_point - mins)
    return (count / num_samples) * vol

def compute_crowding_distance(archive: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
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
    return np.array_equal(entry1[0], entry2[0]) and np.array_equal(entry1[1], entry2[1])

def update_archive_with_crowding(archive: List[Tuple[np.ndarray, np.ndarray]],
                                 new_entry: Tuple[np.ndarray, np.ndarray],
                                 max_archive_size: int = 50,
                                 epsilon: float = 1e-6) -> List[Tuple[np.ndarray, np.ndarray]]:
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
    if not archive or true_pareto.size == 0:
        return None
    objs = np.array([entry[1] for entry in archive])
    distances = [np.min(np.linalg.norm(true_pareto - sol, axis=1)) for sol in objs]
    return np.mean(distances)

def compute_spread(archive: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    if len(archive) < 2:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    dists = [np.linalg.norm(objs[i] - objs[j]) for i in range(len(objs)) for j in range(i+1, len(objs))]
    return np.mean(dists)

def compute_fixed_reference(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
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
# ------------------- Random Instance & Task Generation ---------------------
# =============================================================================
def generate_random_tasks(num_tasks: int, workers: Dict[str, int]) -> List[Dict[str, Any]]:
    tasks_list = []
    resource_types = list(workers.keys())
    for i in range(1, num_tasks + 1):
        base_effort = random.randint(50, 150)
        min_alloc = random.randint(1, 3)
        max_alloc = random.randint(min_alloc + 1, 15)
        dependencies = random.sample(range(1, i), random.randint(0, min(3, i - 1))) if i > 1 else []
        resource = random.choice(resource_types)
        deadline = random.randint(20, 150)
        priority = random.choice([1, 2, 3, 4])
        tasks_list.append({
            "id": i,
            "task_name": f"Task {i}",
            "base_effort": base_effort,
            "min": min_alloc,
            "max": max_alloc,
            "dependencies": dependencies,
            "resource": resource,
            "deadline": deadline,
            "priority": priority,
            # Additional constraint attributes with random or default values:
            "vendor_delay": random.choice([0, 1]),
            "compliance_delay": random.choice([0, 2]),
            "quality_assurance_delay": random.choice([0, 2]),
            "skill_gap_delay": random.choice([0, 1]),
            "material_delay": random.choice([0, 1]),
            "equipment_delay": random.choice([0, 1]),
            "contingency_time": random.choice([0, 1, 2]),
            "scope_creep_delay": random.choice([0, 1, 2]),
            "communication_delay": random.choice([0, 1]),
            "env_delay": random.choice([0, 1]),
            "country_risk_factor": round(random.uniform(1, 1.1), 2),
            "complexity_factor": round(random.uniform(1, 1.2), 2),
            "cost_risk_factor": round(random.uniform(0, 0.05), 2),
            "energy_cost_multiplier": round(random.uniform(0, 0.1), 2)
        })
    return tasks_list

# =============================================================================
# ----------------------- Algorithm Implementations -------------------------
# =============================================================================
def MOHHO_with_progress(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    t = 0
    diversity_threshold = 0.1 * np.mean(ub - lb)
    while t < max_iter:
        E1 = 2 * math.cos((t / max_iter) * (math.pi / 2))
        for i in range(search_agents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
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
            new_x = np.clip(new_x, lb, ub)
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
                X[worst_idx, :] = np.clip(new_hawk, lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)
            else:
                X[worst_idx, :] = chaotic_map_initialization(lb, ub, dim, 1)[0]
                step_sizes[worst_idx, :] = np.ones(dim)
        best_makespan = np.min([objf(X[i, :])[0] for i in range(search_agents_no)])
        progress.append(best_makespan)
        t += 1
    return archive, progress

class PSO:
    """
    Adaptive MOPSO for RCPSP with enhancements.
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
                'w': self.w_max
            }
            self.swarm.append(particle)
        self.archive: List[Tuple[np.ndarray, np.ndarray]] = []
        self.disturbance_rate_min = disturbance_rate_min
        self.disturbance_rate_max = disturbance_rate_max
        self.jump_interval = jump_interval

    def evaluate(self, pos: np.ndarray) -> np.ndarray:
        if len(self.obj_funcs) == 1:
            return np.array([self.obj_funcs[0](pos)])
        else:
            return np.array([f(pos) for f in self.obj_funcs])

    def select_leader_hypercube(self) -> List[np.ndarray]:
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
        self.iteration += 1
        leaders = self.select_leader_hypercube()
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj = np.linalg.norm(self.evaluate(old_pos))
            r2 = random.random()
            guide = leaders[idx]
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([int(np.clip(round(new_pos[i]), self.lb[i], self.ub[i])) for i in range(self.dim)])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)
            particle['pbest'] = new_pos.copy()
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
                self.swarm[idx_to_mutate]['position'] = np.array([random.randint(int(self.lb[i]), int(self.ub[i])) for i in range(self.dim)])
                self.swarm[idx_to_mutate]['obj'] = self.evaluate(self.swarm[idx_to_mutate]['position'])
        self.update_archive()

    def update_archive(self) -> None:
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def run(self, max_iter: Optional[int] = None) -> List[float]:
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
                    colony_count: int = 10) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    dim = len(lb)
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
                h_dict[v] = np.nan_to_num(1.0 / duration, nan=0.0, posinf=0.0, neginf=0.0)
            heuristic.append(h_dict)
        colony_pheromones.append(pheromone)
        colony_heuristics.append(heuristic)
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    best_global = float('inf')
    no_improvement_count = 0
    stagnation_threshold = 10
    eps = 1e-6
    for iteration in range(max_iter):
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
                        tau = np.nan_to_num(pheromone[i][v], nan=0.0, posinf=0.0, neginf=0.0)
                        h_val = np.nan_to_num(heuristic[i][v], nan=0.0, posinf=0.0, neginf=0.0)
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
                neighbors = []
                for i in range(dim):
                    for delta in [-1, 1]:
                        neighbor = solution.copy()
                        neighbor[i] = int(np.clip(neighbor[i] + delta, lb[i], ub[i]))
                        neighbors.append(neighbor)
                def compare_objs(obj_a, obj_b):
                    if dominates(obj_a, obj_b):
                        return True
                    elif not dominates(obj_b, obj_a) and np.sum(obj_a) < np.sum(obj_b):
                        return True
                    return False
                best_neighbor = solution
                best_obj = objf(np.array(solution))
                for neighbor in neighbors:
                    n_obj = objf(np.array(neighbor))
                    if compare_objs(n_obj, best_obj):
                        best_obj = n_obj
                        best_neighbor = neighbor
                if best_neighbor == solution:
                    extended_neighbors = []
                    for i in range(dim):
                        for delta in [-2, 2]:
                            neighbor = solution.copy()
                            neighbor[i] = int(np.clip(neighbor[i] + delta, lb[i], ub[i]))
                            extended_neighbors.append(neighbor)
                    for neighbor in extended_neighbors:
                        n_obj = objf(np.array(neighbor))
                        if compare_objs(n_obj, best_obj):
                            best_obj = n_obj
                            best_neighbor = neighbor
                solution = best_neighbor
                obj_val = objf(np.array(solution))
                colony_solutions.append((solution, obj_val))
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), obj_val))
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.nan_to_num(np.array(all_values), nan=0.0, posinf=0.0, neginf=0.0)
            var_pheromone = np.var(all_values)
            if var_pheromone < 0.001:
                evap_rate_current = min(0.9, evaporation_rate * 1.5)
            else:
                evap_rate_current = evaporation_rate
            for i in range(dim):
                for v in pheromone[i]:
                    pheromone[i][v] *= (1 - evap_rate_current)
        crowding = compute_crowding_distance(archive)
        max_cd = np.max(crowding) if len(crowding) > 0 else 1.0
        if not np.isfinite(max_cd) or max_cd <= 0:
            max_cd = 1.0
        decay_factor = 1.0 - (iteration / max_iter)
        for idx, (sol, obj_val) in enumerate(archive):
            deposit = w1 * lambda3 * (crowding[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i, v in enumerate(sol):
                    colony_pheromones[colony_idx][i][v] += deposit
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.nan_to_num(np.array(all_values), nan=0.0, posinf=0.0, neginf=0.0)
            if np.var(all_values) < 0.001:
                for i in range(dim):
                    possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
                    pheromone[i] = {v: 1.0 for v in possible_values}
        merged_pheromone = []
        for i in range(dim):
            merged = {}
            possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
            for v in possible_values:
                val = sum(colony_pheromones[colony_idx][i].get(v, 0) for colony_idx in range(colony_count)) / colony_count
                merged[v] = val
            merged_pheromone.append(merged)
        for colony_idx in range(colony_count):
            colony_pheromones[colony_idx] = [merged_pheromone[i].copy() for i in range(dim)]
        current_best = min(obj_val[0] for _, obj_val in colony_solutions)
        progress.append(current_best)
        if current_best < best_global:
            best_global = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= stagnation_threshold:
            for colony_idx in range(colony_count):
                num_to_reinit = max(1, ants_per_colony // 10)
                for _ in range(num_to_reinit):
                    new_solution = [random.randint(int(lb[i]), int(ub[i])) for i in range(dim)]
                    archive = update_archive_with_crowding(archive, (np.array(new_solution), objf(np.array(new_solution))))
            no_improvement_count = 0
    return archive, progress

# =============================================================================
# ------------------------- Experiment Runner -------------------------------
# =============================================================================
def run_experiments(POP, ITER, runs: int = 1, use_random_instance: bool = False, num_tasks: int = 10
                   ) -> Tuple[Dict[str, Any], Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]], List[Dict[str, Any]]]:
    workers = {"Developer": 10, "Manager": 2, "Tester": 3}
    worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}

    if use_random_instance:
        tasks = generate_random_tasks(num_tasks, workers)
    else:
        tasks = get_extended_tasks()

    model = ExtendedRCPSPModel(tasks, workers, worker_cost)
    dim = len(model.tasks)
    lb_current = np.array([task["min"] for task in model.tasks])
    ub_current = np.array([task["max"] for task in model.tasks])
    
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

        # MOHHO with extended multi-objective function (5 objectives)
        hho_iter = ITER
        search_agents_no = POP
        archive_hho, _ = MOHHO_with_progress(lambda x: multi_objective_extended(x, model), lb_current, ub_current, dim, search_agents_no, hho_iter)
        best_ms_hho = min(archive_hho, key=lambda entry: entry[1][0])[1][0] if archive_hho else None
        results["MOHHO"]["best_makespan"].append(best_ms_hho)
        archives_all["MOHHO"].append(archive_hho)

        # PSO with 5 objectives
        objectives = [lambda x: objective_makespan(x, model),
                      lambda x: objective_total_cost(x, model),
                      lambda x: objective_neg_utilization(x, model),
                      lambda x: objective_weighted_tardiness(x, model),
                      lambda x: objective_workload_balance(x, model)]
        optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                        pop=POP, c2=1.05, w_max=0.9, w_min=0.4,
                        disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
        _ = optimizer.run(max_iter=ITER)
        archive_pso = optimizer.archive
        best_ms_pso = min(archive_pso, key=lambda entry: entry[1][0])[1][0] if archive_pso else None
        results["PSO"]["best_makespan"].append(best_ms_pso)
        archives_all["PSO"].append(archive_pso)

        # MOACO with extended multi-objective function
        ant_count = POP
        moaco_iter = ITER
        archive_moaco, _ = MOACO_improved(lambda x: multi_objective_extended(x, model), model.tasks, workers,
                                          lb_current, ub_current, ant_count, moaco_iter,
                                          alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0)
        best_ms_moaco = min(archive_moaco, key=lambda entry: entry[1][0])[1][0] if archive_moaco else None
        results["MOACO"]["best_makespan"].append(best_ms_moaco)
        archives_all["MOACO"].append(archive_moaco)

    fixed_ref = compute_fixed_reference(archives_all)
    logging.info(f"Fixed hypervolume reference point: {fixed_ref}")

    for alg in ["MOHHO", "PSO", "MOACO"]:
        for archive in archives_all[alg]:
            norm_hv = normalized_hypervolume_fixed(archive, fixed_ref)
            results[alg]["normalized_hypervolume"].append(norm_hv)
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
    sol1 = np.array([1, 2, 3])
    obj1 = np.array([10, 20, 30, 5, 2])
    sol2 = np.array([2, 3, 4])
    obj2 = np.array([12, 22, 32, 7, 3])
    archive = []
    archive = update_archive_with_crowding(archive, (sol1, obj1))
    archive = update_archive_with_crowding(archive, (sol2, obj2))
    if len(archive) != 1:
        logging.error("Unit Test Failed: Archive contains dominated solutions.")
    else:
        logging.info("Unit Test Passed: Archive update produces non-dominated set.")

    workers = {"Developer": 5, "Manager": 2, "Tester": 3}
    worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
    tasks = get_extended_tasks()
    model = ExtendedRCPSPModel(tasks, workers, worker_cost)
    x = np.array([task["min"] for task in tasks])
    schedule, ms = model.compute_schedule(x)
    if schedule and ms > 0:
        logging.info("Unit Test Passed: Extended RCPSP schedule is computed successfully.")
    else:
        logging.error("Unit Test Failed: Extended RCPSP schedule computation issue.")

# =============================================================================
# ------------------------- Main Comparison ---------------------------------
# =============================================================================
if __name__ == '__main__':
    run_unit_tests()
    
    runs = 1
    use_random_instance = False
    num_tasks = 10
    POP = 50
    ITER = 200

    if use_random_instance:
        tasks_for_exp = generate_random_tasks(num_tasks, {"Developer": 10, "Manager": 2, "Tester": 3})
    else:
        tasks_for_exp = get_extended_tasks()

    results, archives_all, base_schedules = run_experiments(POP, ITER, runs=runs, use_random_instance=use_random_instance, num_tasks=num_tasks)
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    means, stds = statistical_analysis(results)
    
    plot_convergence({alg: results[alg]["best_makespan"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Best Makespan (hours)")
    plot_convergence({alg: results[alg]["normalized_hypervolume"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Normalized Hypervolume (%)")
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Spread (Diversity)")
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    
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
    default_tasks = get_extended_tasks()
    model_for_grid = ExtendedRCPSPModel(default_tasks, workers, worker_cost)
    lb_array = np.array([task["min"] for task in default_tasks])
    ub_array = np.array([task["max"] for task in default_tasks])
    
    logging.info("Experiment complete. Results saved to 'experiment_results.json'.")

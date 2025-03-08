# objectives.py
import numpy as np
from typing import List
from rcpsp_model import RCPSPModel
import utils
from utils import round_half


def objective_makespan(x: np.ndarray, model: RCPSPModel) -> float:
    """Objective 1: Minimize project makespan."""
    schedule, ms = model.compute_schedule(x)
    return schedule, ms


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
        total_effort = utils.convertDurationtodaysCost(duration, alloc)
        wage_rate = model.worker_cost[resource_type]
        total_cost += total_effort * wage_rate
    return total_cost


def objective_neg_utilization(x: np.ndarray, model: RCPSPModel, schedule, makespan) -> float:
    """
    Objective 3: Maximize average resource utilization.
    (Negated so that all objectives are minimized.)
    
    For each task, we compute the effective (free) capacity available on its resource during its execution.
    This is done by subtracting the workers assigned to other tasks (using the same resource) that overlap 
    in time with the current task from the full capacity.
    
    Then, we compute the task's utilization as:
         utilization = allocation / free_capacity.
    
    Finally, we return the negative mean utilization over all tasks.
    """
    time_points = list(np.arange(0, makespan - 0.25, 0.25))
    resource_pool = model.workers
    ru = {}
    for key in resource_pool:
        ru[key] = 0
    total_ru = 0
    for time in time_points:
        rut = ru.copy()
        for task in schedule:
            if task["finish"] > time >= task["start"]:
                rut[task["resource"]] += task["workers"]
        total_rut = 0
        for key in rut:
            total_rut += rut[key]/resource_pool[key]
        total_ru += total_rut / len(rut.keys())
    average_ru = total_ru / len(time_points)
    return -average_ru


def objective_nega_utilization(x: np.ndarray, model: RCPSPModel) -> float:
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


def objective_negs_utilization(x: np.ndarray, model: 'RCPSPModel') -> float:
    """
    Objective 3: Maximize average resource utilization (negated for minimization).

    Instead of computing utilization per task, we sample the entire schedule in 0.25 increments.
    At each time point t (from 0 to makespan), for each resource we:
      - Sum the workers allocated to tasks that are active at time t.
      - Compute the fraction used by dividing this sum by the full capacity for that resource.
    We then average these fractions over all resources at time t and finally over all time points.
    
    This gives a single overall utilization value (between 0 and 1), which we then negate (so that
    higher utilization leads to a lower objective value).
    
    Parameters:
      x: Decision vector (allocations, etc.) used by the model.
      model: An instance of RCPSPModel, which must have attributes 'tasks' and 'workers' and a 
             method compute_schedule(x) returning (schedule, makespan).
    
    Returns:
      A scalar (negative average resource utilization).
    """
    schedule, makespan = model.compute_schedule(x)
    # Create time samples from 0 to makespan with a 0.25 time unit interval.
    time_points = np.arange(0, makespan, 0.25)
    
    resource_pool = model.workers  # Dictionary: resource -> full capacity
    total_util = 0.0  # Sum of utilization values at each time sample
    count_time = 0    # Count of time samples
    
    # Loop over each time sample.
    for t in time_points:
        utilizations_at_t = []
        # For each resource in the model:
        for resource, capacity in resource_pool.items():
            # Sum the number of workers used by tasks on this resource that are active at time t.
            used = sum(task["workers"] for task in schedule 
                       if task["resource"] == resource and task["start"] <= t < task["finish"])
            # Compute the utilization fraction for this resource (if capacity > 0).
            if capacity > 0:
                utilizations_at_t.append(used / capacity)
        # If we have data for at least one resource, compute the average utilization at time t.
        if utilizations_at_t:
            avg_util_t = np.mean(utilizations_at_t)
            total_util += avg_util_t
            count_time += 1
    
    overall_util = total_util / count_time if count_time > 0 else 0.0
    # Return the negative average utilization so that higher utilization (closer to 1) gives a lower objective.
    return -overall_util


def multi_objective(x: np.ndarray, model: RCPSPModel) -> np.ndarray:
    """
    Return the multi-objective vector for a given allocation vector x.
    """
    schedule, makespan = objective_makespan(x, model)
    return np.array([
        makespan,
        objective_total_cost(x, model),
        objective_neg_utilization(x, model, schedule, makespan)
    ])

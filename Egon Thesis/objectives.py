# objectives.py
import numpy as np
from typing import List
from rcpsp_model import RCPSPModel
import utils


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
        total_effort = utils.convertDurationtodaysCost(duration, alloc)
        wage_rate = model.worker_cost[resource_type]
        total_cost += total_effort * wage_rate
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
    """
    return np.array([
        objective_makespan(x, model),
        objective_total_cost(x, model),
        objective_neg_utilization(x, model)
    ])

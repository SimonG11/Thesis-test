import numpy as np
from rcpsp_model import RCPSPModel
from utils import round_half, compute_billable_cost


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
        alloc = round_half(x[tid - 1])
        alloc = max(task["min"], min(effective_max, alloc))
        new_effort = model.calculate_new_effort(task["base_effort"],  task["min"], task["max"], alloc)
        duration = new_effort / alloc
        wage_rate = model.worker_cost[resource_type]
        task_cost = compute_billable_cost(duration, alloc, wage_rate)
        total_cost += task_cost
    return total_cost


def objective_neg_utilization(model: RCPSPModel, schedule, makespan) -> float:
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


def multi_objective(x: np.ndarray, model: RCPSPModel) -> np.ndarray:
    """
    Return the multi-objective vector for a given allocation vector x.
    """
    schedule, makespan = objective_makespan(x, model)
    return np.array([
        makespan,
        objective_total_cost(x, model),
        objective_neg_utilization(model, schedule, makespan)
    ])

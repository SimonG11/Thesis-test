# multi_objective.py

import numpy as np
from project_schedule import tasks

def calculate_effort(x, task):
    """
    Calculate the effort for a task based on its base effort and worker allocation.
    """
    alloc = int(round(x[task["id"] - 1]))
    alloc = max(task["min"], min(task["max"], alloc))
    new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
    # new_effort = task["base_effort"] + 3 * ((alloc * ( alloc - 1))/2) # Brooks law
    # new_effort = task["base_effort"] * (1 + (1 / task[max])) ** (alloc - 1)) # Grans law
    return new_effort


def compute_schedule(x, tasks):
    """
    Compute the schedule from a decision vector x (worker allocations).
    Returns the schedule (list of task dicts) and the makespan.
    """
    schedule = []
    finish_times = {}  # task id -> finish time
    for task in tasks:
        tid = task["id"]
        # Round and clip worker allocation:
        alloc = round(x[tid - 1] * 2) / 2 # New rounding technuique to allow 0.5-steps in allocation
        alloc = max(task["min"], min(task["max"], alloc))
        # Adjusted effort and duration:
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        # new_effort = task["base_effort"] + 3 * ((alloc * ( alloc - 1))/2) # Brooks law
        # new_effort = task["base_effort"] * (1 + (1 / task[max])) ** (alloc - 1)) # Grans law
        duration = new_effort / alloc
        # Start time: maximum finish time among dependencies (or 0 if none)
        if task["dependencies"]:
            start_time = max(finish_times[dep] for dep in task["dependencies"])
        else:
            start_time = 0
        finish_time = start_time + duration
        finish_times[tid] = finish_time
        schedule.append({
            "task_id": tid,
            "task_name": task["task_name"],
            "start": start_time,
            "finish": finish_time,
            "duration": duration,
            "workers": alloc
        })
    makespan = max(item["finish"] for item in schedule)
    return schedule, makespan

def multi_objective(x):
    """
    Multi-objective evaluation function.
    Returns a vector: [makespan, total cost, -average utilization]
    """
    schedule, makespan = compute_schedule(x, tasks)
    wage_rate = 50  # Cost per man-hour
    total_cost = 0
    utilizations = []
    for task in tasks:
        tid = task["id"]
        alloc = round(x[tid - 1] * 2) / 2
        alloc = max(task["min"], min(task["max"], alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        # new_effort = task["base_effort"] + 3 * ((alloc * ( alloc - 1))/2) # Brooks law
        # new_effort = task["base_effort"] * (1 + (1 / task[max])) ** (alloc - 1)) # Grans law
        duration = new_effort / alloc
        total_cost += duration * alloc * wage_rate
        utilizations.append(alloc / task["max"])
    avg_util = np.mean(utilizations)
    # Minimizing makespan and cost, while maximizing utilization (via -avg_util)
    return np.array([makespan, total_cost, -avg_util])

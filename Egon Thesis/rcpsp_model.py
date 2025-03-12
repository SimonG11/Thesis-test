# rcpsp_model.py
from typing import List, Tuple, Dict, Any
import numpy as np
from utils import find_earliest_start, convert_hours_to_billable_days


class RCPSPModel:
    """
    A model representing the Resource-Constrained Project Scheduling Problem (RCPSP).
    """
    def __init__(self, tasks: List[Dict[str, Any]], workers: Dict[str, int], worker_cost: Dict[str, int]) -> None:
        self.tasks = tasks
        self.workers = workers
        self.worker_cost = worker_cost

    def compute_schedule(self, x: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Compute a feasible schedule using the Serial Schedule Generation Scheme (SSGS).
        """
        schedule = []
        finish_times: Dict[int, float] = {}
        for task in self.tasks:
            tid = task["id"]
            resource_type = task["resource"]
            capacity = self.workers[resource_type]
            effective_max = min(task["max"], capacity)
            alloc = round(x[tid - 1])
            alloc = max(task["min"], min(effective_max, alloc))
            if alloc == 0.5:
                new_effort = task["base_effort"] * 1.2
                duration = new_effort
            else:
                new_effort = self.calculate_new_effort(task["base_effort"],  task["min"], task["max"], alloc)
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

    def calculate_new_effort(self, base_effort, min, max, alloc):
        return base_effort * ((1 + (1.0 / max)) ** (alloc - min))
    
    def baseline_allocation(self) -> Tuple[List[Dict[str, Any]], float]:
        """
        Generate a baseline schedule by assigning the minimum required workers to all tasks.
        """
        x = np.array([task["min"] for task in self.tasks])
        return self.compute_schedule(x)

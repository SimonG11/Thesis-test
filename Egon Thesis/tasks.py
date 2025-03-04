# tasks.py
from typing import List, Dict, Any
import random

def get_default_tasks() -> List[Dict[str, Any]]:
    """
    Return a fixed list of tasks for the RCPSP.
    """
    return [
        {"id": 1, "task_name": "Requirements Gathering", "base_effort": 80, "min": 1, "max": 14, "dependencies": [], "resource": "Manager"},
        {"id": 2, "task_name": "System Design", "base_effort": 100, "min": 1, "max": 14, "dependencies": [1], "resource": "Manager"},
        {"id": 3, "task_name": "Module 1 Development", "base_effort": 150, "min": 1, "max": 14, "dependencies": [2], "resource": "Developer"},
        {"id": 4, "task_name": "Module 2 Development", "base_effort": 150, "min": 1, "max": 14, "dependencies": [2], "resource": "Developer"},
        {"id": 5, "task_name": "Integration", "base_effort": 100, "min": 1, "max": 14, "dependencies": [4], "resource": "Developer"},
        {"id": 6, "task_name": "Testing", "base_effort": 100, "min": 1, "max": 14, "dependencies": [4], "resource": "Tester"},
        {"id": 7, "task_name": "Acceptance Testing", "base_effort": 100, "min": 1, "max": 14, "dependencies": [4], "resource": "Tester"},
        {"id": 8, "task_name": "Documentation", "base_effort": 100, "min": 1, "max": 14, "dependencies": [4], "resource": "Developer"},
        {"id": 9, "task_name": "Training", "base_effort": 50, "min": 1, "max": 14, "dependencies": [7, 8], "resource": "Tester"},
        {"id": 10, "task_name": "Deployment", "base_effort": 70, "min": 2, "max": 14, "dependencies": [7, 9], "resource": "Manager"}
    ]

def generate_random_tasks(num_tasks: int, workers: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Generate a list of random, acyclic tasks for scalability testing.
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

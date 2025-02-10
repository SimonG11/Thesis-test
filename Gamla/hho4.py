#!/usr/bin/env python3
"""
Multi-Objective RCPSP – Focused on HHO Optimization

This module implements a resource-constrained project scheduling problem (RCPSP)
and uses a modified Harris Hawks Optimization (HHO) algorithm to optimize schedules.
Each candidate solution is represented as a permutation of task IDs. A serial
schedule generation scheme decodes a permutation into a feasible schedule (respecting
precedence and resource constraints) and computes three objectives:
  - Makespan (project duration) [minimize]
  - Waiting cost (total waiting time) [minimize]
  - Average resource utilization (fraction of total capacity) [maximize]

A comprehensive manual RCPSP instance with 40 tasks is provided along with a manually
optimized schedule. An improved visualization shows an overlapping comparison of the
manual (before) and HHO-optimized (after) schedules.

Detailed logging is provided (when DEBUG=True) to trace key steps.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------
# Global Debug Flag & Logger
# ------------------------
DEBUG = False
def debug_print(message):
    if DEBUG:
        print(message)

# -----------------------------------------------
# RCPSP Data Structures
# -----------------------------------------------
class Task:
    def __init__(self, id, duration, resource_requirements, predecessors=None):
        """
        Parameters:
          id: Unique task identifier.
          duration: Time units required to complete the task.
          resource_requirements: Dictionary of resource needs (e.g., {"A": 2}).
          predecessors: List of task IDs that must be completed before this task.
        """
        self.id = id
        self.duration = duration
        self.resource_requirements = resource_requirements
        self.predecessors = predecessors if predecessors is not None else []
    def __repr__(self):
        return f"Task(id={self.id}, duration={self.duration})"

class RCPSP:
    def __init__(self, tasks, resources):
        """
        Parameters:
          tasks: List of Task objects.
          resources: Dictionary with resource capacities (e.g., {"A": 5, "B": 5, "C": 5}).
        """
        self.tasks = tasks
        self.resources = resources

# -----------------------------------------------
# Randomized Data Generation (Feasible Tasks)
# -----------------------------------------------
def generate_randomized_RCPSP(num_tasks=20, resource_types=("A", "B", "C"), max_predecessors=2):
    """
    Generates a randomized RCPSP instance.
    Process:
      1. Generate resource capacities uniformly (e.g., between 3 and 5).
      2. For each task, choose a duration (2 to 15) and set each resource requirement
         as a random integer between 1 and the resource’s capacity.
      3. Randomly assign up to max_predecessors from tasks already created.
    Returns:
      An RCPSP instance.
    """
    resources = {r: random.randint(3, 5) for r in resource_types}
    debug_print(f"Randomized Data - Resources: {resources}")
    tasks = []
    for i in range(1, num_tasks+1):
        duration = random.randint(2, 15)
        req = {r: random.randint(1, resources[r]) for r in resource_types}
        if tasks:
            num_preds = random.randint(0, min(max_predecessors, len(tasks)))
            pred = [t.id for t in random.sample(tasks, num_preds)] if num_preds > 0 else []
        else:
            pred = []
        tasks.append(Task(id=i, duration=duration, resource_requirements=req, predecessors=pred))
        debug_print(f"Randomized Data - Task {i}: duration={duration}, req={req}, preds={pred}")
    return RCPSP(tasks, resources)

# -----------------------------------------------
# Decoding and Evaluation Functions
# -----------------------------------------------
def decode_solution(permutation, tasks, total_resources):
    """
    Decodes a permutation (priority list) into a feasible schedule.
    Checks both precedence and resource constraints.
    Returns:
      schedule: dict mapping task id -> (start_time, finish_time)
      makespan: Total project duration (max finish time)
      waiting_cost: Sum of waiting times (start minus earliest possible start)
      avg_util: Average resource utilization (fraction of total capacity over time)
    """
    tasks_by_id = {task.id: task for task in tasks}
    schedule = {}
    finished_time = {}
    time = 0
    running_tasks = []  # (task_id, finish_time, resource_requirements)
    resource_usage = []
    unscheduled = permutation.copy()
    
    debug_print(f"Decoding schedule for permutation: {permutation}")
    MAX_TIME = 10000
    while (unscheduled or running_tasks) and time < MAX_TIME:
        # Release tasks that have finished.
        finished = [rt for rt in running_tasks if rt[1] <= time]
        for ft in finished:
            running_tasks.remove(ft)
            debug_print(f"Time {time}: Task {ft[0]} finished.")
        current_usage = {r: 0 for r in total_resources}
        for rt in running_tasks:
            for r, amt in rt[2].items():
                current_usage[r] += amt
        available = {r: total_resources[r] - current_usage[r] for r in total_resources}
        resource_usage.append(sum(current_usage.values()))
        debug_print(f"Time {time}: Usage {current_usage}, Available {available}")
        # Schedule tasks in the order of the permutation.
        for tid in unscheduled.copy():
            task = tasks_by_id[tid]
            if all(pred in finished_time for pred in task.predecessors):
                feasible = True
                for r, amt in task.resource_requirements.items():
                    if available[r] < amt:
                        feasible = False
                        break
                if feasible:
                    start_time = time
                    finish_time_val = time + task.duration
                    schedule[tid] = (start_time, finish_time_val)
                    finished_time[tid] = finish_time_val
                    running_tasks.append((tid, finish_time_val, task.resource_requirements))
                    unscheduled.remove(tid)
                    for r, amt in task.resource_requirements.items():
                        available[r] -= amt
                    debug_print(f"Time {time}: Scheduled Task {tid} from {start_time} to {finish_time_val}.")
        time += 1
    if time >= MAX_TIME:
        debug_print("Warning: Reached MAX_TIME in decoding; some tasks unscheduled.")
    makespan = max(finished_time.values()) if finished_time else 0
    total_capacity = sum(total_resources.values())
    avg_util = np.mean(resource_usage) / total_capacity if total_capacity > 0 else 0
    waiting_cost = 0
    for tid, (start, _) in schedule.items():
        task = tasks_by_id[tid]
        earliest = max((finished_time[pred] for pred in task.predecessors), default=0)
        waiting_cost += (start - earliest)
        debug_print(f"Task {tid}: Start {start}, Earliest {earliest}, Wait {start-earliest}")
    return schedule, makespan, waiting_cost, avg_util

def evaluate_solution(permutation, tasks, resources):
    """
    Evaluates a candidate solution.
    Returns a dictionary of objectives:
      - "makespan": (minimize)
      - "waiting_cost": (minimize)
      - "avg_util": (maximize)
    """
    schedule, makespan, wait_cost, avg_util = decode_solution(permutation, tasks, resources)
    return {"makespan": makespan, "waiting_cost": wait_cost, "avg_util": avg_util}

# -----------------------------------------------
# Comprehensive Manual RCPSP Instance (40 Tasks)
# -----------------------------------------------
def generate_manual_RCPSP_comprehensive():
    """
    Constructs a manually designed RCPSP instance with 40 tasks arranged in 4 levels.
    Level 1 (Tasks 1-10): No predecessors.
    Level 2 (Tasks 11-20): Each depends on 1-3 tasks from Level 1.
    Level 3 (Tasks 21-30): Each depends on 1-3 tasks from Level 2.
    Level 4 (Tasks 31-40): Each depends on 1-3 tasks from Level 3.
    Returns:
      An RCPSP instance and a manually optimized schedule (level-by-level scheduling).
    """
    resources = {"A": 5, "B": 5, "C": 5}
    tasks = []
    # Level 1
    for i in range(1, 11):
        tasks.append(Task(id=i, duration=3, resource_requirements={"A": 1, "B": 1, "C": 1}))
    # Level 2
    for i in range(11, 21):
        preds = random.sample(range(1, 11), k=random.randint(1, 3))
        tasks.append(Task(id=i, duration=4, resource_requirements={"A": 1, "B": 1, "C": 1}, predecessors=preds))
    # Level 3
    for i in range(21, 31):
        preds = random.sample(range(11, 21), k=random.randint(1, 3))
        tasks.append(Task(id=i, duration=3, resource_requirements={"A": 1, "B": 2, "C": 1}, predecessors=preds))
    # Level 4
    for i in range(31, 41):
        preds = random.sample(range(21, 31), k=random.randint(1, 3))
        tasks.append(Task(id=i, duration=5, resource_requirements={"A": 2, "B": 1, "C": 1}, predecessors=preds))
    manual_schedule = {}
    current_time = 0
    for i in range(1, 11):
        manual_schedule[i] = (current_time, current_time+3)
        current_time += 3
    current_time += 1
    for i in range(11, 21):
        manual_schedule[i] = (current_time, current_time+4)
        current_time += 4
    current_time += 1
    for i in range(21, 31):
        manual_schedule[i] = (current_time, current_time+3)
        current_time += 3
    current_time += 1
    for i in range(31, 41):
        manual_schedule[i] = (current_time, current_time+5)
        current_time += 5
    return RCPSP(tasks, resources), manual_schedule

# -----------------------------------------------
# Helper Functions: Swap Operators & Pareto Dominance
# -----------------------------------------------
def get_swap_sequence(perm1, perm2):
    swaps = []
    p1 = perm1.copy()
    for i in range(len(p1)):
        if p1[i] != perm2[i]:
            j = p1.index(perm2[i])
            swaps.append((i, j))
            p1[i], p1[j] = p1[j], p1[i]
    return swaps

def dominates(obj1, obj2):
    better_or_equal = (obj1["makespan"] <= obj2["makespan"] and
                       obj1["waiting_cost"] <= obj2["waiting_cost"] and
                       obj1["avg_util"] >= obj2["avg_util"])
    strictly_better = (obj1["makespan"] < obj2["makespan"] or
                       obj1["waiting_cost"] < obj2["waiting_cost"] or
                       obj1["avg_util"] > obj2["avg_util"])
    return better_or_equal and strictly_better

def non_dominated_sort(population):
    nd = []
    for sol, obj in population:
        dominated_flag = False
        for other_sol, other_obj in population:
            if dominates(other_obj, obj):
                dominated_flag = True
                break
        if not dominated_flag:
            nd.append((sol, obj))
    return nd

# -----------------------------------------------
# Base Class for Optimization Algorithms
# -----------------------------------------------
class OptimizationAlgorithm:
    def __init__(self, problem, params):
        self.problem = problem
        self.params = params
    def run(self):
        raise NotImplementedError("Subclasses must implement run()")

# -----------------------------------------------
# HHO Implementation (Modified for Better Optimization)
# -----------------------------------------------
class HHOAlgorithm(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.num_agents = params.get("num_agents", 30)
        self.iterations = params.get("iterations", 200)
        self.dim = len(problem.tasks)
        self.lb = params.get("lb", 0.0)
        self.ub = params.get("ub", 1.0)
        self.X = np.random.uniform(self.lb, self.ub, (self.num_agents, self.dim))
        self.rabbit_position = None
        self.rabbit_fitness = np.inf
        self.schedule_snapshots = []
    def position_to_permutation(self, x):
        indices = np.argsort(x)
        return [self.problem.tasks[i].id for i in indices]
    def evaluate(self, x):
        perm = self.position_to_permutation(x)
        obj = evaluate_solution(perm, self.problem.tasks, self.problem.resources)
        return obj["makespan"]
    def levy(self, dim):
        beta = 1.5
        sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = 0.01 * np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        return u / (np.abs(v)**(1/beta))
    def run(self):
        X = self.X.copy()
        num_agents = self.num_agents
        lb, ub = self.lb, self.ub
        rabbit_position = None
        rabbit_fitness = np.inf
        convergence_curve = np.zeros(self.iterations)
        snapshot_interval = max(1, self.iterations // 10)
        for t in range(self.iterations):
            debug_print(f"HHO Iteration {t}")
            for i in range(num_agents):
                X[i] = np.clip(X[i], lb, ub)
                fitness = self.evaluate(X[i])
                if fitness < rabbit_fitness:
                    rabbit_fitness = fitness
                    rabbit_position = X[i].copy()
                    debug_print(f"Agent {i}: New best fitness = {fitness:.4f}")
            if t % snapshot_interval == 0 or t == self.iterations - 1:
                best_perm = self.position_to_permutation(rabbit_position)
                schedule, _, _, _ = decode_solution(best_perm, self.problem.tasks, self.problem.resources)
                self.schedule_snapshots.append((t, schedule))
            E1 = 2 * (1 - (t / self.iterations))
            for i in range(num_agents):
                E0 = 2 * random.random() - 1
                Escaping_Energy = E1 * E0
                if abs(Escaping_Energy) >= 1:
                    q = random.random()
                    rand_idx = random.randint(0, num_agents-1)
                    X_rand = X[rand_idx].copy()
                    if q < 0.5:
                        X[i] = X_rand - random.random() * abs(X_rand - 2 * random.random() * X[i])
                    else:
                        X[i] = (rabbit_position - np.mean(X, axis=0)) - random.random() * ((ub - lb) * random.random() + lb)
                else:
                    r = random.random()
                    if r >= 0.5:
                        X[i] = rabbit_position - Escaping_Energy * abs(rabbit_position - X[i])
                    else:
                        Jump_strength = 2 * (1 - random.random())
                        X1 = rabbit_position - Escaping_Energy * abs(Jump_strength * rabbit_position - X[i])
                        if self.evaluate(X1) < self.evaluate(X[i]):
                            X[i] = X1.copy()
                        else:
                            X2 = rabbit_position - Escaping_Energy * abs(Jump_strength * rabbit_position - X[i]) + np.random.randn(self.dim) * self.levy(self.dim)
                            if self.evaluate(X2) < self.evaluate(X[i]):
                                X[i] = X2.copy()
            convergence_curve[t] = rabbit_fitness
            debug_print(f"Iteration {t}: Best fitness = {rabbit_fitness:.4f}")
        self.rabbit_position = rabbit_position
        self.X = X
        best_perm = self.position_to_permutation(rabbit_position)
        best_obj = evaluate_solution(best_perm, self.problem.tasks, self.problem.resources)
        return {"permutation": best_perm, "objectives": best_obj,
                "convergence": convergence_curve, "snapshots": self.schedule_snapshots}

# -----------------------------------------------
# Visualization Functions
# -----------------------------------------------
def visualize_schedule(schedule, title="Schedule"):
    tasks_sorted = sorted(schedule.items(), key=lambda x: x[1][0])
    fig, ax = plt.subplots(figsize=(10,6))
    yticks, yticklabels = [], []
    for i, (tid, (start, finish)) in enumerate(tasks_sorted):
        ax.barh(i, finish - start, left=start, height=0.4, color="skyblue", edgecolor="black")
        yticks.append(i)
        yticklabels.append(f"Task {tid}")
        ax.text(start + (finish - start)/2, i, f"{start}-{finish}", va="center", ha="center", fontsize=9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def visualize_schedule_evolution(snapshots):
    num_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, num_snapshots, figsize=(5*num_snapshots,6))
    if num_snapshots == 1:
        axes = [axes]
    for ax, (it, sched) in zip(axes, snapshots):
        tasks_sorted = sorted(sched.items(), key=lambda x: x[1][0])
        yticks, yticklabels = [], []
        for i, (tid, (start, finish)) in enumerate(tasks_sorted):
            ax.barh(i, finish - start, left=start, height=0.4, color="lightgreen", edgecolor="black")
            yticks.append(i)
            yticklabels.append(f"Task {tid}")
            ax.text(start + (finish - start)/2, i, f"{start}-{finish}", va="center", ha="center", fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        ax.set_title(f"Iter {it}")
    plt.tight_layout()
    plt.show()

def visualize_comparison(manual_schedule, optimized_schedule, title="Schedule Comparison"):
    """
    Overlays the manual and optimized schedules for direct comparison.
    The manual schedule is plotted in semi-transparent blue and the optimized in semi-transparent red.
    """
    fig, ax = plt.subplots(figsize=(12,8))
    manual_sorted = sorted(manual_schedule.items(), key=lambda x: x[0])
    optimized_sorted = sorted(optimized_schedule.items(), key=lambda x: x[0])
    yticks, yticklabels = [], []
    offset = 0.2
    for i, ((tid1, (m_start, m_finish)), (tid2, (o_start, o_finish))) in enumerate(zip(manual_sorted, optimized_sorted)):
        ax.barh(i+offset, m_finish-m_start, left=m_start, height=0.4, color="blue", alpha=0.5, label="Manual" if i==0 else "")
        ax.barh(i-offset, o_finish-o_start, left=o_start, height=0.4, color="red", alpha=0.5, label="Optimized" if i==0 else "")
        yticks.append(i)
        yticklabels.append(f"Task {tid1}")
        ax.text(m_start+(m_finish-m_start)/2, i+offset, f"{m_start}-{m_finish}", va="center", ha="center", fontsize=8, color="black")
        ax.text(o_start+(o_finish-o_start)/2, i-offset, f"{o_start}-{o_finish}", va="center", ha="center", fontsize=8, color="black")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Main Test Driver (Focus on HHO)
# -----------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)
    
    # Generate a randomized RCPSP instance (for HHO optimization).
    problem_random = generate_randomized_RCPSP(num_tasks=20, resource_types=("A", "B", "C"), max_predecessors=2)
    random_permutation = random.sample([task.id for task in problem_random.tasks], len(problem_random.tasks))
    gen_schedule, gen_makespan, gen_wait, gen_util = decode_solution(random_permutation, problem_random.tasks, problem_random.resources)
    print("=== Randomized RCPSP Instance ===")
    print("Permutation:", random_permutation)
    print("Schedule:", gen_schedule)
    visualize_schedule(gen_schedule, title="Randomized Instance Schedule")
    print("Objectives:", evaluate_solution(random_permutation, problem_random.tasks, problem_random.resources))
    print()
    
    # Generate the comprehensive manual RCPSP instance (40 tasks).
    problem_manual, manual_schedule = generate_manual_RCPSP_comprehensive()
    print("=== Manual Optimized RCPSP Instance (40 Tasks) ===")
    print("Manual Schedule Data (first 10 tasks):")
    for tid in sorted(manual_schedule.keys())[:10]:
        print(f"Task {tid}: Start {manual_schedule[tid][0]}, Finish {manual_schedule[tid][1]}")
    visualize_schedule(manual_schedule, title="Manual Optimized Schedule")
    manual_perm = list(range(1, 41))
    print("Manual Objectives:", evaluate_solution(manual_perm, problem_manual.tasks, problem_manual.resources))
    print()
    
    # Run the HHO algorithm on the randomized instance.
    common_params_hho = {"num_agents": 30, "iterations": 200, "lb": 0.0, "ub": 1.0}
    hho = HHOAlgorithm(problem_random, common_params_hho)
    print("\nRunning HHO...")
    hho_result = hho.run()
    print("--- HHO Best Solution ---")
    print("Solution permutation:", hho_result["permutation"])
    print("Objectives:", hho_result["objectives"])
    
    # Visualize HHO schedule evolution.
    visualize_schedule_evolution(hho_result["snapshots"])
    final_iter, final_schedule = hho_result["snapshots"][-1]
    visualize_schedule(final_schedule, title=f"Final Schedule (Iter {final_iter})")
    
    # For direct comparison, decode the best HHO permutation from the randomized instance.
    best_perm_hho = hho_result["permutation"]
    opt_schedule, _, _, _ = decode_solution(best_perm_hho, problem_random.tasks, problem_random.resources)
    # Overlapping comparison of manual (40 tasks) vs. HHO optimized schedule.
    visualize_comparison(manual_schedule, opt_schedule, title="Manual (40 Tasks) vs. HHO Optimized Schedule")
    
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Objective RCPSP: Scientific Implementations of MOPSO, MOHHO, and MOACO

This module implements three metaheuristic algorithms for solving a
multi-objective resource-constrained project scheduling problem (RCPSP).

Each candidate solution is represented as a permutation of task IDs.
A serial schedule generation scheme decodes a permutation into a feasible schedule
(with resource constraints) and computes three objectives:
  1. Makespan (project duration) – to be minimized.
  2. Waiting cost (total waiting time) – to be minimized.
  3. Average resource utilization (converted to minimization by taking its negative).

For brevity and clarity, many details (e.g. parameter tuning, advanced archive maintenance)
are simplified relative to full scientific implementations.
"""

import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# RCPSP Data Structures and Scheduling (Decoding) Functionality
# -----------------------------------------------

class Task:
    def __init__(self, id, duration, resource_requirements, predecessors=None):
        """
        Parameters:
          id: Unique identifier for the task.
          duration: Time required to complete the task.
          resource_requirements: Dictionary of required resources (e.g., {"A": 2}).
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
        Represents a resource-constrained project scheduling problem.
        Parameters:
          tasks: A list of Task objects.
          resources: A dictionary with available resources (e.g., {"A": 3, "B": 3, "C": 3}).
        """
        self.tasks = tasks
        self.resources = resources

def generate_extended_RCPSP(num_tasks=20, resource_types=("A", "B", "C")):
    """
    Generates an extended RCPSP instance.
    Each task has a random duration between 2 and 10,
    resource requirements between 1 and 3 (so every task consumes some resource),
    and with a 50% chance of having one predecessor (chosen from already generated tasks).
    Available resource capacity is set low (3 per type) to force contention.
    """
    tasks = []
    for i in range(1, num_tasks + 1):
        duration = random.randint(2, 10)
        req = {r: random.randint(1, 3) for r in resource_types}
        if tasks and random.random() < 0.5:
            pred = [random.choice(tasks).id]
        else:
            pred = []
        tasks.append(Task(id=i, duration=duration, resource_requirements=req, predecessors=pred))
    resources = {r: 3 for r in resource_types}  # tight capacities
    return RCPSP(tasks, resources)

def decode_solution(permutation, tasks, total_resources):
    """
    Decodes a permutation (priority list) into a schedule using a serial scheduling scheme
    that respects precedence and resource constraints.
    
    Returns:
      - schedule: Dictionary mapping task id -> (start_time, finish_time)
      - makespan: Project finish time
      - waiting_cost: Sum of waiting times for each task (start_time minus earliest possible start)
      - avg_util: Average resource utilization over the schedule horizon.
    """
    tasks_by_id = {task.id: task for task in tasks}
    schedule = {}
    finished_time = {}
    time = 0
    running_tasks = []  # tuples: (task_id, finish_time, resource_requirements)
    resource_usage_over_time = []
    unscheduled = permutation.copy()

    while unscheduled or running_tasks:
        # Remove finished tasks
        finished = [rt for rt in running_tasks if rt[1] <= time]
        for ft in finished:
            running_tasks.remove(ft)
        # Current resource usage
        current_usage = {r: 0 for r in total_resources}
        for rt in running_tasks:
            for r, amt in rt[2].items():
                current_usage[r] += amt
        available = {r: total_resources[r] - current_usage[r] for r in total_resources}
        resource_usage_over_time.append(sum(current_usage.values()))
        # Try to schedule tasks (in the order of the permutation) if feasible
        for tid in unscheduled.copy():
            task = tasks_by_id[tid]
            # Check if all predecessors are finished
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
                    # Update available resources (for this time step only)
                    for r, amt in task.resource_requirements.items():
                        available[r] -= amt
        time += 1

    makespan = max(finished_time.values()) if finished_time else 0
    total_capacity = sum(total_resources.values())
    avg_util = np.mean(resource_usage_over_time) / total_capacity if total_capacity > 0 else 0

    # Compute waiting cost: for each task, waiting = start_time - (max(predecessors finish) or 0)
    total_wait = 0
    for tid, (start, _) in schedule.items():
        task = tasks_by_id[tid]
        if task.predecessors:
            earliest = max(finished_time[pred] for pred in task.predecessors)
        else:
            earliest = 0
        total_wait += (start - earliest)
    return schedule, makespan, total_wait, avg_util

def evaluate_solution(permutation, tasks, resources):
    """
    Evaluates a candidate solution.
    Returns a tuple of objectives: (makespan, waiting_cost, -avg_util)
    (Note: We multiply avg_util by -1 so that maximizing utilization is equivalent to minimizing -avg_util.)
    """
    schedule, makespan, wait_cost, avg_util = decode_solution(permutation, tasks, resources)
    return (makespan, wait_cost, -avg_util)

# -----------------------------------------------
# Helper Functions for Permutation Operators and Pareto Dominance
# -----------------------------------------------

def get_swap_sequence(perm1, perm2):
    """
    Returns a list of swaps (index pairs) that transform perm1 into perm2.
    """
    swaps = []
    p1 = perm1.copy()
    for i in range(len(p1)):
        if p1[i] != perm2[i]:
            j = p1.index(perm2[i])
            swaps.append((i, j))
            p1[i], p1[j] = p1[j], p1[i]
    return swaps

def dominates(obj1, obj2):
    """
    Returns True if objective vector obj1 dominates obj2 (minimization assumed).
    """
    return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))

def non_dominated_sort(population):
    """
    Given a list of (solution, objective) pairs, returns the non-dominated subset.
    """
    non_dominated = []
    for sol, obj in population:
        dominated_flag = False
        for other_sol, other_obj in population:
            if dominates(other_obj, obj):
                dominated_flag = True
                break
        if not dominated_flag:
            non_dominated.append((sol, obj))
    return non_dominated

# -----------------------------------------------
# Base Class for Optimization Algorithms
# -----------------------------------------------

class OptimizationAlgorithm:
    def __init__(self, problem, params):
        """
        Base class for an optimization algorithm.
        Parameters:
          problem: An instance of RCPSP.
          params: A dictionary of parameters.
        """
        self.problem = problem
        self.params = params

    def run(self):
        """
        Executes the algorithm.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the run() method")

# -----------------------------------------------
# MOPSO Implementation (Permutation–Based)
# -----------------------------------------------

class MOPSO(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.num_particles = params.get("num_particles", 20)
        self.iterations = params.get("iterations", 100)
        self.c1 = params.get("c1", 0.5)
        self.c2 = params.get("c2", 0.5)
        task_ids = [task.id for task in self.problem.tasks]
        # Initialize swarm as random permutations.
        self.swarm = [random.sample(task_ids, len(task_ids)) for _ in range(self.num_particles)]
        self.pbest = self.swarm.copy()
        self.pbest_obj = [evaluate_solution(sol, self.problem.tasks, self.problem.resources) for sol in self.swarm]
        self.archive = []  # list of (solution, objective) tuples

    def update_particle(self, particle, leader):
        """
        Updates a particle by computing a swap sequence toward the leader
        and applying a subset of those swaps, plus a random swap.
        """
        new_particle = particle.copy()
        swaps = get_swap_sequence(new_particle, leader)
        for swap in swaps:
            if random.random() < self.c1:
                i, j = swap
                new_particle[i], new_particle[j] = new_particle[j], new_particle[i]
        if random.random() < self.c2:
            i, j = random.sample(range(len(new_particle)), 2)
            new_particle[i], new_particle[j] = new_particle[j], new_particle[i]
        return new_particle

    def run(self):
        for it in range(self.iterations):
            for i in range(self.num_particles):
                if self.archive:
                    leader = random.choice(self.archive)[0]
                else:
                    leader = self.pbest[i]
                new_solution = self.update_particle(self.swarm[i], leader)
                obj = evaluate_solution(new_solution, self.problem.tasks, self.problem.resources)
                if dominates(obj, self.pbest_obj[i]):
                    self.pbest[i] = new_solution
                    self.pbest_obj[i] = obj
                self.swarm[i] = new_solution
            combined = [(sol, evaluate_solution(sol, self.problem.tasks, self.problem.resources)) for sol in self.swarm] + self.archive
            self.archive = non_dominated_sort(combined)
        return [{"permutation": sol, "objectives": obj} for sol, obj in self.archive]

# -----------------------------------------------
# MOHHO Implementation (Permutation–Based)
# -----------------------------------------------

class MOHHO(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.num_hawks = params.get("num_hawks", 20)
        self.iterations = params.get("iterations", 100)
        self.q = params.get("q", 0.5)
        task_ids = [task.id for task in self.problem.tasks]
        self.population = [random.sample(task_ids, len(task_ids)) for _ in range(self.num_hawks)]
        self.archive = []

    def update_hawk(self, hawk, prey):
        """
        Updates a hawk (candidate solution). With probability q, performs a random swap;
        otherwise, moves toward the prey by partially applying the swap sequence.
        """
        new_hawk = hawk.copy()
        if random.random() < self.q:
            i, j = random.sample(range(len(new_hawk)), 2)
            new_hawk[i], new_hawk[j] = new_hawk[j], new_hawk[i]
        else:
            swaps = get_swap_sequence(new_hawk, prey)
            for swap in swaps:
                if random.random() < 0.5:
                    i, j = swap
                    new_hawk[i], new_hawk[j] = new_hawk[j], new_hawk[i]
        return new_hawk

    def run(self):
        pop = self.population.copy()
        pop_eval = [(hawk, evaluate_solution(hawk, self.problem.tasks, self.problem.resources)) for hawk in pop]
        self.archive = non_dominated_sort(pop_eval)
        for it in range(self.iterations):
            if self.archive:
                prey = random.choice(self.archive)[0]
            else:
                prey = random.choice(pop)
            new_pop = []
            for hawk in pop:
                new_hawk = self.update_hawk(hawk, prey)
                new_pop.append(new_hawk)
            pop = new_pop
            pop_eval = [(hawk, evaluate_solution(hawk, self.problem.tasks, self.problem.resources)) for hawk in pop]
            combined = pop_eval + self.archive
            self.archive = non_dominated_sort(combined)
        return [{"permutation": sol, "objectives": obj} for sol, obj in self.archive]

# -----------------------------------------------
# MOACO Implementation (Pheromone-Based Construction)
# -----------------------------------------------

class MOACO(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.num_ants = params.get("num_ants", 20)
        self.iterations = params.get("iterations", 100)
        self.num_tasks = len(problem.tasks)
        self.tau0 = params.get("tau0", 1.0)
        # Initialize pheromone matrix (num_tasks x num_tasks)
        self.pheromone = [[self.tau0 for _ in range(self.num_tasks)] for _ in range(self.num_tasks)]
        # Heuristic information: inverse of task duration
        self.heuristic = [[1.0 / problem.tasks[j].duration if problem.tasks[j].duration > 0 else 0
                           for j in range(self.num_tasks)] for _ in range(self.num_tasks)]
        self.task_ids = [task.id for task in problem.tasks]
        self.index_to_id = {i: task.id for i, task in enumerate(problem.tasks)}
        self.id_to_index = {task.id: i for i, task in enumerate(problem.tasks)}
        self.rho = params.get("rho", 0.1)

    def construct_solution(self):
        available = set(self.task_ids)
        solution = []
        while available:
            feasible = []
            for tid in available:
                task = next(t for t in self.problem.tasks if t.id == tid)
                if all(pred in solution for pred in task.predecessors):
                    feasible.append(tid)
            if not feasible:
                feasible = list(available)
            if not solution:
                probs = [1.0 for _ in feasible]
            else:
                last_index = self.id_to_index[solution[-1]]
                probs = []
                for tid in feasible:
                    j = self.id_to_index[tid]
                    phero = self.pheromone[last_index][j] ** 1.0
                    heur = self.heuristic[last_index][j] ** 2.0
                    probs.append(phero * heur)
            total = sum(probs)
            probs = [p / total for p in probs]
            r = random.random()
            cumulative = 0.0
            for tid, p in zip(feasible, probs):
                cumulative += p
                if r <= cumulative:
                    next_task = tid
                    break
            solution.append(next_task)
            available.remove(next_task)
        return solution

    def update_pheromones(self, solutions):
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                self.pheromone[i][j] *= (1 - self.rho)
        for sol, obj in solutions:
            deposit = 1.0 / (sum(obj) + 1e-6)
            for k in range(len(sol) - 1):
                i = self.id_to_index[sol[k]]
                j = self.id_to_index[sol[k + 1]]
                self.pheromone[i][j] += deposit

    def run(self):
        population = []
        for it in range(self.iterations):
            ants = []
            for _ in range(self.num_ants):
                sol = self.construct_solution()
                ants.append(sol)
            ants_eval = [(sol, evaluate_solution(sol, self.problem.tasks, self.problem.resources)) for sol in ants]
            population.extend(ants_eval)
            nd = non_dominated_sort(ants_eval)
            self.update_pheromones(nd)
        archive = non_dominated_sort(population)
        return [{"permutation": sol, "objectives": obj} for sol, obj in archive]

# -----------------------------------------------
# Visualization Functions
# -----------------------------------------------

def plot_pareto_fronts(algorithms_results):
    """
    Plots scatter plots of the Pareto fronts (Makespan vs. Waiting Cost).
    The color indicates average resource utilization (converted back from -avg_util).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, results) in zip(axes, algorithms_results.items()):
        makespans = [res["objectives"][0] for res in results]
        costs = [res["objectives"][1] for res in results]
        rus = [-res["objectives"][2] for res in results]  # convert back to positive
        sc = ax.scatter(makespans, costs, c=rus, cmap="viridis", s=100, edgecolors="black")
        ax.set_title(f"{name} Pareto Front")
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Waiting Cost")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Avg Resource Utilization")
    plt.tight_layout()
    plt.show()

def plot_aggregated_metrics(aggregated_metrics):
    """
    Plots bar charts for aggregated metrics (average objectives) across algorithms.
    """
    metric_names = ["Makespan", "Waiting Cost", "Avg Resource Utilization"]
    algorithms = list(aggregated_metrics.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 6))
    for i, metric in enumerate(metric_names):
        values = []
        for algo in algorithms:
            objs = [res["objectives"] for res in aggregated_metrics[algo]]
            if metric == "Avg Resource Utilization":
                vals = [-obj[2] for obj in objs]
            elif metric == "Makespan":
                vals = [obj[0] for obj in objs]
            elif metric == "Waiting Cost":
                vals = [obj[1] for obj in objs]
            values.append(np.mean(vals))
        axes[i].bar(algorithms, values, color=["#1f77b4", "#2ca02c", "#d62728"])
        axes[i].set_title(metric)
        axes[i].set_ylabel("Average " + metric)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Main Test Driver
# -----------------------------------------------

def main():
    random.seed(42)  # For reproducibility

    # 1. Generate extended RCPSP instance.
    problem = generate_extended_RCPSP(num_tasks=20, resource_types=("A", "B", "C"))

    # 2. Define common parameter settings.
    common_params_mopso = {
        "num_particles": 30,
        "iterations": 100,
        "c1": 0.6,
        "c2": 0.3
    }
    common_params_mohho = {
        "num_hawks": 30,
        "iterations": 100,
        "q": 0.5
    }
    common_params_moaco = {
        "num_ants": 30,
        "iterations": 100,
        "tau0": 1.0,
        "rho": 0.1
    }

    # 3. Instantiate algorithm objects.
    algorithms = {
        "MOPSO": MOPSO(problem, common_params_mopso),
        "MOHHO": MOHHO(problem, common_params_mohho),
        "MOACO": MOACO(problem, common_params_moaco)
    }

    # 4. Run each algorithm.
    results = {}
    for name, algo in algorithms.items():
        print(f"\nRunning {name} ...")
        pareto_front = algo.run()
        results[name] = pareto_front
        print(f"\n--- {name} Pareto Front (Representative Solutions) ---")
        for res in pareto_front:
            makespan, cost, neg_util = res["objectives"]
            avg_util = -neg_util
            print(f"Solution: {res['permutation']}\n  Makespan: {makespan}, Waiting Cost: {cost}, Avg Utilization: {avg_util}")
        print("-----------------------------------------------------")
    
    # 5. Visualize the Pareto fronts.
    plot_pareto_fronts(results)
    
    # 6. Plot aggregated metrics.
    agg_metrics = {name: res for name, res in results.items()}
    plot_aggregated_metrics(agg_metrics)

if __name__ == "__main__":
    main()

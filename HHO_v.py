#!/usr/bin/env python3
"""
Multi-Objective RCPSP: Scientific Implementations of MOPSO, MOHHO, MOACO, and HHO (adapted)

This module implements four metaheuristic algorithms for solving a
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
import math

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
# HHO Implementation (Continuous Domain with Permutation Decoding)
# Adapted from research code (Harris Hawks Optimization)
# Also stores schedule snapshots for visualization.
# -----------------------------------------------

class HHOAlgorithm(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        # For permutation problems, we work in a continuous space [lb, ub]^dim and then decode via sorting.
        self.num_agents = params.get("num_agents", 20)
        self.iterations = params.get("iterations", 100)
        self.dim = len(problem.tasks)  # dimension equals number of tasks
        self.lb = params.get("lb", 0.0)
        self.ub = params.get("ub", 1.0)
        # Initialize continuous positions for each hawk.
        self.X = np.random.uniform(self.lb, self.ub, (self.num_agents, self.dim))
        self.rabbit_position = None
        self.rabbit_fitness = np.inf
        # List to store schedule snapshots as (iteration, schedule)
        self.schedule_snapshots = []

    def position_to_permutation(self, x):
        """
        Convert a continuous vector into a permutation by ranking.
        """
        indices = np.argsort(x)  # indices sorted in increasing order
        # Map indices to task IDs (assuming problem.tasks order corresponds to task IDs)
        permutation = [self.problem.tasks[i].id for i in indices]
        return permutation

    def evaluate(self, x):
        """
        Decode x into a permutation and return the makespan.
        """
        perm = self.position_to_permutation(x)
        obj = evaluate_solution(perm, self.problem.tasks, self.problem.resources)
        return obj[0]  # use makespan as the fitness

    def levy(self, dim):
        beta = 1.5
        sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)) )**(1/beta)
        u = 0.01 * np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / (np.abs(v)**(1/beta))
        return step

    def run(self):
        X = self.X.copy()
        num_agents = self.num_agents
        dim = self.dim
        lb = self.lb
        ub = self.ub
        rabbit_position = None
        rabbit_fitness = np.inf
        convergence_curve = np.zeros(self.iterations)
        # Define snapshot interval (e.g., every 20% of iterations)
        snapshot_interval = max(1, self.iterations // 5)
        for t in range(self.iterations):
            # Evaluate fitness of each hawk (after clipping to bounds)
            for i in range(num_agents):
                X[i] = np.clip(X[i], lb, ub)
                fitness = self.evaluate(X[i])
                if fitness < rabbit_fitness:
                    rabbit_fitness = fitness
                    rabbit_position = X[i].copy()
            # Store schedule snapshot at defined intervals
            if t % snapshot_interval == 0 or t == self.iterations - 1:
                best_perm = self.position_to_permutation(rabbit_position)
                schedule, _, _, _ = decode_solution(best_perm, self.problem.tasks, self.problem.resources)
                self.schedule_snapshots.append((t, schedule))
            E1 = 2 * (1 - (t / self.iterations))
            for i in range(num_agents):
                E0 = 2 * random.random() - 1
                Escaping_Energy = E1 * E0
                if abs(Escaping_Energy) >= 1:
                    # Exploration phase
                    q = random.random()
                    rand_idx = random.randint(0, num_agents-1)
                    X_rand = X[rand_idx].copy()
                    if q < 0.5:
                        X[i] = X_rand - random.random() * np.abs(X_rand - 2 * random.random() * X[i])
                    else:
                        X[i] = (rabbit_position - np.mean(X, axis=0)) - random.random() * ((ub - lb)*random.random() + lb)
                else:
                    # Exploitation phase
                    r = random.random()
                    if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                        X[i] = rabbit_position - Escaping_Energy * np.abs(rabbit_position - X[i])
                    if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                        Jump_strength = 2 * (1 - random.random())
                        X[i] = (rabbit_position - X[i]) - Escaping_Energy * np.abs(Jump_strength * rabbit_position - X[i])
                    if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                        Jump_strength = 2 * (1 - random.random())
                        X1 = rabbit_position - Escaping_Energy * np.abs(Jump_strength * rabbit_position - X[i])
                        if self.evaluate(X1) < self.evaluate(X[i]):
                            X[i] = X1.copy()
                        else:
                            X2 = rabbit_position - Escaping_Energy * np.abs(Jump_strength * rabbit_position - X[i]) + np.random.randn(dim) * self.levy(dim)
                            if self.evaluate(X2) < self.evaluate(X[i]):
                                X[i] = X2.copy()
                    if r < 0.5 and abs(Escaping_Energy) < 0.5:
                        Jump_strength = 2 * (1 - random.random())
                        X1 = rabbit_position - Escaping_Energy * np.abs(Jump_strength * rabbit_position - np.mean(X, axis=0))
                        if self.evaluate(X1) < self.evaluate(X[i]):
                            X[i] = X1.copy()
                        else:
                            X2 = rabbit_position - Escaping_Energy * np.abs(Jump_strength * rabbit_position - np.mean(X, axis=0)) + np.random.randn(dim) * self.levy(dim)
                            if self.evaluate(X2) < self.evaluate(X[i]):
                                X[i] = X2.copy()
            convergence_curve[t] = rabbit_fitness
            print(f"Iteration {t}: best fitness = {rabbit_fitness}")
        self.rabbit_position = rabbit_position
        self.X = X
        best_perm = self.position_to_permutation(rabbit_position)
        best_obj = evaluate_solution(best_perm, self.problem.tasks, self.problem.resources)
        return {"permutation": best_perm, "objectives": best_obj, "convergence": convergence_curve, "snapshots": self.schedule_snapshots}

# -----------------------------------------------
# Visualization Functions
# -----------------------------------------------

def plot_pareto_fronts(algorithms_results):
    """
    Plots scatter plots of the Pareto fronts (Makespan vs. Waiting Cost).
    The color indicates average resource utilization (converted back from -avg_util).
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
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
        axes[i].bar(algorithms, values, color=["#1f77b4", "#2ca02c", "#d62728", "#9467bd"])
        axes[i].set_title(metric)
        axes[i].set_ylabel("Average " + metric)
    plt.tight_layout()
    plt.show()

def visualize_schedule(schedule, title="Schedule"):
    """
    Visualizes a schedule as a Gantt chart.
    'schedule' is a dict mapping task ID to (start, finish).
    """
    # Sort tasks by start time
    tasks_sorted = sorted(schedule.items(), key=lambda x: x[1][0])
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks = []
    yticklabels = []
    for i, (tid, (start, finish)) in enumerate(tasks_sorted):
        ax.barh(i, finish - start, left=start, height=0.4, color="skyblue", edgecolor="black")
        yticks.append(i)
        yticklabels.append(f"Task {tid}")
        ax.text(start + (finish - start)/2, i, f"{start}-{finish}", 
                va="center", ha="center", color="black", fontsize=9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_schedule_evolution(snapshots):
    """
    Plots schedule snapshots collected during optimization.
    'snapshots' is a list of tuples: (iteration, schedule).
    Each schedule is visualized as a Gantt chart.
    """
    num_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, num_snapshots, figsize=(5 * num_snapshots, 6))
    if num_snapshots == 1:
        axes = [axes]
    for ax, (it, sched) in zip(axes, snapshots):
        # Sort tasks by start time
        tasks_sorted = sorted(sched.items(), key=lambda x: x[1][0])
        yticks = []
        yticklabels = []
        for i, (tid, (start, finish)) in enumerate(tasks_sorted):
            ax.barh(i, finish - start, left=start, height=0.4, color="lightgreen", edgecolor="black")
            yticks.append(i)
            yticklabels.append(f"Task {tid}")
            ax.text(start + (finish - start)/2, i, f"{start}-{finish}", 
                    va="center", ha="center", color="black", fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        ax.set_title(f"Iter {it}")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Main Test Driver
# -----------------------------------------------

def main():
    random.seed(42)  # For reproducibility
    np.random.seed(42)

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
    common_params_hho = {
        "num_agents": 30,
        "iterations": 100,
        "lb": 0.0,
        "ub": 1.0
    }

    # 3. Instantiate algorithm objects.
    algorithms = {
        #"MOPSO": MOPSO(problem, common_params_mopso),
        #"MOHHO": MOHHO(problem, common_params_mohho),
        #"MOACO": MOACO(problem, common_params_moaco),
        "HHO": HHOAlgorithm(problem, common_params_hho)
    }

    # 4. Run each algorithm.
    results = {}
    snapshots = None  # For schedule evolution visualization from HHO
    for name, algo in algorithms.items():
        print(f"\nRunning {name} ...")
        result = algo.run()
        # For consistency with the other algorithms, wrap the result in a list.
        if name == "HHO":
            # HHO returns a dict with "permutation", "objectives", "convergence", and "snapshots"
            results[name] = [result]
            snapshots = result.get("snapshots", [])
            print(f"\n--- {name} Best Solution ---")
            makespan, cost, neg_util = result["objectives"]
            avg_util = -neg_util
            print(f"Solution: {result['permutation']}\n  Makespan: {makespan}, Waiting Cost: {cost}, Avg Utilization: {avg_util}")
        else:
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
    
    # 7. If HHO was run, visualize the schedule evolution.
    if snapshots:
        print("\nVisualizing schedule evolution for HHO:")
        plot_schedule_evolution(snapshots)
        # Optionally, also show the final best schedule in detail.
        final_iter, final_schedule = snapshots[-1]
        visualize_schedule(final_schedule, title=f"Final Schedule (Iter {final_iter})")

if __name__ == "__main__":
    main()

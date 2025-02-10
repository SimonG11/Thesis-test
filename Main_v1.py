#!/usr/bin/env python3
"""
Multi-Objective RCPSP: Scientific Implementations of MOPSO, MOHHO, MOACO, HHO, and PSO (adapted)

This module implements several metaheuristic algorithms for solving a
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
# Manual Schedule Example
# -----------------------------------------------
# In addition to the schedule generated by our RCPSP decoding procedure,
# we also illustrate a "manual" schedule.
#
# In this example, we manually define a schedule as a Python dictionary.
# Each activity is represented as a key (the task ID) with a dictionary of its attributes.
#
# The manual schedule below is defined for illustration:
#
#   - "start": time when the task begins.
#   - "finish": time when the task completes.
#   - "duration": calculated as finish - start.
#   - "resources": a dictionary of resource usage.
#   - "predecessors": a list of task IDs that must be completed before the task starts.
#
# This manual definition allows us to see how the schedule data is structured and
# demonstrates our ability to manipulate or adjust the schedule “by hand.”
#
manual_schedule = {
    1: {"start": 0, "finish": 4, "duration": 4, "resources": {"A": 2, "B": 1}, "predecessors": []},
    2: {"start": 4, "finish": 9, "duration": 5, "resources": {"A": 1, "B": 2}, "predecessors": [1]},
    3: {"start": 2, "finish": 6, "duration": 4, "resources": {"B": 1, "C": 2}, "predecessors": []},
    4: {"start": 9, "finish": 14, "duration": 5, "resources": {"A": 2, "C": 1}, "predecessors": [2, 3]},
    5: {"start": 14, "finish": 18, "duration": 4, "resources": {"B": 1, "C": 1}, "predecessors": [4]}
}

# -----------------------------------------------
# Helper Functions for Permutation Operators and Pareto Dominance
# (unchanged from before)
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
    return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))

def non_dominated_sort(population):
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
        self.problem = problem
        self.params = params

    def run(self):
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
        self.swarm = [random.sample(task_ids, len(task_ids)) for _ in range(self.num_particles)]
        self.pbest = self.swarm.copy()
        self.pbest_obj = [evaluate_solution(sol, self.problem.tasks, self.problem.resources) for sol in self.swarm]
        self.archive = []

    def update_particle(self, particle, leader):
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
                leader = random.choice(self.archive)[0] if self.archive else self.pbest[i]
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
            prey = random.choice(self.archive)[0] if self.archive else random.choice(pop)
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
        self.pheromone = [[self.tau0 for _ in range(self.num_tasks)] for _ in range(self.num_tasks)]
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
# Also stores schedule snapshots for visualization.
# -----------------------------------------------

class HHOAlgorithm(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.num_agents = params.get("num_agents", 20)
        self.iterations = params.get("iterations", 100)
        self.dim = len(problem.tasks)
        self.lb = params.get("lb", 0.0)
        self.ub = params.get("ub", 1.0)
        self.X = np.random.uniform(self.lb, self.ub, (self.num_agents, self.dim))
        self.rabbit_position = None
        self.rabbit_fitness = np.inf
        self.schedule_snapshots = []  # (iteration, schedule)

    def position_to_permutation(self, x):
        indices = np.argsort(x)
        permutation = [self.problem.tasks[i].id for i in indices]
        return permutation

    def evaluate(self, x):
        perm = self.position_to_permutation(x)
        obj = evaluate_solution(perm, self.problem.tasks, self.problem.resources)
        return obj[0]

    def levy(self, dim):
        beta = 1.5
        sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
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
        snapshot_interval = max(1, self.iterations // 5)
        for t in range(self.iterations):
            for i in range(num_agents):
                X[i] = np.clip(X[i], lb, ub)
                fitness = self.evaluate(X[i])
                if fitness < rabbit_fitness:
                    rabbit_fitness = fitness
                    rabbit_position = X[i].copy()
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
                        X[i] = X_rand - random.random() * np.abs(X_rand - 2 * random.random() * X[i])
                    else:
                        X[i] = (rabbit_position - np.mean(X, axis=0)) - random.random() * ((ub - lb)*random.random() + lb)
                else:
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
# PSO Implementation (Continuous Domain with Permutation Decoding)
# Adapted from the provided PSO code.
# -----------------------------------------------

class PSOAlgorithm(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        # The search dimension equals the number of tasks.
        self.dim = len(problem.tasks)
        self.pop = params.get("pop", 20)
        self.v_max = params.get("v_max", 0.1)
        self.personal_c = params.get("personal_c", 2.0)
        self.social_c = params.get("social_c", 2.0)
        self.inertia_weight = params.get("inertia_weight", 0.7)
        self.convergence_threshold = params.get("convergence", 0.001)
        self.max_iter = params.get("max_iter", 100)
        self.lb = params.get("lb", 0.0)
        self.ub = params.get("ub", 1.0)
        self.particles = []
        self.global_best_pos = None
        self.global_best_cost = np.inf
        # Initialize particles
        for _ in range(self.pop):
            pos = np.random.uniform(self.lb, self.ub, self.dim)
            vel = np.random.uniform(-self.v_max, self.v_max, self.dim)
            cost = self.evaluate(pos)
            particle = {"pos": pos, "vel": vel, "best_pos": pos.copy(), "cost": cost, "best_cost": cost}
            self.particles.append(particle)
            if cost < self.global_best_cost:
                self.global_best_cost = cost
                self.global_best_pos = pos.copy()

    def position_to_permutation(self, x):
        indices = np.argsort(x)
        permutation = [self.problem.tasks[i].id for i in indices]
        return permutation

    def evaluate(self, x):
        perm = self.position_to_permutation(x)
        obj = evaluate_solution(perm, self.problem.tasks, self.problem.resources)
        return obj[0]  # we use makespan as the cost

    def run(self):
        convergence_curve = []
        iter = 0
        while iter < self.max_iter:
            for particle in self.particles:
                for i in range(self.dim):
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    particle["vel"][i] = (self.inertia_weight * particle["vel"][i] +
                                            self.personal_c * r1 * (particle["best_pos"][i] - particle["pos"][i]) +
                                            self.social_c * r2 * (self.global_best_pos[i] - particle["pos"][i]))
                    # Enforce velocity limits
                    if particle["vel"][i] > self.v_max:
                        particle["vel"][i] = self.v_max
                    elif particle["vel"][i] < -self.v_max:
                        particle["vel"][i] = -self.v_max
                # Update particle's position and clip to bounds
                particle["pos"] = particle["pos"] + particle["vel"]
                particle["pos"] = np.clip(particle["pos"], self.lb, self.ub)
                cost = self.evaluate(particle["pos"])
                particle["cost"] = cost
                # Update personal best
                if cost < particle["best_cost"]:
                    particle["best_cost"] = cost
                    particle["best_pos"] = particle["pos"].copy()
                # Update global best
                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_pos = particle["pos"].copy()
            convergence_curve.append(self.global_best_cost)
            if abs(self.global_best_cost) < self.convergence_threshold:
                print("PSO has met convergence criteria after", iter, "iterations.")
                break
            iter += 1
        best_perm = self.position_to_permutation(self.global_best_pos)
        return {"permutation": best_perm, "objectives": (self.global_best_cost, 0, 0), "convergence": convergence_curve}

# -----------------------------------------------
# Visualization Functions
# -----------------------------------------------

def plot_pareto_fronts(algorithms_results):
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for ax, (name, results) in zip(axes, algorithms_results.items()):
        makespans = [res["objectives"][0] for res in results]
        costs = [res["objectives"][1] for res in results]
        rus = [-res["objectives"][2] for res in results]
        sc = ax.scatter(makespans, costs, c=rus, cmap="viridis", s=100, edgecolors="black")
        ax.set_title(f"{name} Pareto Front")
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Waiting Cost")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Avg Resource Utilization")
    plt.tight_layout()
    plt.show()

def plot_aggregated_metrics(aggregated_metrics):
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
        axes[i].bar(algorithms, values, color=["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"])
        axes[i].set_title(metric)
        axes[i].set_ylabel("Average " + metric)
    plt.tight_layout()
    plt.show()

def visualize_schedule(schedule, title="Schedule"):
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
    num_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, num_snapshots, figsize=(5 * num_snapshots, 6))
    if num_snapshots == 1:
        axes = [axes]
    for ax, (it, sched) in zip(axes, snapshots):
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
    random.seed(42)
    np.random.seed(42)

    # 1. Generate extended RCPSP instance.
    problem = generate_extended_RCPSP(num_tasks=20, resource_types=("A", "B", "C"))

    # 2. Display a "Generated Randomized Schedule"
    random_permutation = random.sample([task.id for task in problem.tasks], len(problem.tasks))
    generated_schedule, gen_makespan, gen_wait, gen_util = decode_solution(random_permutation, problem.tasks, problem.resources)
    print("=== Generated Randomized Schedule ===")
    print("Permutation:", random_permutation)
    print("Schedule:", generated_schedule)
    visualize_schedule(generated_schedule, title="Generated Randomized Schedule")

    # 3. Display the Manual Schedule Example.
    print("\n=== Manual Schedule Example ===")
    print("Manual Schedule Data:")
    for tid, data in manual_schedule.items():
        print(f"Task {tid}: {data}")
    visualize_schedule({tid: (data["start"], data["finish"]) for tid, data in manual_schedule.items()},
                         title="Manual Schedule Example")

    # 4. Define common parameter settings.
    common_params_mopso = {"num_particles": 30, "iterations": 100, "c1": 0.6, "c2": 0.3}
    common_params_mohho = {"num_hawks": 30, "iterations": 100, "q": 0.5}
    common_params_moaco = {"num_ants": 30, "iterations": 100, "tau0": 1.0, "rho": 0.1}
    common_params_hho = {"num_agents": 30, "iterations": 100, "lb": 0.0, "ub": 1.0}
    common_params_pso = {"pop": 20, "v_max": 0.1, "personal_c": 2.0, "social_c": 2.0,
                         "inertia_weight": 0.7, "convergence": 0.001, "max_iter": 100,
                         "lb": 0.0, "ub": 1.0}

    # 5. Instantiate algorithm objects.
    algorithms = {
        #"MOPSO": MOPSO(problem, common_params_mopso),
        #"MOHHO": MOHHO(problem, common_params_mohho),
        #"MOACO": MOACO(problem, common_params_moaco),
        "HHO": HHOAlgorithm(problem, common_params_hho),
        "PSO": PSOAlgorithm(problem, common_params_pso)
    }

    results = {}
    snapshots = None  # For HHO schedule evolution
    for name, algo in algorithms.items():
        print(f"\nRunning {name} ...")
        result = algo.run()
        if name == "HHO":
            results[name] = [result]
            snapshots = result.get("snapshots", [])
            makespan, cost, neg_util = result["objectives"]
            avg_util = -neg_util
            print(f"--- {name} Best Solution ---")
            print(f"Solution: {result['permutation']}")
            print(f"Makespan: {makespan}, Waiting Cost: {cost}, Avg Utilization: {avg_util}")
        elif name == "PSO":
            results[name] = [result]
            makespan = result["objectives"][0]
            print(f"--- {name} Best Solution ---")
            print(f"Solution: {result['permutation']}")
            print(f"Makespan: {makespan}")
        else:
            pareto_front = algo.run()
            results[name] = pareto_front
            print(f"--- {name} Pareto Front (Representative Solutions) ---")
            for res in pareto_front:
                makespan, cost, neg_util = res["objectives"]
                avg_util = -neg_util
                print(f"Solution: {res['permutation']}")
                print(f"Makespan: {makespan}, Waiting Cost: {cost}, Avg Utilization: {avg_util}")
        print("-----------------------------------------------------")
    
    # 6. Visualize the Pareto fronts.
    plot_pareto_fronts(results)
    
    # 7. Plot aggregated metrics.
    agg_metrics = {name: res for name, res in results.items()}
    plot_aggregated_metrics(agg_metrics)
    
    # 8. If HHO was run, visualize the schedule evolution.
    if snapshots:
        print("\nVisualizing schedule evolution for HHO:")
        plot_schedule_evolution(snapshots)
        final_iter, final_schedule = snapshots[-1]
        visualize_schedule(final_schedule, title=f"Final Schedule (Iter {final_iter})")
    
    # 9. (Optional) Plot PSO convergence curve.
    if "PSO" in results:
        pso_result = results["PSO"][0]
        plt.figure("PSO Convergence")
        plt.plot(pso_result["convergence"], marker='o', color='r')
        plt.title("PSO Convergence Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Best Makespan")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()

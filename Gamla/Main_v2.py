#!/usr/bin/env python3
"""
Multi-Objective RCPSP:
Scientific Implementations of MOPSO, MOHHO, MOACO, HHO, and PSO

This module implements several metaheuristic algorithms for solving a
multi-objective resource-constrained project scheduling problem (RCPSP).

Each candidate solution is represented as a permutation of task IDs.
A serial schedule generation scheme decodes a permutation into a feasible schedule
(with resource constraints) and computes three objectives:
  - Makespan (project duration) [to be minimized]
  - Waiting cost (total waiting time) [to be minimized]
  - Average resource utilization (expressed as a positive fraction of total capacity, to be maximized]
  
In multi-objective mode, no single “correct” answer exists – rather, a set of
non-dominated solutions (Pareto front) is returned.
  
Detailed logging is provided (when DEBUG=True) to trace key steps.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------
# Global Debug Flag & Logger
# ------------------------
DEBUG = True

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
          id: Unique identifier for the task.
          duration: Time required to complete the task.
          resource_requirements: Dictionary of required resources, e.g. {"A": 2}.
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
          resources: A dictionary with available resources, e.g. {"A": 3, "B": 3, "C": 3}.
        """
        self.tasks = tasks
        self.resources = resources

# -----------------------------------------------
# Enhanced Randomized Data Generation (Feasible Tasks)
# -----------------------------------------------

def generate_randomized_RCPSP(num_tasks=20, resource_types=("A", "B", "C"), max_predecessors=2):
    """
    Generates a randomized RCPSP instance.
    
    Process:
      1. Generate available resource capacities first (random between 3 and 5).
      2. For each task:
         - Choose a duration randomly (e.g., 2 to 15).
         - For each resource, choose a requirement uniformly between 1 and the capacity.
         - Randomly assign up to max_predecessors from tasks already generated.
    
    Returns:
      An RCPSP instance.
    """
    # Generate resource capacities first.
    resources = {r: random.randint(3, 5) for r in resource_types}
    debug_print(f"Randomized Data - Resources: {resources}")
    
    tasks = []
    for i in range(1, num_tasks + 1):
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
    Decodes a permutation (priority list) into a schedule using a serial scheduling scheme.
    
    The scheduler considers:
      - Precedence constraints: A task can only be scheduled if all its predecessors are finished.
      - Resource constraints: A task is scheduled only if available resource units are sufficient.
    
    Returns:
      schedule: Dictionary mapping task id to (start_time, finish_time)
      makespan: Total project duration (max finish time)
      waiting_cost: Sum of waiting times for tasks (start time minus earliest possible start)
      avg_util: Average resource utilization as a fraction of total capacity over time
    """
    tasks_by_id = {task.id: task for task in tasks}
    schedule = {}
    finished_time = {}
    time = 0
    running_tasks = []  # (task_id, finish_time, resource_requirements)
    resource_usage_over_time = []
    unscheduled = permutation.copy()
    
    debug_print(f"Decoding schedule for permutation: {permutation}")
    
    MAX_TIME = 10000  # safety cutoff to avoid infinite loops
    
    while (unscheduled or running_tasks) and time < MAX_TIME:
        # Remove finished tasks
        finished = [rt for rt in running_tasks if rt[1] <= time]
        for ft in finished:
            running_tasks.remove(ft)
            debug_print(f"Time {time}: Task {ft[0]} finished.")
        # Compute current resource usage
        current_usage = {r: 0 for r in total_resources}
        for rt in running_tasks:
            for r, amt in rt[2].items():
                current_usage[r] += amt
        available = {r: total_resources[r] - current_usage[r] for r in total_resources}
        resource_usage_over_time.append(sum(current_usage.values()))
        debug_print(f"Time {time}: Usage {current_usage}, Available {available}")
        
        # Try to schedule tasks (in order of the permutation)
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
        debug_print("Warning: Reached MAX_TIME in decoding; some tasks may remain unscheduled.")
    makespan = max(finished_time.values()) if finished_time else 0
    total_capacity = sum(total_resources.values())
    avg_util = np.mean(resource_usage_over_time) / total_capacity if total_capacity > 0 else 0
    
    total_wait = 0
    for tid, (start, _) in schedule.items():
        task = tasks_by_id[tid]
        earliest = max((finished_time[pred] for pred in task.predecessors), default=0)
        wait_time = start - earliest
        total_wait += wait_time
        debug_print(f"Task {tid}: Start {start}, Earliest {earliest}, Wait {wait_time}")
    
    return schedule, makespan, total_wait, avg_util

def evaluate_solution(permutation, tasks, resources):
    """
    Evaluates a candidate solution.
    
    Returns a dictionary of objectives:
      - "makespan": to be minimized
      - "waiting_cost": to be minimized
      - "avg_util": to be maximized
    """
    schedule, makespan, wait_cost, avg_util = decode_solution(permutation, tasks, resources)
    return {
        "makespan": makespan,
        "waiting_cost": wait_cost,
        "avg_util": avg_util
    }

def combine_objectives(obj_dict, weights):
    """
    Optionally combine multiple objectives into a single scalar value.
    (Not used directly in the multi-objective algorithms.)
    """
    return (weights.get("makespan", 1) * obj_dict["makespan"] +
            weights.get("waiting_cost", 1) * obj_dict["waiting_cost"] +
            weights.get("avg_util", 1) * obj_dict["avg_util"])

# -----------------------------------------------
# Manual Optimized Data Set
# -----------------------------------------------

def generate_manual_RCPSP():
    """
    Constructs a small, manually optimized RCPSP instance.
    """
    tasks = [
        Task(id=1, duration=3, resource_requirements={"A": 1, "B": 1}, predecessors=[]),
        Task(id=2, duration=4, resource_requirements={"A": 1, "B": 2}, predecessors=[1]),
        Task(id=3, duration=2, resource_requirements={"B": 1, "C": 1}, predecessors=[1]),
        Task(id=4, duration=3, resource_requirements={"A": 2, "C": 1}, predecessors=[2, 3]),
        Task(id=5, duration=2, resource_requirements={"B": 1, "C": 1}, predecessors=[4])
    ]
    resources = {"A": 3, "B": 3, "C": 3}
    manual_optimized_schedule = {
        1: (0, 3),
        2: (3, 7),
        3: (3, 5),
        4: (7, 10),
        5: (10, 12)
    }
    return RCPSP(tasks, resources), manual_optimized_schedule

# -----------------------------------------------
# Helper Functions: Swap Operators and Pareto Dominance
# -----------------------------------------------

def get_swap_sequence(perm1, perm2):
    """
    Computes a list of swap operations to transform perm1 into perm2.
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
    Returns True if obj1 dominates obj2.
    (Makespan and waiting_cost are minimized; avg_util is maximized.)
    """
    better_or_equal = (obj1["makespan"] <= obj2["makespan"] and
                       obj1["waiting_cost"] <= obj2["waiting_cost"] and
                       obj1["avg_util"] >= obj2["avg_util"])
    strictly_better = (obj1["makespan"] < obj2["makespan"] or
                       obj1["waiting_cost"] < obj2["waiting_cost"] or
                       obj1["avg_util"] > obj2["avg_util"])
    return better_or_equal and strictly_better

def non_dominated_sort(population):
    """
    Returns the non-dominated solutions from the population.
    Each element is a tuple: (solution, objective_dict)
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
        self.pbest_obj = [evaluate_solution(sol, self.problem.tasks, self.problem.resources)
                          for sol in self.swarm]
        self.archive = []

    def update_particle(self, particle, leader):
        new_particle = particle.copy()
        swaps = get_swap_sequence(new_particle, leader)
        for swap in swaps:
            if random.random() < self.c1:
                i, j = swap
                new_particle[i], new_particle[j] = new_particle[j], new_particle[i]
                debug_print(f"MOPSO: Swap applied between indices {i} and {j}.")
        if random.random() < self.c2:
            i, j = random.sample(range(len(new_particle)), 2)
            new_particle[i], new_particle[j] = new_particle[j], new_particle[i]
            debug_print(f"MOPSO: Additional random swap between indices {i} and {j}.")
        return new_particle

    def run(self):
        for it in range(self.iterations):
            debug_print(f"MOPSO Iteration {it}")
            for i in range(self.num_particles):
                leader = random.choice(self.archive)[0] if self.archive else self.pbest[i]
                new_solution = self.update_particle(self.swarm[i], leader)
                obj = evaluate_solution(new_solution, self.problem.tasks, self.problem.resources)
                if dominates(obj, self.pbest_obj[i]):
                    self.pbest[i] = new_solution
                    self.pbest_obj[i] = obj
                self.swarm[i] = new_solution
            combined = [(sol, evaluate_solution(sol, self.problem.tasks, self.problem.resources))
                        for sol in self.swarm] + self.archive
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
            debug_print(f"MOHHO: Random swap between indices {i} and {j}.")
        else:
            swaps = get_swap_sequence(new_hawk, prey)
            for swap in swaps:
                if random.random() < 0.5:
                    i, j = swap
                    new_hawk[i], new_hawk[j] = new_hawk[j], new_hawk[i]
                    debug_print(f"MOHHO: Guided swap between indices {i} and {j}.")
        return new_hawk

    def run(self):
        pop = self.population.copy()
        pop_eval = [(hawk, evaluate_solution(hawk, self.problem.tasks, self.problem.resources))
                    for hawk in pop]
        self.archive = non_dominated_sort(pop_eval)
        for it in range(self.iterations):
            debug_print(f"MOHHO Iteration {it}")
            prey = random.choice(self.archive)[0] if self.archive else random.choice(pop)
            new_pop = []
            for hawk in pop:
                new_hawk = self.update_hawk(hawk, prey)
                new_pop.append(new_hawk)
            pop = new_pop
            pop_eval = [(hawk, evaluate_solution(hawk, self.problem.tasks, self.problem.resources))
                        for hawk in pop]
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
        self.task_ids = [task.id for task in self.problem.tasks]
        self.index_to_id = {i: task.id for i, task in enumerate(self.problem.tasks)}
        self.id_to_index = {task.id: i for i, task in enumerate(self.problem.tasks)}
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
            debug_print(f"MOACO: Selected task {next_task} with threshold {r:.4f}")
        return solution

    def update_pheromones(self, solutions):
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                self.pheromone[i][j] *= (1 - self.rho)
        for sol, obj in solutions:
            deposit = 1.0 / (obj["makespan"] + obj["waiting_cost"] + 1e-6)
            for k in range(len(sol) - 1):
                i = self.id_to_index[sol[k]]
                j = self.id_to_index[sol[k+1]]
                self.pheromone[i][j] += deposit
                debug_print(f"MOACO: Deposited on edge {sol[k]}->{sol[k+1]}: {deposit:.4f}")

    def run(self):
        population = []
        for it in range(self.iterations):
            debug_print(f"MOACO Iteration {it}")
            ants = []
            for _ in range(self.num_ants):
                sol = self.construct_solution()
                ants.append(sol)
            ants_eval = [(sol, evaluate_solution(sol, self.problem.tasks, self.problem.resources))
                         for sol in ants]
            population.extend(ants_eval)
            nd = non_dominated_sort(ants_eval)
            self.update_pheromones(nd)
        archive = non_dominated_sort(population)
        return [{"permutation": sol, "objectives": obj} for sol, obj in archive]

# -----------------------------------------------
# HHO Implementation (Continuous Domain with Permutation Decoding)
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
        lb = self.lb
        ub = self.ub
        rabbit_position = None
        rabbit_fitness = np.inf
        convergence_curve = np.zeros(self.iterations)
        snapshot_interval = max(1, self.iterations // 5)
        for t in range(self.iterations):
            debug_print(f"HHO Iteration {t}")
            for i in range(num_agents):
                X[i] = np.clip(X[i], lb, ub)
                fitness = self.evaluate(X[i])
                if fitness < rabbit_fitness:
                    rabbit_fitness = fitness
                    rabbit_position = X[i].copy()
                    debug_print(f"HHO: New rabbit by agent {i} with fitness {fitness:.4f}")
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
                            X2 = rabbit_position - Escaping_Energy * np.abs(Jump_strength * rabbit_position - X[i]) + np.random.randn(self.dim) * self.levy(self.dim)
                            if self.evaluate(X2) < self.evaluate(X[i]):
                                X[i] = X2.copy()
                    if r < 0.5 and abs(Escaping_Energy) < 0.5:
                        Jump_strength = 2 * (1 - random.random())
                        X1 = rabbit_position - Escaping_Energy * np.abs(Jump_strength * rabbit_position - np.mean(X, axis=0))
                        if self.evaluate(X1) < self.evaluate(X[i]):
                            X[i] = X1.copy()
                        else:
                            X2 = rabbit_position - Escaping_Energy * np.abs(Jump_strength * rabbit_position - np.mean(X, axis=0)) + np.random.randn(self.dim) * self.levy(self.dim)
                            if self.evaluate(X2) < self.evaluate(X[i]):
                                X[i] = X2.copy()
            convergence_curve[t] = rabbit_fitness
            debug_print(f"HHO Iteration {t}: Best fitness = {rabbit_fitness:.4f}")
        self.rabbit_position = rabbit_position
        self.X = X
        best_perm = self.position_to_permutation(rabbit_position)
        best_obj = evaluate_solution(best_perm, self.problem.tasks, self.problem.resources)
        return {"permutation": best_perm, "objectives": best_obj,
                "convergence": convergence_curve, "snapshots": self.schedule_snapshots}

# -----------------------------------------------
# PSO Implementation (Continuous Domain with Permutation Decoding)
# -----------------------------------------------

class PSOAlgorithm(OptimizationAlgorithm):
    def __init__(self, problem, params):
        super().__init__(problem, params)
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
        for idx in range(self.pop):
            pos = np.random.uniform(self.lb, self.ub, self.dim)
            vel = np.random.uniform(-self.v_max, self.v_max, self.dim)
            cost = self.evaluate(pos)
            particle = {"pos": pos, "vel": vel, "best_pos": pos.copy(), "cost": cost, "best_cost": cost}
            self.particles.append(particle)
            if cost < self.global_best_cost:
                self.global_best_cost = cost
                self.global_best_pos = pos.copy()
            debug_print(f"PSO: Initialized particle {idx} with cost {cost:.4f}")

    def position_to_permutation(self, x):
        indices = np.argsort(x)
        return [self.problem.tasks[i].id for i in indices]

    def evaluate(self, x):
        perm = self.position_to_permutation(x)
        obj = evaluate_solution(perm, self.problem.tasks, self.problem.resources)
        return obj["makespan"]

    def run(self):
        convergence_curve = []
        iter = 0
        while iter < self.max_iter:
            debug_print(f"PSO Iteration {iter}")
            for particle in self.particles:
                for i in range(self.dim):
                    r1 = random.random()
                    r2 = random.random()
                    particle["vel"][i] = (self.inertia_weight * particle["vel"][i] +
                                            self.personal_c * r1 * (particle["best_pos"][i] - particle["pos"][i]) +
                                            self.social_c * r2 * (self.global_best_pos[i] - particle["pos"][i]))
                    if particle["vel"][i] > self.v_max:
                        particle["vel"][i] = self.v_max
                    elif particle["vel"][i] < -self.v_max:
                        particle["vel"][i] = -self.v_max
                particle["pos"] += particle["vel"]
                particle["pos"] = np.clip(particle["pos"], self.lb, self.ub)
                cost = self.evaluate(particle["pos"])
                particle["cost"] = cost
                if cost < particle["best_cost"]:
                    particle["best_cost"] = cost
                    particle["best_pos"] = particle["pos"].copy()
                    debug_print(f"PSO: Updated personal best with cost {cost:.4f}")
                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_pos = particle["pos"].copy()
                    debug_print(f"PSO: Updated global best with cost {cost:.4f}")
            convergence_curve.append(self.global_best_cost)
            if abs(self.global_best_cost) < self.convergence_threshold:
                debug_print(f"PSO converged after {iter} iterations.")
                break
            iter += 1
        best_perm = self.position_to_permutation(self.global_best_pos)
        return {"permutation": best_perm,
                "objectives": evaluate_solution(best_perm, self.problem.tasks, self.problem.resources),
                "convergence": convergence_curve}

# -----------------------------------------------
# Visualization Functions
# -----------------------------------------------

def plot_pareto_fronts(algorithms_results):
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for ax, (name, results) in zip(axes, algorithms_results.items()):
        makespans = [res["objectives"]["makespan"] for res in results]
        waiting = [res["objectives"]["waiting_cost"] for res in results]
        avg_utils = [res["objectives"]["avg_util"] for res in results]
        sc = ax.scatter(makespans, waiting, c=avg_utils, cmap="viridis", s=100, edgecolors="black")
        ax.set_title(f"{name} Pareto Front")
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Waiting Cost")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Avg Resource Utilization")
    plt.tight_layout()
    plt.show()

def plot_aggregated_metrics(aggregated_metrics):
    metric_names = ["makespan", "waiting_cost", "avg_util"]
    algorithms = list(aggregated_metrics.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 6))
    for i, metric in enumerate(metric_names):
        values = []
        for algo in algorithms:
            objs = [res["objectives"][metric] for res in aggregated_metrics[algo]]
            values.append(np.mean(objs))
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
        ax.text(start + (finish - start)/2, i, f"{start}-{finish}", va="center", ha="center", fontsize=9)
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
            ax.text(start + (finish - start)/2, i, f"{start}-{finish}", va="center", ha="center", fontsize=8)
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

    # 1. Generate a randomized RCPSP instance.
    problem_random = generate_randomized_RCPSP(num_tasks=20, resource_types=("A", "B", "C"), max_predecessors=2)
    random_permutation = random.sample([task.id for task in problem_random.tasks], len(problem_random.tasks))
    generated_schedule, gen_makespan, gen_wait, gen_util = decode_solution(random_permutation, problem_random.tasks, problem_random.resources)
    print("=== Randomized RCPSP Instance ===")
    print("Permutation:", random_permutation)
    print("Schedule:", generated_schedule)
    visualize_schedule(generated_schedule, title="Randomized Instance Schedule")
    print("Objectives:", evaluate_solution(random_permutation, problem_random.tasks, problem_random.resources))
    print()

    # 2. Generate the manual optimized RCPSP instance and its schedule.
    problem_manual, manual_optimized_schedule = generate_manual_RCPSP()
    print("=== Manual Optimized RCPSP Instance ===")
    print("Manual Optimized Schedule Data:")
    for tid, (start, finish) in manual_optimized_schedule.items():
        print(f"Task {tid}: Start {start}, Finish {finish}")
    visualize_schedule(manual_optimized_schedule, title="Manual Optimized Schedule")
    manual_perm = [1, 2, 3, 4, 5]
    print("Manual Objectives:", evaluate_solution(manual_perm, problem_manual.tasks, problem_manual.resources))
    print()

    # 3. Define common parameters for the metaheuristics.
    common_params_mopso = {"num_particles": 30, "iterations": 100, "c1": 0.6, "c2": 0.3}
    common_params_mohho = {"num_hawks": 30, "iterations": 100, "q": 0.5}
    common_params_moaco = {"num_ants": 30, "iterations": 100, "tau0": 1.0, "rho": 0.1}
    common_params_hho = {"num_agents": 30, "iterations": 100, "lb": 0.0, "ub": 1.0}
    common_params_pso = {"pop": 20, "v_max": 0.1, "personal_c": 2.0, "social_c": 2.0,
                         "inertia_weight": 0.7, "convergence": 0.001, "max_iter": 100,
                         "lb": 0.0, "ub": 1.0}

    # 4. Instantiate algorithm objects (here, only HHO is used for demonstration).
    algorithms = {
        # "MOPSO": MOPSO(problem_random, common_params_mopso),
        # "MOHHO": MOHHO(problem_random, common_params_mohho),
        # "MOACO": MOACO(problem_random, common_params_moaco),
        "HHO": HHOAlgorithm(problem_random, common_params_hho),
        # "PSO": PSOAlgorithm(problem_random, common_params_pso)
    }

    results = {}
    snapshots = None  # For HHO schedule evolution
    for name, algo in algorithms.items():
        print(f"\nRunning {name} ...")
        result = algo.run()
        if name == "HHO":
            results[name] = [result]
            snapshots = result.get("snapshots", [])
            obj = result["objectives"]
            print(f"--- {name} Best Solution ---")
            print(f"Solution permutation: {result['permutation']}")
            print(f"Objectives: {obj}")
        else:
            pareto_front = result
            results[name] = pareto_front
            print(f"--- {name} Pareto Front ---")
            for res in pareto_front:
                print(f"Solution: {res['permutation']}")
                print(f"Objectives: {res['objectives']}")
        print("-----------------------------------------------------")
    
    # 5. Visualize the Pareto fronts.
    plot_pareto_fronts(results)
    
    # 6. Plot aggregated metrics.
    agg_metrics = {name: res for name, res in results.items()}
    plot_aggregated_metrics(agg_metrics)
    
    # 7. Visualize HHO schedule evolution.
    if snapshots:
        print("\nVisualizing schedule evolution for HHO:")
        plot_schedule_evolution(snapshots)
        final_iter, final_schedule = snapshots[-1]
        visualize_schedule(final_schedule, title=f"Final Schedule (Iter {final_iter})")
    
    # 8. (Optional) Plot PSO convergence curve if PSO is run.
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

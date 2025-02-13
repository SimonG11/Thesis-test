#!/usr/bin/env python3
"""
Comparison of Multi–Objective MOACO, HHO, and PSO on a Multi–Skilled Resource Constrained Project Scheduling Problem.

This script:
  • Defines the scheduling problem (10 tasks, resource capacities, cost per man–hour).
  • Implements three objectives: makespan, total cost, and negative average utilization.
  • Implements a modified Multi–Objective Ant Colony Optimization (MOACO) algorithm.
  • Also uses existing implementations for a multi–objective HHO variant and an improved multi–objective PSO.
  • Runs all three algorithms for a fixed number of iterations.
  • Plots convergence curves, Pareto fronts, and Gantt charts for the best schedules.
  
Author: Simon Gottschalk
Date: 2025-02-11
"""

import numpy as np
import matplotlib.pyplot as plt
import random, math, time, copy
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# =============================================================================
# ------------------------ Helper Functions -------------------------------
# =============================================================================

def dominates(obj_a, obj_b):
    """
    In minimization: solution a dominates b if every objective of a is less than or equal to that of b
    and at least one objective is strictly less.
    """
    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

def levy(dim):
    """Compute a Levy flight step (used in jump improved operation)."""
    beta = 1.5
    sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
             (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2))) ** (1/beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.power(np.abs(v), 1/beta))

def find_earliest_start(earliest, duration, allocated, scheduled_tasks, capacity, resource):
    """
    Given:
      - earliest: the earliest possible start time (from dependency constraints)
      - duration: how long the task will run,
      - allocated: number of workers needed for the task,
      - scheduled_tasks: a list of already scheduled tasks (each with "start", "finish", "workers", "resource")
      - capacity: maximum workers available concurrently for this resource type,
      - resource: the resource type required for the current task.
    This function searches for the earliest time >= earliest such that during the entire interval
    [start, start+duration], the sum of workers from overlapping tasks (using the same resource) plus
    the allocated workers does not exceed the capacity.
    """
    candidate = earliest
    while True:
        events = [candidate, candidate + duration]
        for task in scheduled_tasks:
            if task.get("resource") != resource:
                continue
            if task["finish"] > candidate and task["start"] < candidate + duration:
                if candidate < task["start"] < candidate + duration:
                    events.append(task["start"])
                if candidate < task["finish"] < candidate + duration:
                    events.append(task["finish"])
        events = sorted(set(events))
        feasible = True
        conflict_time = None
        for i in range(len(events)-1):
            mid = (events[i] + events[i+1]) / 2.0
            usage = sum(task["workers"] for task in scheduled_tasks 
                        if task.get("resource") == resource and task["start"] <= mid < task["finish"])
            if usage + allocated > capacity:
                feasible = False
                conflict_time = events[i+1]
                break
        if feasible:
            return candidate
        else:
            candidate = conflict_time

# =============================================================================
# -------------------------- Scheduling Problem Definition ------------------
# =============================================================================

# Define available resources.
workers = {
    "Developer": 10,
    "Manager": 2,
    "Tester": 3
}

# Define cost per man–hour for each worker type.
worker_cost = {
    "Developer": 50,
    "Manager": 75,
    "Tester": 40
}

# Define 10 tasks – each with base effort, allocation bounds, dependencies, and required resource.
tasks = [
    {"id": 1, "task_name": "Requirements Gathering", "base_effort": 80,  "min": 1, "max": 14, "dependencies": [],         "resource": "Manager"},
    {"id": 2, "task_name": "System Design",          "base_effort": 100, "min": 1, "max": 14, "dependencies": [1],        "resource": "Manager"},
    {"id": 3, "task_name": "Module 1 Development",   "base_effort": 150, "min": 1, "max": 14, "dependencies": [2],        "resource": "Developer"},
    {"id": 4, "task_name": "Module 2 Development",   "base_effort": 150, "min": 1, "max": 14, "dependencies": [2],        "resource": "Developer"},
    {"id": 5, "task_name": "Integration",            "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Developer"},
    {"id": 6, "task_name": "Testing",                "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Tester"},
    {"id": 7, "task_name": "Acceptance Testing",     "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Tester"},
    {"id": 8, "task_name": "Documentation",          "base_effort": 100, "min": 1, "max": 14, "dependencies": [4],        "resource": "Developer"},
    {"id": 9, "task_name": "Training",               "base_effort": 50,  "min": 1, "max": 14, "dependencies": [7, 8],     "resource": "Tester"},
    {"id": 10,"task_name": "Deployment",             "base_effort": 70,  "min": 2, "max": 14, "dependencies": [7, 9],     "resource": "Manager"}
]

def compute_schedule(x, tasks, available):
    """
    Given a decision vector x (worker allocations for each task), this function:
      - Clamps the allocation to [min, effective_max] (where effective_max = min(task["max"], available)).
      - Computes the adjusted effort and duration.
      - Determines the earliest feasible start time (after dependencies) that satisfies resource constraints.
    Returns:
      - schedule: list of dictionaries (one per task, including resource info)
      - makespan: overall project finish time.
    """
    schedule = []
    finish_times = {}
    for task in tasks:
        tid = task["id"]
        resource_type = task["resource"]
        capacity = available[resource_type]
        effective_max = min(task["max"], capacity)
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(effective_max, alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
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

# =============================================================================
# ----------------------- Objective Functions -------------------------------
# =============================================================================

def objective_makespan(x):
    """Minimize project makespan."""
    _, ms = compute_schedule(x, tasks, workers)
    return ms

def objective_total_cost(x):
    """Minimize total cost (cost per man–hour)."""
    total_cost = 0
    for task in tasks:
        tid = task["id"]
        resource_type = task["resource"]
        capacity = workers[resource_type]
        effective_max = min(task["max"], capacity)
        alloc = round(x[tid - 1]*2)/2
        alloc = max(task["min"], min(effective_max, alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
        wage_rate = worker_cost[resource_type]
        total_cost += duration * alloc * wage_rate
    return total_cost

def objective_neg_utilization(x):
    """
    Maximize average resource utilization.
    (Return negative so that minimization works.)
    """
    utils = []
    for task in tasks:
        tid = task["id"]
        resource_type = task["resource"]
        capacity = workers[resource_type]
        effective_max = min(task["max"], capacity)
        alloc = round(x[tid - 1]*2)/2
        alloc = max(task["min"], min(effective_max, alloc))
        utils.append(alloc / task["max"])
    return -np.mean(utils)

def multi_objective(x):
    """Return an objective vector: [makespan, total cost, -average utilization]."""
    return np.array([objective_makespan(x), objective_total_cost(x), objective_neg_utilization(x)])

# =============================================================================
# ----------------------- Visualization Functions ---------------------------
# =============================================================================

def plot_gantt(schedule, title):
    """Plot a Gantt chart for the given schedule."""
    fig, ax = plt.subplots(figsize=(10,6))
    yticks = []
    yticklabels = []
    for i, task in enumerate(schedule):
        ax.broken_barh([(task["start"], task["duration"])],
                       (i*10, 9),
                       facecolors='tab:blue')
        yticks.append(i*10+5)
        yticklabels.append(f"Task {task['task_id']}: {task['task_name']} ({task['resource']})\n(Workers: {task['workers']})")
        ax.text(task["start"]+task["duration"]/2, i*10+5, f"{task['start']:.1f}-{task['finish']:.1f}",
                ha='center', va='center', color='white', fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Tasks")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# =============================================================================
# ----------------------- MOHHO Implementation ------------------------------
# (Existing implementation)
# =============================================================================

def MOHHO_with_progress(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Modified Multi–Objective HHO that:
      - Uses operators similar to standard HHO.
      - Records the best makespan (first objective) at each iteration.
      - Returns an archive of Pareto–optimal solutions and a list of best makespan per iteration.
    """
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    archive = []
    progress = []
    t = 0
    while t < Max_iter:
        for i in range(SearchAgents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            f_val = objf(X[i, :])
            dominated_flag = False
            removal_list = []
            for (sol_arch, f_arch) in archive:
                if dominates(f_arch, f_val):
                    dominated_flag = True
                    break
                if dominates(f_val, f_arch):
                    removal_list.append((sol_arch, f_arch))
            if not dominated_flag:
                new_archive = []
                for entry in archive:
                    should_remove = False
                    for rem in removal_list:
                        if np.array_equal(entry[0], rem[0]) and np.array_equal(entry[1], rem[1]):
                            should_remove = True
                            break
                    if not should_remove:
                        new_archive.append(entry)
                archive = new_archive
                archive.append((X[i, :].copy(), f_val.copy()))
        if archive:
            rabbit = random.choice(archive)[0]
        else:
            rabbit = X[0, :].copy()
        E1 = 2 * (1 - (t / Max_iter))
        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_index = random.randint(0, SearchAgents_no - 1)
                X_rand = X[rand_index, :].copy()
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * np.abs(X_rand - 2 * random.random() * X[i, :])
                else:
                    X[i, :] = (rabbit - np.mean(X, axis=0)) - random.random() * ((ub - lb) * random.random() + lb)
            else:
                r = random.random()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = rabbit - Escaping_Energy * np.abs(rabbit - X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X[i, :] = (rabbit - X[i, :]) - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :])
                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :])
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :]) + np.multiply(np.random.randn(dim), levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0)) + np.multiply(np.random.randn(dim), levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
        best_makespan = np.min([objf(X[i, :])[0] for i in range(SearchAgents_no)])
        progress.append(best_makespan)
        t += 1
    return archive, progress

# =============================================================================
# ------------------------- PSO Implementation (Improved) ---------------------
# =============================================================================

class PSO:
    def __init__(self, dim, lb, ub, obj_funcs, pop=30, c2=1.05, w_max=0.9, w_min=0.4,
                 disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20):
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.obj_funcs = obj_funcs if isinstance(obj_funcs, list) else [obj_funcs]
        self.pop = pop
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.iteration = 0
        self.max_iter = 200  # adjust as needed
        self.vmax = self.ub - self.lb
        self.integer = [True]*dim  # decision variables are integer (workers)
        # Initialize swarm: positions and velocities (each particle as a dict)
        self.swarm = []
        for _ in range(pop):
            pos = np.array([random.randint(self.lb[i], self.ub[i]) for i in range(dim)])
            vel = np.array([random.uniform(-self.vmax[i], self.vmax[i]) for i in range(dim)])
            particle = {
                'position': pos,
                'velocity': vel,
                'pbest': pos.copy(),  # pbest is set equal to current position
                'obj': self.evaluate(pos)
            }
            self.swarm.append(particle)
        # External archive to store non-dominated solutions (each entry: (position, obj))
        self.archive = []
        self.disturbance_rate_min = disturbance_rate_min
        self.disturbance_rate_max = disturbance_rate_max
        self.jump_interval = jump_interval

    def evaluate(self, pos):
        """Evaluate the multi-objective vector at pos."""
        if len(self.obj_funcs) == 1:
            return np.array([self.obj_funcs[0](pos)])
        else:
            return np.array([f(pos) for f in self.obj_funcs])

    def update_archive(self):
        """Update external archive with non-dominated solutions from the swarm."""
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            dominated_flag = False
            removal_list = []
            for (arch_pos, arch_obj) in self.archive:
                if dominates(arch_obj, obj_val):
                    dominated_flag = True
                    break
                if dominates(obj_val, arch_obj):
                    removal_list.append((arch_pos, arch_obj))
            if not dominated_flag:
                self.archive = [entry for entry in self.archive 
                                if not any(np.array_equal(entry[0], rem[0]) and np.array_equal(entry[1], rem[1])
                                           for rem in removal_list)]
                self.archive.append((pos, obj_val))
        # (Optional: cluster archive members to maintain diversity)

    def proportional_distribution(self):
        """
        For each archive member, compute a density (average Euclidean distance in objective space).
        Then assign each particle a guide (archive solution) with probability proportional to density.
        """
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        arch_objs = [entry[1] for entry in self.archive]
        n = len(arch_objs)
        densities = []
        for i in range(n):
            if n == 1:
                densities.append(1.0)
            else:
                dists = [np.linalg.norm(arch_objs[i]-arch_objs[j]) for j in range(n) if i != j]
                densities.append(np.mean(dists))
        total_density = sum(densities)
        probs = [d/total_density for d in densities]
        guides = []
        for _ in range(self.pop):
            r = random.random()
            cum_prob = 0
            for idx, p in enumerate(probs):
                cum_prob += p
                if r <= cum_prob:
                    guides.append(self.archive[idx][0])
                    break
        return guides

    def jump_improved_operation(self):
        """
        Jump improved operation (outward jumping) on the archive.
        Two archive members are chosen randomly and new candidates are generated.
        """
        if len(self.archive) < 2:
            return
        c1, c2 = random.sample(self.archive, 2)
        a1, a2 = random.uniform(0,1), random.uniform(0,1)
        oc1 = c1[0] + a1 * (c1[0] - c2[0])
        oc2 = c2[0] + a2 * (c2[0] - c1[0])
        oc1 = np.array([int(np.clip(val, self.lb[i], self.ub[i])) for i, val in enumerate(oc1)])
        oc2 = np.array([int(np.clip(val, self.lb[i], self.ub[i])) for i, val in enumerate(oc2)])
        for oc in [oc1, oc2]:
            obj_val = self.evaluate(oc)
            dominated_flag = False
            removal_list = []
            for (arch_pos, arch_obj) in self.archive:
                if dominates(arch_obj, obj_val):
                    dominated_flag = True
                    break
                if dominates(obj_val, arch_obj):
                    removal_list.append((arch_pos, arch_obj))
            if not dominated_flag:
                self.archive = [entry for entry in self.archive 
                                if not any(np.array_equal(entry[0], rem[0]) and np.array_equal(entry[1], rem[1])
                                           for rem in removal_list)]
                self.archive.append((oc, obj_val))

    def disturbance_operation(self, particle):
        """
        Disturb a particle’s position on a randomly selected subset of dimensions.
        The disturbance rate increases linearly with iterations.
        """
        rate = self.disturbance_rate_min + (self.disturbance_rate_max - self.disturbance_rate_min) * (self.iteration / self.max_iter)
        if random.random() < rate:
            k = random.randint(1, self.dim)
            dims = random.sample(range(self.dim), k)
            new_pos = particle['position'].copy()
            for d in dims:
                rn = np.random.normal(0.5, 1)
                if rn < 0.5:
                    new_pos[d] = new_pos[d] - 0.5 * (new_pos[d] - self.lb[d]) * rn
                else:
                    new_pos[d] = new_pos[d] + 0.5 * (self.ub[d] - new_pos[d]) * rn
                new_pos[d] = int(np.clip(new_pos[d], self.lb[d], self.ub[d]))
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)

    def move(self):
        """
        One iteration:
         - Update velocity & position (using guides from the archive via proportional distribution)
         - Apply disturbance on particles
         - Update the external archive
         - Every jump_interval iterations, apply jump improved operation
        """
        self.iteration += 1
        w = self.w_max - ((self.w_max - self.w_min) * (self.iteration / self.max_iter))
        guides = self.proportional_distribution()
        for idx, particle in enumerate(self.swarm):
            r2 = random.random()
            guide = guides[idx]
            new_v = w * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([int(np.clip(round(new_pos[i]), self.lb[i], self.ub[i])) for i in range(self.dim)])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)
            particle['pbest'] = new_pos.copy()
            self.disturbance_operation(particle)
        self.update_archive()
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()

    def run(self, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
        convergence = []
        for _ in range(max_iter):
            self.move()
            best_ms = min([p['obj'][0] for p in self.swarm])
            convergence.append(best_ms)
        return convergence

# =============================================================================
# ------------------------- MOACO Implementation ----------------------------
# =============================================================================
def MOACO_improved(objf, tasks, workers, lb, ub, ant_count, max_iter,
                     alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0,
                     P=0.6, w1=1.0, w2=1.0, sigma_share=1.0, lambda3=2.0, lambda4=5.0):
    """
    An improved multi-objective ACO that incorporates:
      - A heuristic layout updating (local search) strategy: each constructed solution is refined
        by exploring its neighborhood (by perturbing one decision variable at a time).
      - Two pheromone updating modes:
           (a) Local pheromone communication update (applied with probability 1-P),
           (b) Global search update based on niche technology (applied with probability P).
    
    In this implementation the decision vector corresponds to the worker allocations for tasks.
    
    Parameters:
      objf               : The multi-objective function.
      tasks, workers, lb, ub : Problem definitions (as in the scheduling problem).
      ant_count          : Number of ants.
      max_iter           : Maximum iterations.
      alpha, beta        : Exponents for pheromone and heuristic influence.
      evaporation_rate   : Global evaporation rate.
      Q                  : Base pheromone deposit factor.
      P                  : Probability threshold for choosing niche-based update.
      w1, w2             : Weights for the two update modes.
      sigma_share        : Niche radius (used in computing shared function).
      lambda3, lambda4   : Quality function levels (for local and niche‐based updates).
    
    Returns:
      archive: list of tuples (solution, obj_vector)
      progress: list recording best primary objective (here, makespan) per iteration.
    """
    dim = len(lb)
    # Initialize pheromone and heuristic for each task.
    pheromone = []
    heuristic = []
    for i in range(dim):
        possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
        pheromone.append({v: 1.0 for v in possible_values})
        h_dict = {}
        task = tasks[i]
        for v in possible_values:
            new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (v - 1))
            duration = new_effort / v
            h_dict[v] = 1.0 / duration  # shorter duration → higher heuristic
        heuristic.append(h_dict)
        
    archive = []  # global archive of non-dominated solutions: list of (solution, obj_vector)
    progress = []
    
    for iteration in range(max_iter):
        population = []  # solutions produced in current iteration
        for ant in range(ant_count):
            # ---- Construct solution: sample each decision variable with probability proportional
            # to [pheromone^alpha * heuristic^beta] (for each task)
            solution = []
            for i in range(dim):
                possible_values = list(pheromone[i].keys())
                probs = []
                for v in possible_values:
                    tau = pheromone[i][v]
                    h_val = heuristic[i][v]
                    probs.append((tau ** alpha) * (h_val ** beta))
                total = sum(probs)
                if total == 0:
                    probs = [1/len(probs)] * len(probs)
                else:
                    probs = [p/total for p in probs]
                r = random.random()
                cumulative = 0.0
                chosen = possible_values[-1]
                for idx, v in enumerate(possible_values):
                    cumulative += probs[idx]
                    if r <= cumulative:
                        chosen = v
                        break
                solution.append(chosen)
            
            # ---- Heuristic layout updating / local search step:
            # Generate neighbors by perturbing each decision variable by +/- 1 (if within bounds)
            neighbors = []
            for i in range(dim):
                for delta in [-1, 1]:
                    neighbor = solution.copy()
                    neighbor[i] = int(np.clip(neighbor[i] + delta, lb[i], ub[i]))
                    neighbors.append(neighbor)
            # Evaluate neighbors using the primary objective (makespan) and pick the best improvement.
            best_neighbor = solution
            best_obj = objf(solution)
            for neighbor in neighbors:
                n_obj = objf(neighbor)
                if n_obj[0] < best_obj[0]:
                    best_obj = n_obj
                    best_neighbor = neighbor
            solution = best_neighbor  # update solution after local search
            
            # Evaluate final solution and add to population.
            obj_val = objf(solution)
            population.append((solution, obj_val))
        
        # ---- Identify non-dominated solutions among population.
        non_dominated = []
        for sol, obj_val in population:
            dominated_flag = False
            for other_sol, other_obj in population:
                if dominates(other_obj, obj_val):
                    dominated_flag = True
                    break
            if not dominated_flag:
                non_dominated.append((sol, obj_val))
                
        # ---- Update global archive (using standard Pareto-dominance checks).
        for sol, obj_val in non_dominated:
            dominated_flag = False
            removal_list = []
            for a_sol, a_obj in archive:
                if dominates(a_obj, obj_val):
                    dominated_flag = True
                    break
                if dominates(obj_val, a_obj):
                    removal_list.append((a_sol, a_obj))
            if not dominated_flag:
                archive = [entry for entry in archive if entry not in removal_list]
                archive.append((sol, obj_val))
        
        # ---- Pheromone evaporation:
        for i in range(dim):
            for v in pheromone[i]:
                pheromone[i][v] *= (1 - evaporation_rate)
        
        # ---- Pheromone deposit/update:
        # For each non-dominated solution we use one of two update strategies.
        for sol, obj_val in non_dominated:
            r = random.random()
            if r > P:
                # Use local pheromone communication update.
                # In our simple version we set the quality function to lambda3.
                deposit = w1 * lambda3
            else:
                # Use global (niche-based) search update.
                # First, compute niche counts for each solution in the archive.
                niche_counts = []
                for (arch_sol, arch_obj) in archive:
                    count = 0.0
                    for (other_sol, other_obj) in archive:
                        if np.array_equal(arch_sol, other_sol):
                            continue
                        d = np.linalg.norm(arch_obj - other_obj)
                        if d < sigma_share:
                            count += (1 - d/sigma_share)
                    niche_counts.append(count)
                # Select the solution in the archive with the smallest niche count.
                min_index = np.argmin(niche_counts)
                chosen_sol, chosen_obj = archive[min_index]
                # Compute mu as the minimum Euclidean distance in objective space from the chosen solution to any other.
                distances = []
                for (other_sol, other_obj) in archive:
                    if np.array_equal(chosen_sol, other_sol):
                        continue
                    distances.append(np.linalg.norm(chosen_obj - other_obj))
                mu = min(distances) if distances else 0
                deposit = w2 * (lambda4 if mu > 0 else lambda3)
            # Now update the pheromone for each task with the deposit from this solution.
            for i, v in enumerate(sol):
                pheromone[i][v] += deposit  # note: this is an additive update
                
        # ---- Record best primary objective value from current population.
        best_ms = min(obj_val[0] for sol, obj_val in population)
        progress.append(best_ms)
    
    return archive, progress

# =============================================================================
# ---------------------------- Main Comparison ------------------------------
# =============================================================================

if __name__ == '__main__':
    # Define decision space: one worker allocation per task.
    dim = len(tasks)
    lb = np.array([task["min"] for task in tasks])
    ub = np.array([task["max"] for task in tasks])
    
    # Compute a baseline schedule (using midpoint allocation).
    baseline_x = (lb + ub) / 2.0
    baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks, workers)
    print("Baseline Makespan (hours):", baseline_makespan)
    plot_gantt(baseline_schedule, "Baseline Schedule")
    
    # ----------------------- Run MOHHO -----------------------
    hho_iterations = 100
    SearchAgents_no = 10
    archive_hho, hho_progress = MOHHO_with_progress(multi_objective, lb, ub, dim, SearchAgents_no, hho_iterations)
    if archive_hho:
        best_particle_hho = min(archive_hho, key=lambda entry: entry[1][0])
        best_solution_hho = best_particle_hho[0]
        best_makespan_hho = best_particle_hho[1][0]
        best_schedule_hho, _ = compute_schedule(best_solution_hho, tasks, workers)
        print("[HHO] Best Makespan (hours):", best_makespan_hho)
    else:
        best_schedule_hho = None
        best_makespan_hho = None
        print("[HHO] No feasible solution found.")
    
    # ----------------------- Run Improved PSO -----------------------
    objectives = [objective_makespan, objective_total_cost, objective_neg_utilization]
    optimizer = PSO(dim=dim, lb=lb, ub=ub, obj_funcs=objectives,
                    pop=10, c2=1.05, w_max=0.9, w_min=0.4,
                    disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
    pso_progress = optimizer.run(max_iter=100)
    archive_pso = optimizer.archive
    print(f"[PSO] Number of archive (non-dominated) solutions found: {len(archive_pso)}")
    best_arch = min(archive_pso, key=lambda entry: entry[1][0])
    best_solution_pso = best_arch[0]
    best_schedule_pso, best_makespan_pso = compute_schedule(best_solution_pso, tasks, workers)
    print("[PSO] Best Makespan (hours):", best_makespan_pso)
    
    # ----------------------- Run MOACO -----------------------
    moaco_iterations = 10
    ant_count = 10
    # MOACO parameters: alpha (pheromone influence), beta (heuristic influence),
    # evaporation_rate, and Q (pheromone deposit factor)
    archive_moaco, moaco_progress = MOACO_improved(multi_objective, tasks, workers, lb, ub,
                                          ant_count, moaco_iterations,
                                          alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0)
    if archive_moaco:
        best_particle_moaco = min(archive_moaco, key=lambda entry: entry[1][0])
        best_solution_moaco = best_particle_moaco[0]
        best_makespan_moaco = best_particle_moaco[1][0]
        best_schedule_moaco, _ = compute_schedule(best_solution_moaco, tasks, workers)
        print("[MOACO] Best Makespan (hours):", best_makespan_moaco)
    else:
        best_schedule_moaco = None
        best_makespan_moaco = None
        print("[MOACO] No feasible solution found.")
    
    # ----------------- Convergence Comparison Plot -----------------
    plt.figure(figsize=(10,6))
    plt.plot(range(hho_iterations), hho_progress, label="HHO", linewidth=2)
    plt.plot(range(len(pso_progress)), pso_progress, label="PSO", linewidth=2)
    plt.plot(range(moaco_iterations), moaco_progress, label="MOACO", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Makespan (hours)")
    plt.title("Convergence Comparison: HHO vs. PSO vs. MOACO")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ----------------- Pareto Front Plots (2D: Makespan vs. Cost) -----------------
    def plot_pareto(archive, label, marker, color):
        objs = np.array([entry[1] for entry in archive])
        makespans = objs[:, 0]
        costs = objs[:, 1]
        utils = -objs[:, 2]  # convert negative utilization back
        plt.scatter(makespans, costs, c=color, marker=marker, s=80, edgecolor='k', label=label)
    
    plt.figure(figsize=(8,6))
    if archive_hho:
        plot_pareto(archive_hho, "HHO", 'o', 'blue')
    if archive_pso:
        plot_pareto(archive_pso, "PSO", '^', 'red')
    if archive_moaco:
        plot_pareto(archive_moaco, "MOACO", 's', 'green')
    plt.xlabel("Makespan (hours)")
    plt.ylabel("Total Cost")
    plt.title("Combined 2D Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ----------------- Combined 3D Pareto Front Comparison -----------------
    if archive_hho and archive_pso and archive_moaco:
        hho_objs = np.array([entry[1] for entry in archive_hho])
        pso_objs = np.array([entry[1] for entry in archive_pso])
        moaco_objs = np.array([entry[1] for entry in archive_moaco])
        fig = plt.figure(figsize=(16, 7))
        
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(hho_objs[:,0], hho_objs[:,1], -hho_objs[:,2], c='blue', marker='o', s=80, edgecolor='k', label="HHO")
        ax.scatter(pso_objs[:,0], pso_objs[:,1], -pso_objs[:,2], c='red', marker='^', s=80, edgecolor='k', label="PSO")
        ax.scatter(moaco_objs[:,0], moaco_objs[:,1], -moaco_objs[:,2], c='green', marker='s', s=80, edgecolor='k', label="MOACO")
        ax.set_xlabel("Makespan (hours)")
        ax.set_ylabel("Total Cost")
        ax.set_zlabel("Average Utilization")
        ax.set_title("Combined 3D Pareto Front")
        ax.legend()
        plt.show()
    
    # ----------------- Display Final Schedules via Gantt Charts -----------------
    if best_schedule_hho is not None:
        plot_gantt(best_schedule_hho, f"Optimized Schedule (HHO)\nMakespan: {best_makespan_hho:.2f} hrs")
    if best_schedule_pso is not None:
        plot_gantt(best_schedule_pso, f"Optimized Schedule (PSO)\nMakespan: {best_makespan_pso:.2f} hrs")
        print(best_solution_pso)
    if best_schedule_moaco is not None:
        plot_gantt(best_schedule_moaco, f"Optimized Schedule (MOACO)\nMakespan: {best_makespan_moaco:.2f} hrs")

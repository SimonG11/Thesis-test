#!/usr/bin/env python3
"""
Multi-Objective Comparison for RCPSP using MOHHO, MOPSO, and MOACO
Incorporates improvements for scientific rigor including:
 - Additional multi-objective performance metrics (hypervolume, generational distance, spread)
 - Multiple independent runs and statistical logging
 - Enhanced archive update with crowding distance
 - Modular code structure with reproducibility (fixed random seed)
 - Placeholder for parameter tuning

Author: Simon Gottschalk
Date: 2025-02-13
"""

import numpy as np
import matplotlib.pyplot as plt
import random, math, time, copy, json
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# ----------------------------- Reproducibility -----------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)

# =============================================================================
# -------------------------- Helper Functions -------------------------------
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
    This function finds the earliest start time such that the resource capacity constraint is satisfied.
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
# ----------------------- Problem Definition -------------------------------
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

# Define 10 tasks with dependencies and resource requirements.
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
    Given a decision vector x (worker allocations for each task), compute the schedule.
    Returns:
      - schedule: list of task dictionaries (with start, finish, duration, workers, resource)
      - makespan: overall project finish time.
    """
    schedule = []
    finish_times = {}
    for task in tasks:
        tid = task["id"]
        resource_type = task["resource"]
        capacity = available[resource_type]
        effective_max = min(task["max"], capacity)
        # Clamp and round allocation
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
# ----------------------- Performance Metrics -------------------------------
# =============================================================================

def approximate_hypervolume(archive, reference_point, num_samples=10000):
    """
    Approximate hypervolume using Monte Carlo sampling.
    Archive: list of (solution, obj_vector)
    reference_point: a vector that is dominated by all solutions.
    Note: Assumes minimization for all objectives.
    """
    objs = np.array([entry[1] for entry in archive])
    # Determine bounds of hyper-rectangle (best values from archive)
    mins = np.min(objs, axis=0)
    ref = np.array(reference_point)
    # Generate random samples in the hyper-rectangle defined by ref and mins.
    samples = np.random.uniform(low=mins, high=ref, size=(num_samples, len(ref)))
    count = 0
    for sample in samples:
        # Check if sample is dominated by at least one archive member.
        if any(np.all(sol <= sample) for sol in objs):
            count += 1
    vol = np.prod(ref - mins)
    return (count / num_samples) * vol


def compute_crowding_distance(archive):
    """
    Compute crowding distance for each solution in archive.
    Archive is a list of tuples (solution, obj_vector).
    Returns a list of distances corresponding to archive indices.
    """
    if len(archive) == 0:
        return []
    objs = np.array([entry[1] for entry in archive])
    num_objs = objs.shape[1]
    distances = np.zeros(len(archive))
    for m in range(num_objs):
        sorted_indices = np.argsort(objs[:, m])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
        m_values = objs[sorted_indices, m]
        m_range = m_values[-1] - m_values[0]
        if m_range == 0:
            continue
        for i in range(1, len(archive)-1):
            distances[sorted_indices[i]] += (m_values[i+1] - m_values[i-1]) / m_range
    return distances

def same_entry(entry1, entry2):
    """Return True if both archive entries are identical."""
    return np.array_equal(entry1[0], entry2[0]) and np.array_equal(entry1[1], entry2[1])

def update_archive_with_crowding(archive, new_entry, max_archive_size=50):
    """
    Update archive with new_entry (solution, obj_vector).
    If archive exceeds max_archive_size, remove the solution with the smallest crowding distance.
    """
    sol_new, obj_new = new_entry
    dominated_flag = False
    removal_list = []
    for (sol_arch, obj_arch) in archive:
        if dominates(obj_arch, obj_new):
            dominated_flag = True
            break
        if dominates(obj_new, obj_arch):
            removal_list.append((sol_arch, obj_arch))
    if not dominated_flag:
        # Use same_entry to compare elements and filter out dominated ones.
        archive = [entry for entry in archive if not any(same_entry(entry, rem) for rem in removal_list)]
        archive.append(new_entry)
        if len(archive) > max_archive_size:
            distances = compute_crowding_distance(archive)
            min_index = np.argmin(distances)
            archive.pop(min_index)
    return archive

def compute_generational_distance(archive, true_pareto):
    """
    Compute the generational distance (GD) between archive and a true Pareto front.
    Here, true_pareto is assumed to be an array of objective vectors.
    """
    if len(archive) == 0 or len(true_pareto) == 0:
        return None
    objs = np.array([entry[1] for entry in archive])
    distances = []
    for sol in objs:
        dists = np.linalg.norm(true_pareto - sol, axis=1)
        distances.append(np.min(dists))
    return np.mean(distances)

def compute_spread(archive):
    """
    Compute spread/diversity of solutions in archive.
    Here we use the average pairwise Euclidean distance in objective space.
    """
    if len(archive) < 2:
        return 0
    objs = np.array([entry[1] for entry in archive])
    n = len(objs)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(objs[i]-objs[j]))
    return np.mean(dists)

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

def plot_convergence(metrics_dict, metric_name):
    """
    Plot boxplots for a given metric across multiple runs.
    metrics_dict is a dictionary with keys: algorithm names and values: list of metric values.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    data = [metrics_dict[key] for key in metrics_dict]
    ax.boxplot(data, labels=list(metrics_dict.keys()))
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribution of {metric_name} across runs")
    plt.grid(True)
    plt.show()

def plot_pareto_2d(archives, labels, markers, colors):
    """
    Plot 2D Pareto front (Makespan vs. Cost) for given archives.
    archives: list of archives (each archive is a list of (solution, obj_vector))
    """
    plt.figure(figsize=(8,6))
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
            makespans = objs[:, 0]
            costs = objs[:, 1]
            plt.scatter(makespans, costs, c=color, marker=marker, s=80, edgecolor='k', label=label)
    plt.xlabel("Makespan (hours)")
    plt.ylabel("Total Cost")
    plt.title("2D Pareto Front (Makespan vs. Cost)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pareto_3d(archives, labels, markers, colors):
    """
    Plot 3D Pareto front (Makespan, Cost, Average Utilization) for given archives.
    """
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111, projection='3d')
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
            ax.scatter(objs[:,0], objs[:,1], -objs[:,2], c=color, marker=marker, s=80, edgecolor='k', label=label)
    ax.set_xlabel("Makespan (hours)")
    ax.set_ylabel("Total Cost")
    ax.set_zlabel("Average Utilization")
    ax.set_title("3D Pareto Front")
    ax.legend()
    plt.show()

# =============================================================================
# ----------------------- Parameter Tuning (Placeholder) --------------------
# =============================================================================

def tune_parameters(algorithm_func, param_grid):
    """
    Placeholder function for hyper-parameter tuning.
    Given an algorithm function and a grid of parameters, perform a search to select the best parameters.
    """
    # This function can be implemented using grid search or Bayesian optimization.
    # For now, simply return default parameters.
    return {k: v[0] for k, v in param_grid.items()}

# =============================================================================
# ------------------------- MOHHO Implementation ----------------------------
# =============================================================================

def MOHHO_with_progress(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Modified Multi–Objective HHO.
    Returns an archive of Pareto–optimal solutions and a list of best makespan per iteration.
    """
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    archive = []
    progress = []
    t = 0
    while t < Max_iter:
        for i in range(SearchAgents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            f_val = objf(X[i, :])
            # Update archive with crowding
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))
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
        self.max_iter = 200  # can be tuned
        self.vmax = self.ub - self.lb
        self.integer = [True]*dim  # decision variables are integers (workers)
        self.swarm = []
        for _ in range(pop):
            pos = np.array([random.randint(self.lb[i], self.ub[i]) for i in range(dim)])
            vel = np.array([random.uniform(-self.vmax[i], self.vmax[i]) for i in range(dim)])
            particle = {
                'position': pos,
                'velocity': vel,
                'pbest': pos.copy(),
                'obj': self.evaluate(pos)
            }
            self.swarm.append(particle)
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
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))
    

    def proportional_distribution(self):
        if not self.archive:
            # If the archive is empty, pick swarm positions at random
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        
        distances = compute_crowding_distance(self.archive)
        total = sum(distances)
        
        # --- Fix here: handle inf or NaN in 'total' ---
        if total == 0 or math.isinf(total) or math.isnan(total):
            # Fallback to uniform probabilities
            # (Because we might have only 1 or 2 solutions in the archive)
            probs = [1.0 / len(distances)] * len(distances)
        else:
            probs = [d / total for d in distances]
    
        guides = []
        for _ in range(self.pop):
            r = random.random()
            cum_prob = 0
            chosen_idx = len(probs) - 1
            for idx, p in enumerate(probs):
                cum_prob += p
                if r <= cum_prob:
                    chosen_idx = idx
                    break
            guides.append(self.archive[chosen_idx][0])
    
        return guides

    def jump_improved_operation(self):
        """
        Jump improved operation on the archive.
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
            self.archive = update_archive_with_crowding(self.archive, (oc, obj_val))

    def disturbance_operation(self, particle):
        """
        Disturb a particle’s position on a random subset of dimensions.
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
        One iteration: update velocities and positions using guides from the archive.
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
    Improved multi-objective ACO.
    Returns archive of Pareto–optimal solutions and a progress list.
    """
    dim = len(lb)
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
            h_dict[v] = 1.0 / duration
        heuristic.append(h_dict)
        
    archive = []  # global archive of non-dominated solutions
    progress = []
    
    for iteration in range(max_iter):
        population = []
        for ant in range(ant_count):
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
            
            # Local search: explore neighborhood (perturbation by +/- 1)
            neighbors = []
            for i in range(dim):
                for delta in [-1, 1]:
                    neighbor = solution.copy()
                    neighbor[i] = int(np.clip(neighbor[i] + delta, lb[i], ub[i]))
                    neighbors.append(neighbor)
            best_neighbor = solution
            best_obj = objf(solution)
            for neighbor in neighbors:
                n_obj = objf(neighbor)
                if n_obj[0] < best_obj[0]:
                    best_obj = n_obj
                    best_neighbor = neighbor
            solution = best_neighbor
            obj_val = objf(solution)
            population.append((solution, obj_val))
        
        # Identify non-dominated solutions among population.
        non_dominated = []
        for sol, obj_val in population:
            if not any(dominates(other_obj, obj_val) for _, other_obj in population):
                non_dominated.append((sol, obj_val))
                
        # Update global archive.
        for sol, obj_val in non_dominated:
            archive = update_archive_with_crowding(archive, (sol, obj_val))
        
        # Pheromone evaporation.
        for i in range(dim):
            for v in pheromone[i]:
                pheromone[i][v] *= (1 - evaporation_rate)
        
        # Pheromone deposit/update.
        for sol, obj_val in non_dominated:
            r = random.random()
            if r > P:
                deposit = w1 * lambda3
            else:
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
                min_index = np.argmin(niche_counts)
                chosen_sol, chosen_obj = archive[min_index]
                distances = []
                for (other_sol, other_obj) in archive:
                    if np.array_equal(chosen_sol, other_sol):
                        continue
                    distances.append(np.linalg.norm(chosen_obj - other_obj))
                mu = min(distances) if distances else 0
                deposit = w2 * (lambda4 if mu > 0 else lambda3)
            for i, v in enumerate(sol):
                pheromone[i][v] += deposit
                
        best_ms = min(obj_val[0] for sol, obj_val in population)
        progress.append(best_ms)
    
    return archive, progress

# =============================================================================
# ------------------------- Experiment Runner -------------------------------
# =============================================================================

def run_experiments(runs=10):
    """
    Run multiple independent experiments for MOHHO, PSO, and MOACO.
    Returns a dictionary with collected metrics.
    """
    # Decision space
    dim = len(tasks)
    lb = np.array([task["min"] for task in tasks])
    ub = np.array([task["max"] for task in tasks])
    
    # Reference point for hypervolume (set conservatively worse than any found solution)
    reference_point = np.array([200, 20000, 0])  # adjust as needed
    
    # Containers for metrics across runs.
    results = {
        "MOHHO": {"best_makespan": [], "hypervolume": [], "spread": []},
        "PSO": {"best_makespan": [], "hypervolume": [], "spread": []},
        "MOACO": {"best_makespan": [], "hypervolume": [], "spread": []},
    }
    archives_all = {"MOHHO": [], "PSO": [], "MOACO": []}  # To later compute a union for GD
    
    for run in range(runs):
        print(f"Run {run+1}/{runs} ...")
        # Baseline schedule (for comparison and Gantt chart display)
        baseline_x = (lb + ub) / 2.0
        baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks, workers)
        
        # Run MOHHO
        hho_iter = 100
        SearchAgents_no = 10
        archive_hho, hho_progress = MOHHO_with_progress(multi_objective, lb, ub, dim, SearchAgents_no, hho_iter)
        if archive_hho:
            best_particle_hho = min(archive_hho, key=lambda entry: entry[1][0])
            best_makespan_hho = best_particle_hho[1][0]
        else:
            best_makespan_hho = None
        results["MOHHO"]["best_makespan"].append(best_makespan_hho)
        archives_all["MOHHO"].append(archive_hho)
        
        # Run PSO
        objectives = [objective_makespan, objective_total_cost, objective_neg_utilization]
        optimizer = PSO(dim=dim, lb=lb, ub=ub, obj_funcs=objectives,
                        pop=10, c2=1.05, w_max=0.9, w_min=0.4,
                        disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
        pso_progress = optimizer.run(max_iter=100)
        archive_pso = optimizer.archive
        if archive_pso:
            best_arch = min(archive_pso, key=lambda entry: entry[1][0])
            best_makespan_pso = best_arch[1][0]
        else:
            best_makespan_pso = None
        results["PSO"]["best_makespan"].append(best_makespan_pso)
        archives_all["PSO"].append(archive_pso)
        
        # Run MOACO
        ant_count = 10
        moaco_iter = 100  # increased iterations for fairness
        archive_moaco, moaco_progress = MOACO_improved(multi_objective, tasks, workers, lb, ub,
                                                       ant_count, moaco_iter,
                                                       alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0)
        if archive_moaco:
            best_particle_moaco = min(archive_moaco, key=lambda entry: entry[1][0])
            best_makespan_moaco = best_particle_moaco[1][0]
        else:
            best_makespan_moaco = None
        results["MOACO"]["best_makespan"].append(best_makespan_moaco)
        archives_all["MOACO"].append(archive_moaco)
        
        # Compute hypervolume and spread for each algorithm's archive
        for alg in ["MOHHO", "PSO", "MOACO"]:
            archive = archives_all[alg][-1]
            if archive:
                hv = approximate_hypervolume(archive, reference_point)
                sp = compute_spread(archive)
            else:
                hv, sp = None, None
            results[alg]["hypervolume"].append(hv)
            results[alg]["spread"].append(sp)
    
    # Compute union of archives (as approximate true Pareto front) for GD.
    union_archive = []
    for alg in archives_all:
        for arch in archives_all[alg]:
            union_archive.extend(arch)
    # Remove dominated solutions from union_archive.
    true_pareto = []
    for sol, obj in union_archive:
        if not any(dominates(other_obj, obj) for _, other_obj in union_archive if not np.array_equal(other_obj, obj)):
            true_pareto.append(obj)
    true_pareto = np.array(true_pareto)
    
    # Compute generational distance for each algorithm.
    gd_results = {"MOHHO": [], "PSO": [], "MOACO": []}
    for alg in ["MOHHO", "PSO", "MOACO"]:
        for archive in archives_all[alg]:
            if archive and len(true_pareto) > 0:
                gd = compute_generational_distance(archive, true_pareto)
            else:
                gd = None
            gd_results[alg].append(gd)
    results["Generational_Distance"] = gd_results
    return results, archives_all

# =============================================================================
# ------------------------- Main Comparison -------------------------------
# =============================================================================

if __name__ == '__main__':
    runs = 10  # Number of independent runs
    results, archives_all = run_experiments(runs=runs)
    
    # Save results to a JSON file.
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot boxplots for best makespan.
    plot_convergence({alg: results[alg]["best_makespan"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Best Makespan (hours)")
    
    # Plot hypervolume distribution.
    plot_convergence({alg: results[alg]["hypervolume"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Hypervolume")
    
    # Plot spread distribution.
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Spread (Diversity)")
    
    # Plot generational distance distribution.
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    
    # Combined 2D Pareto Front (from the last run of each algorithm)
    last_archives = [archives_all[alg][-1] for alg in ["MOHHO", "PSO", "MOACO"]]
    plot_pareto_2d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'])
    
    # Combined 3D Pareto Front Comparison.
    plot_pareto_3d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'])
    
    # Display one sample baseline Gantt chart.
    baseline_x = ((np.array([task["min"] for task in tasks]) + np.array([task["max"] for task in tasks])) / 2.0)
    baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks, workers)
    plot_gantt(baseline_schedule, f"Baseline Schedule\nMakespan: {baseline_makespan:.2f} hrs")
    
    print("Experiment complete. Results saved to 'experiment_results.json'.")

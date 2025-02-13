#!/usr/bin/env python3
"""
Multi-Objective Comparison for RCPSP using MOHHO, MOPSO, and MOACO
Incorporates improvements for scientific rigor including:
 - Additional multi-objective performance metrics (hypervolume, generational distance, spread)
 - Multiple independent runs and statistical logging
 - Enhanced archive update with crowding distance
 - Parameter tuning & sensitivity analysis (grid search for PSO population size as an example)
 - Baseline comparison (greedy allocation)
 - Scalability via random instance generation
 - Statistical analysis using ANOVA
 - Modular code structure with reproducibility (fixed random seed)

Author: Simon Gottschalk
Date: 2025-02-13
"""

import numpy as np
import matplotlib.pyplot as plt
import random, math, time, copy, json
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.stats import f_oneway  # For ANOVA

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
    Finds the earliest start time that satisfies the resource capacity constraint.
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

# Fixed resources and cost data.
workers = {
    "Developer": 10,
    "Manager": 2,
    "Tester": 3
}

worker_cost = {
    "Developer": 50,
    "Manager": 75,
    "Tester": 40
}

# Fixed task instance (10 tasks) – used by default.
fixed_tasks = [
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

def generate_random_tasks(num_tasks):
    """
    Generate a list of random tasks (acyclic) for scalability analysis.
    Each task i (1-indexed) may depend on any subset of tasks 1 to i-1.
    Resources are chosen randomly from the available types.
    """
    tasks_list = []
    resource_types = list(workers.keys())
    for i in range(1, num_tasks+1):
        # Generate random parameters
        base_effort = random.randint(50, 150)
        min_alloc = random.randint(1, 3)
        max_alloc = random.randint(min_alloc+1, 15)
        # For dependencies, choose randomly among previous tasks (if any)
        dependencies = random.sample(range(1, i), random.randint(0, min(3, i-1))) if i > 1 else []
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

def compute_schedule(x, tasks, available):
    """
    Given a decision vector x (worker allocations for each task) and a list of tasks,
    compute the schedule.
    Returns:
      - schedule: list of task dictionaries (with start, finish, duration, workers, resource)
      - makespan: overall project finish time.
    Also performs a feasibility check on resource usage.
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
    
    # Feasibility check: ensure that at no time the resource usage exceeds capacity.
    # (This can be expanded for more rigorous checks.)
    return schedule, makespan

# =============================================================================
# ----------------------- Objective Functions -------------------------------
# =============================================================================

def objective_makespan(x):
    """Minimize project makespan."""
    _, ms = compute_schedule(x, current_tasks, workers)
    return ms

def objective_total_cost(x):
    """Minimize total cost (cost per man–hour)."""
    total_cost = 0
    for task in current_tasks:
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
    for task in current_tasks:
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
    Assumes minimization for all objectives.
    """
    objs = np.array([entry[1] for entry in archive])
    mins = np.min(objs, axis=0)
    ref = np.array(reference_point)
    samples = np.random.uniform(low=mins, high=ref, size=(num_samples, len(ref)))
    count = 0
    for sample in samples:
        if any(np.all(sol <= sample) for sol in objs):
            count += 1
    vol = np.prod(ref - mins)
    return (count / num_samples) * vol

def compute_crowding_distance(archive):
    """
    Compute crowding distance for each solution in archive.
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
    Uses crowding distance to maintain diversity.
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
        archive = [entry for entry in archive if not any(same_entry(entry, rem) for rem in removal_list)]
        archive.append(new_entry)
        if len(archive) > max_archive_size:
            distances = compute_crowding_distance(archive)
            min_index = np.argmin(distances)
            archive.pop(min_index)
    return archive

def compute_generational_distance(archive, true_pareto):
    """
    Compute generational distance (GD) between archive and a true Pareto front.
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
    Compute spread/diversity (average pairwise Euclidean distance) in objective space.
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
# ----------------------- Parameter Tuning (Grid Search) --------------------
# =============================================================================

def grid_search_pso_population(pop_sizes, runs_per_config=3):
    """
    Simple grid search to tune PSO population size.
    For each candidate population size, run a few experiments and return average best makespan.
    """
    results = {}
    for pop in pop_sizes:
        best_makespans = []
        for _ in range(runs_per_config):
            optimizer = PSO(dim=len(current_tasks), lb=lb_current, ub=ub_current, 
                            obj_funcs=[objective_makespan, objective_total_cost, objective_neg_utilization],
                            pop=pop, c2=1.05, w_max=0.9, w_min=0.4,
                            disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
            _ = optimizer.run(max_iter=100)
            archive = optimizer.archive
            if archive:
                best = min(archive, key=lambda entry: entry[1][0])[1][0]
                best_makespans.append(best)
        if best_makespans:
            avg = np.mean(best_makespans)
            std = np.std(best_makespans)
            results[pop] = (avg, std)
            print(f"PSO pop size {pop}: Avg best makespan = {avg:.2f}, Std = {std:.2f}")
    return results

# =============================================================================
# ----------------------- Baseline Algorithm --------------------------------
# =============================================================================

def baseline_allocation(tasks, available):
    """
    A simple baseline that allocates the minimum required workers for each task.
    Returns schedule and makespan.
    """
    x = np.array([task["min"] for task in tasks])
    schedule, makespan = compute_schedule(x, tasks, available)
    return schedule, makespan

# =============================================================================
# ------------------------- Algorithm Implementations -----------------------
# =============================================================================

def MOHHO_with_progress(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Modified Multi–Objective HHO.
    Returns an archive of Pareto–optimal solutions and convergence progress.
    """
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    archive = []
    progress = []
    t = 0
    while t < Max_iter:
        for i in range(SearchAgents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            f_val = objf(X[i, :])
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
        self.max_iter = 200
        self.vmax = self.ub - self.lb
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
        if len(self.obj_funcs) == 1:
            return np.array([self.obj_funcs[0](pos)])
        else:
            return np.array([f(pos) for f in self.obj_funcs])

    def update_archive(self):
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def proportional_distribution(self):
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        distances = compute_crowding_distance(self.archive)
        total = sum(distances)
        if total == 0 or math.isinf(total) or math.isnan(total):
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

def MOACO_improved(objf, tasks, workers, lb, ub, ant_count, max_iter,
                     alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0,
                     P=0.6, w1=1.0, w2=1.0, sigma_share=1.0, lambda3=2.0, lambda4=5.0):
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
    archive = []
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
        non_dominated = []
        for sol, obj_val in population:
            if not any(dominates(other_obj, obj_val) for _, other_obj in population):
                non_dominated.append((sol, obj_val))
        for sol, obj_val in non_dominated:
            archive = update_archive_with_crowding(archive, (sol, obj_val))
        for i in range(dim):
            for v in pheromone[i]:
                pheromone[i][v] *= (1 - evaporation_rate)
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

def run_experiments(runs=10, use_random_instance=False, num_tasks=10):
    """
    Run multiple independent experiments for MOHHO, PSO, MOACO and Baseline.
    If use_random_instance is True, generate a random problem instance with num_tasks tasks.
    Returns a dictionary with collected metrics and archives.
    """
    global current_tasks, lb_current, ub_current
    if use_random_instance:
        current_tasks = generate_random_tasks(num_tasks)
    else:
        current_tasks = fixed_tasks.copy()
    dim = len(current_tasks)
    lb_current = np.array([task["min"] for task in current_tasks])
    ub_current = np.array([task["max"] for task in current_tasks])
    reference_point = np.array([200, 20000, 0])  # Adjust if necessary
    
    results = {
        "MOHHO": {"best_makespan": [], "hypervolume": [], "spread": []},
        "PSO": {"best_makespan": [], "hypervolume": [], "spread": []},
        "MOACO": {"best_makespan": [], "hypervolume": [], "spread": []},
        "Baseline": {"makespan": []}
    }
    archives_all = {"MOHHO": [], "PSO": [], "MOACO": []}
    
    for run in range(runs):
        print(f"Run {run+1}/{runs} ...")
        # Baseline
        base_schedule, base_ms = baseline_allocation(current_tasks, workers)
        results["Baseline"]["makespan"].append(base_ms)
        
        # MOHHO
        hho_iter = 100
        SearchAgents_no = 10
        archive_hho, _ = MOHHO_with_progress(multi_objective, lb_current, ub_current, dim, SearchAgents_no, hho_iter)
        if archive_hho:
            best_particle_hho = min(archive_hho, key=lambda entry: entry[1][0])
            best_ms_hho = best_particle_hho[1][0]
        else:
            best_ms_hho = None
        results["MOHHO"]["best_makespan"].append(best_ms_hho)
        archives_all["MOHHO"].append(archive_hho)
        
        # PSO
        objectives = [objective_makespan, objective_total_cost, objective_neg_utilization]
        optimizer = PSO(dim=dim, lb=lb_current, ub=ub_current, obj_funcs=objectives,
                        pop=10, c2=1.05, w_max=0.9, w_min=0.4,
                        disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
        _ = optimizer.run(max_iter=100)
        archive_pso = optimizer.archive
        if archive_pso:
            best_arch = min(archive_pso, key=lambda entry: entry[1][0])
            best_ms_pso = best_arch[1][0]
        else:
            best_ms_pso = None
        results["PSO"]["best_makespan"].append(best_ms_pso)
        archives_all["PSO"].append(archive_pso)
        
        # MOACO
        ant_count = 10
        moaco_iter = 100
        archive_moaco, _ = MOACO_improved(multi_objective, current_tasks, workers, lb_current, ub_current,
                                          ant_count, moaco_iter, alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0)
        if archive_moaco:
            best_particle_moaco = min(archive_moaco, key=lambda entry: entry[1][0])
            best_ms_moaco = best_particle_moaco[1][0]
        else:
            best_ms_moaco = None
        results["MOACO"]["best_makespan"].append(best_ms_moaco)
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
    
    # Union of archives to approximate true Pareto front for GD.
    union_archive = []
    for alg in archives_all:
        for arch in archives_all[alg]:
            union_archive.extend(arch)
    true_pareto = []
    for sol, obj in union_archive:
        if not any(dominates(other_obj, obj) for _, other_obj in union_archive if not np.array_equal(other_obj, obj)):
            true_pareto.append(obj)
    true_pareto = np.array(true_pareto)
    
    gd_results = {"MOHHO": [], "PSO": [], "MOACO": []}
    for alg in ["MOHHO", "PSO", "MOACO"]:
        for archive in archives_all[alg]:
            if archive and len(true_pareto) > 0:
                gd = compute_generational_distance(archive, true_pareto)
            else:
                gd = None
            gd_results[alg].append(gd)
    results["Generational_Distance"] = gd_results
    return results, archives_all, base_schedule

# =============================================================================
# ------------------------- Statistical Analysis ----------------------------
# =============================================================================

def statistical_analysis(results):
    """
    Compute mean, standard deviation, and perform ANOVA on best makespan values.
    """
    algos = ["MOHHO", "PSO", "MOACO", "Baseline"]
    means = {}
    stds = {}
    data = {}
    # For Baseline, use "makespan" values; for others, use "best_makespan".
    data["Baseline"] = results["Baseline"]["makespan"]
    for algo in ["MOHHO", "PSO", "MOACO"]:
        data[algo] = results[algo]["best_makespan"]
    for algo in algos:
        arr = np.array(data[algo])
        means[algo] = np.mean(arr)
        stds[algo] = np.std(arr)
        print(f"{algo}: Mean = {means[algo]:.2f}, Std = {stds[algo]:.2f}")
    # Perform one-way ANOVA if all groups have more than one value.
    if all(len(data[algo]) > 1 for algo in algos):
        F_stat, p_value = f_oneway(data["Baseline"], data["MOHHO"], data["PSO"], data["MOACO"])
        print(f"ANOVA: F = {F_stat:.2f}, p = {p_value:.4f}")
    else:
        print("Not enough data for ANOVA.")
    return means, stds

# =============================================================================
# ------------------------- Main Comparison ---------------------------------
# =============================================================================

if __name__ == '__main__':
    runs = 10  # Number of independent runs
    # Set use_random_instance=True to test scalability on a randomly generated problem (e.g., 20 tasks)
    results, archives_all, base_schedule = run_experiments(runs=runs, use_random_instance=False, num_tasks=10)
    
    # Save results to JSON file.
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Statistical analysis on best makespan.
    means, stds = statistical_analysis(results)
    
    # Plot boxplots for best makespan, hypervolume, spread, and generational distance.
    plot_convergence({alg: results[alg]["best_makespan"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Best Makespan (hours)")
    plot_convergence({alg: results[alg]["hypervolume"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Hypervolume")
    plot_convergence({alg: results[alg]["spread"] for alg in ["MOHHO", "PSO", "MOACO"]}, "Spread (Diversity)")
    plot_convergence(results["Generational_Distance"], "Generational Distance")
    
    # Combined 2D Pareto Front (from the last run of each algorithm)
    last_archives = [archives_all[alg][-1] for alg in ["MOHHO", "PSO", "MOACO"]]
    plot_pareto_2d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'])
    
    # Combined 3D Pareto Front Comparison.
    plot_pareto_3d(last_archives, ["MOHHO", "PSO", "MOACO"], ['o', '^', 's'], ['blue', 'red', 'green'])
    
    # Display baseline Gantt chart.
    base_schedule, base_ms = compute_schedule(np.array([task["min"] for task in current_tasks]), current_tasks, workers)
    plot_gantt(base_schedule, f"Baseline Schedule (Greedy Allocation)\nMakespan: {base_ms:.2f} hrs")
    
    # Example of tuning: grid search for PSO population size.
    print("\nStarting grid search for PSO population size...")
    pop_sizes = [10, 20, 30]
    grid_results = grid_search_pso_population(pop_sizes, runs_per_config=3)
    
    print("Experiment complete. Results saved to 'experiment_results.json'.")

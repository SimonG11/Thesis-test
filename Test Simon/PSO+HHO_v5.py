#!/usr/bin/env python3
"""
Comparison of Multi–Objective PSO (Improved) and HHO on a 10–Task Project Scheduling Problem.
This script:
  • Implements a modified multi–objective HHO (MOHHO_with_progress) that records the best makespan.
  • Implements an improved multi–objective PSO variant (renamed here as PSO) that uses proportional distribution,
    jump–improved and disturbance operations.
  • Defines a scheduling problem where each decision vector specifies worker allocations.
  • Defines three objectives: makespan, total cost, and negative average utilization.
  • Integrates resource constraints – each task requires a specific worker type and the total
    number of workers allocated (per type) in overlapping tasks cannot exceed the available amount.
  • Runs both algorithms for a fixed number of iterations.
  • Plots convergence curves, Pareto fronts (in objective space), and Gantt charts for the best schedules.
  • Additionally, compares the Pareto front analyses via a combined 2D plot and a comparative 3D scatter plot.
  
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
        # Gather event times in the candidate window (the window endpoints and any task start/finish inside)
        events = [candidate, candidate + duration]
        for task in scheduled_tasks:
            # Only consider tasks that use the same resource
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
        # Check each sub-interval (using midpoints) for resource usage.
        for i in range(len(events)-1):
            mid = (events[i] + events[i+1]) / 2.0
            # Sum up workers from all tasks (of the same resource type) active at time mid.
            usage = sum(task["workers"] for task in scheduled_tasks 
                        if task.get("resource") == resource and task["start"] <= mid < task["finish"])
            if usage + allocated > capacity:
                feasible = False
                # Move candidate forward to the next event time where resource usage might drop.
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

# Define 10 tasks – each task has a base effort, minimum/maximum worker allocations, dependencies,
# and the type of worker required.
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
      - For each task, clamps the allocation between its [min, effective_max],
        where effective_max = min(task["max"], available for the task's resource).
      - Computes an adjusted effort and duration.
      - Determines the earliest feasible start time (after dependencies) that respects
        resource constraints for the required worker type, using the helper function `find_earliest_start`.
    Returns:
      - schedule: list of dictionaries (one per task, including resource type)
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
        # Find the earliest start time that satisfies dependency and resource constraints.
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
    """Minimize total cost (wage_rate = 50 per man–hour)."""
    wage_rate = 50
    total_cost = 0
    for task in tasks:
        tid = task["id"]
        resource_type = task["resource"]
        capacity = workers[resource_type]
        effective_max = min(task["max"], capacity)
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(effective_max, alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
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
        alloc = int(round(x[tid - 1]))
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
# ----------------------- HHO Implementation -----------------------------
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
# ---------------------------- Main Comparison ------------------------------
# =============================================================================

if __name__ == '__main__':
    # Define decision space: one worker allocation per task
    dim = len(tasks)
    lb = np.array([task["min"] for task in tasks])
    ub = np.array([task["max"] for task in tasks])
    
    # Compute a baseline schedule (using midpoint allocation) under resource constraints.
    baseline_x = (lb + ub) / 2.0
    baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks, workers)
    print("Baseline Makespan (hours):", baseline_makespan)
    plot_gantt(baseline_schedule, "Baseline Schedule")
    
    # ----------------------- Run HHO -----------------------
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
    
    # ----------------------- Run Improved PSO (renamed as PSO) -----------------------
    objectives = [objective_makespan, objective_total_cost, objective_neg_utilization]
    optimizer = PSO(dim=dim, lb=lb, ub=ub, obj_funcs=objectives,
                    pop=10, c2=1.05, w_max=0.9, w_min=0.4,
                    disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
    pso_progress = optimizer.run(max_iter=100)
    archive_pso = optimizer.archive
    print(f"Number of archive (non-dominated) solutions found by PSO: {len(archive_pso)}")
    best_arch = min(archive_pso, key=lambda entry: entry[1][0])
    best_solution_pso = best_arch[0]
    best_schedule_pso, best_makespan_pso = compute_schedule(best_solution_pso, tasks, workers)
    print("\n[PSO] Best Makespan (hours):", best_makespan_pso)
    
    # ----------------- Convergence Comparison Plot -----------------
    plt.figure(figsize=(10,6))
    plt.plot(range(hho_iterations), hho_progress, label="HHO", linewidth=2)
    plt.plot(range(len(pso_progress)), pso_progress, label="PSO", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Makespan (hours)")
    plt.title("Convergence Comparison: HHO vs. PSO")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ----------------- Individual Pareto Front Plots -----------------
    # For HHO:
    if archive_hho:
        hho_objs = np.array([entry[1] for entry in archive_hho])
        hho_makespans = hho_objs[:, 0]
        hho_costs = hho_objs[:, 1]
        hho_utils = -hho_objs[:, 2]  # Convert negative utilization to average utilization
        plt.figure(figsize=(8,6))
        sc = plt.scatter(hho_makespans, hho_costs, c=hho_utils, cmap='viridis', s=80, edgecolor='k')
        plt.xlabel("Makespan (hours)")
        plt.ylabel("Total Cost")
        plt.title("HHO Pareto Front")
        cbar = plt.colorbar(sc)
        cbar.set_label("Avg Utilization")
        plt.grid(True)
        plt.show()
    
    # For PSO:
    if archive_pso:
        pso_objs = np.array([entry[1] for entry in archive_pso])
        pso_makespans = pso_objs[:, 0]
        pso_costs = pso_objs[:, 1]
        pso_utils = -pso_objs[:, 2]
        plt.figure(figsize=(8,6))
        sc = plt.scatter(pso_makespans, pso_costs, c=pso_utils, cmap='viridis', s=80, edgecolor='k')
        plt.xlabel("Makespan (hours)")
        plt.ylabel("Total Cost")
        plt.title("PSO Pareto Front")
        cbar = plt.colorbar(sc)
        cbar.set_label("Avg Utilization")
        plt.grid(True)
        plt.show()
    
    # ----------------- Combined Pareto Front Comparison (2D & 3D) -----------------
    if archive_hho and archive_pso:
        # (Re)extract HHO objectives
        hho_objs = np.array([entry[1] for entry in archive_hho])
        hho_makespans = hho_objs[:, 0]
        hho_costs = hho_objs[:, 1]
        hho_utils = -hho_objs[:, 2]
        
        # Extract PSO objectives
        pso_objs = np.array([entry[1] for entry in archive_pso])
        pso_makespans = pso_objs[:, 0]
        pso_costs = pso_objs[:, 1]
        pso_utils = -pso_objs[:, 2]
        
        # Create a combined figure with 2 subplots: one for 2D and one for 3D
        fig = plt.figure(figsize=(16, 7))
        
        # Combined 2D Pareto Front Plot
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(hho_makespans, hho_costs, c='blue', marker='o', s=80, edgecolor='k', label="HHO")
        ax1.scatter(pso_makespans, pso_costs, c='red', marker='^', s=80, edgecolor='k', label="PSO")
        ax1.set_xlabel("Makespan (hours)")
        ax1.set_ylabel("Total Cost")
        ax1.set_title("Combined 2D Pareto Front")
        ax1.legend()
        ax1.grid(True)
        
        # Combined 3D Pareto Front Plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(hho_makespans, hho_costs, hho_utils, c='blue', marker='o', s=80, edgecolor='k', label="HHO")
        ax2.scatter(pso_makespans, pso_costs, pso_utils, c='red', marker='^', s=80, edgecolor='k', label="PSO")
        ax2.set_xlabel("Makespan (hours)")
        ax2.set_ylabel("Total Cost")
        ax2.set_zlabel("Avg Utilization")
        ax2.set_title("Combined 3D Pareto Front")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    # ----------------- Display Final Schedules via Gantt Charts -----------------
    if best_schedule_hho is not None:
        plot_gantt(best_schedule_hho, f"Optimized Schedule (HHO)\nMakespan: {best_makespan_hho:.2f} hrs")
    plot_gantt(best_schedule_pso, f"Optimized Schedule (PSO)\nMakespan: {best_makespan_pso:.2f} hrs")

import numpy as np
import math, random, time
import matplotlib.pyplot as plt

# =============================================================================
# Define a 30-task project schedule (a simplified model)
# =============================================================================
tasks = [
    {"id": 1, "task_name": "Requirements Gathering", "base_effort": 80, "min": 2, "max": 5, "dependencies": []},
    {"id": 2, "task_name": "System Design",          "base_effort": 100, "min": 3, "max": 6, "dependencies": [1]},
    {"id": 3, "task_name": "Module 1 Development",   "base_effort": 150, "min": 3, "max": 7, "dependencies": [2]},
    {"id": 4, "task_name": "Module 2 Development",   "base_effort": 150, "min": 3, "max": 7, "dependencies": [2]},
    {"id": 5, "task_name": "Integration",            "base_effort": 120, "min": 2, "max": 5, "dependencies": [3, 4]},
    {"id": 6, "task_name": "Testing",                "base_effort": 100, "min": 2, "max": 5, "dependencies": [5]},
    {"id": 7, "task_name": "User Acceptance Testing", "base_effort": 80,  "min": 2, "max": 4, "dependencies": [6]},
    {"id": 8, "task_name": "Documentation",          "base_effort": 60,  "min": 1, "max": 3, "dependencies": [2]},
    {"id": 9, "task_name": "Training",               "base_effort": 50,  "min": 1, "max": 3, "dependencies": [7, 8]},
    {"id": 10, "task_name": "Deployment",            "base_effort": 70,  "min": 2, "max": 4, "dependencies": [7, 9]},
    {"id": 11, "task_name": "Post-Deployment Support", "base_effort": 40, "min": 1, "max": 3, "dependencies": [10]},
    {"id": 12, "task_name": "Project Review",        "base_effort": 30,  "min": 1, "max": 2, "dependencies": [11]},
    {"id": 13, "task_name": "Final Report",          "base_effort": 20,  "min": 1, "max": 2, "dependencies": [12]},
    {"id": 14, "task_name": "Client Feedback",       "base_effort": 25,  "min": 1, "max": 2, "dependencies": [13]},
    {"id": 15, "task_name": "Project Closure",       "base_effort": 15,  "min": 1, "max": 2, "dependencies": [14]},
    {"id": 16, "task_name": "Market Analysis",       "base_effort": 90,  "min": 2, "max": 5, "dependencies": []},
    {"id": 17, "task_name": "Feasibility Study",     "base_effort": 110, "min": 3, "max": 6, "dependencies": [16]},
    {"id": 18, "task_name": "Prototyping",           "base_effort": 130, "min": 3, "max": 7, "dependencies": [17]},
    {"id": 19, "task_name": "Alpha Testing",         "base_effort": 140, "min": 3, "max": 7, "dependencies": [18]},
    {"id": 20, "task_name": "Beta Testing",          "base_effort": 120, "min": 2, "max": 5, "dependencies": [19]},
    {"id": 21, "task_name": "Launch Preparation",    "base_effort": 100, "min": 2, "max": 5, "dependencies": [20]},
    {"id": 22, "task_name": "Marketing Campaign",    "base_effort": 80,  "min": 2, "max": 4, "dependencies": [21]},
    {"id": 23, "task_name": "Sales Training",        "base_effort": 60,  "min": 1, "max": 3, "dependencies": [22]},
    {"id": 24, "task_name": "Customer Support Setup", "base_effort": 50, "min": 1, "max": 3, "dependencies": [23]},
    {"id": 25, "task_name": "Product Launch",        "base_effort": 70,  "min": 2, "max": 4, "dependencies": [24]},
    {"id": 26, "task_name": "Post-Launch Review",    "base_effort": 40,  "min": 1, "max": 3, "dependencies": [25]},
    {"id": 27, "task_name": "Customer Feedback Analysis", "base_effort": 30, "min": 1, "max": 2, "dependencies": [26]},
    {"id": 28, "task_name": "Product Improvement",   "base_effort": 20,  "min": 1, "max": 2, "dependencies": [27]},
    {"id": 29, "task_name": "Final Product Review",  "base_effort": 25,  "min": 1, "max": 2, "dependencies": [28]},
    {"id": 30, "task_name": "Project Closure Meeting", "base_effort": 15, "min": 1, "max": 2, "dependencies": [29]},
]

# =============================================================================
# Helper: Compute the schedule from a decision vector x (one worker allocation per task)
#
# For each task, we:
#  - Round and clip the worker allocation (x[i]) to [min, max] for that task.
#  - Compute the "adjusted effort" as:
#         new_effort = base_effort * (1 + (1/max) * (workers - 1))
#  - Compute the task duration as new_effort / workers.
#  - Compute the start time as the maximum finish time of its dependencies.
#
# The makespan is the maximum finish time.
# =============================================================================
def compute_schedule(x, tasks):
    schedule = []
    finish_times = {}  # task id -> finish time
    for task in tasks:
        tid = task["id"]
        # Round and clip worker assignment:
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(task["max"], alloc))
        # Adjusted effort and duration:
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
        # Start time: maximum finish time among dependencies (or 0 if none)
        if task["dependencies"]:
            start_time = max(finish_times[dep] for dep in task["dependencies"])
        else:
            start_time = 0
        finish_time = start_time + duration
        finish_times[tid] = finish_time
        schedule.append({
            "task_id": tid,
            "task_name": task["task_name"],
            "start": start_time,
            "finish": finish_time,
            "duration": duration,
            "workers": alloc
        })
    makespan = max(item["finish"] for item in schedule)
    return schedule, makespan

# =============================================================================
# Multi-objective evaluation function
#
# Returns a vector of three objectives:
#   1. Makespan (time) to be minimized.
#   2. Total Cost (assuming a constant wage_rate per man-hour) to be minimized.
#   3. Negative average resource utilization (so that minimizing this is equivalent
#      to maximizing the actual average utilization).
# =============================================================================
def multi_objective(x):
    schedule, makespan = compute_schedule(x, tasks)
    wage_rate = 50  # Assume a constant cost per man-hour
    total_cost = 0
    utilizations = []
    for task in tasks:
        tid = task["id"]
        # Use the same rounding and clipping as in compute_schedule:
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(task["max"], alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
        total_cost += duration * alloc * wage_rate
        utilizations.append(alloc / task["max"])
    avg_util = np.mean(utilizations)
    # We wish to minimize time and cost, and maximize utilization (thus minimize -avg_util)
    return np.array([makespan, total_cost, -avg_util])

# =============================================================================
# Pareto Dominance: In a minimization setting, solution a dominates b if
# all objective values of a are less than or equal to those of b and at least one is strictly less.
# =============================================================================
def dominates(obj_a, obj_b):
    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

# =============================================================================
# Levy flight function (used in HHO)
# =============================================================================
def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) / 
             (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2))) ** (1/beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.power(np.abs(v), 1/beta))
    return step

# =============================================================================
# Multi-objective Harris Hawks Optimization (MOHHO)
#
# This algorithm is similar in spirit to the original HHO but it:
#   - Evaluates each solution on a vector of objectives.
#   - Maintains an external archive of non-dominated (Pareto optimal) solutions.
#   - Uses a randomly selected archive member (the "rabbit") to guide the update.
#
# For demonstration purposes, we use a simple scalarization (via the Euclidean norm)
# in some of the move comparisons, but the Pareto archive is used to store the true fronts.
# =============================================================================
def MOHHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Initialize hawks randomly within the bounds.
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    archive = []  # Will store tuples: (solution vector, objective vector)
    convergence_history = []  # For example, store the archive size at each iteration
    
    t = 0
    while t < Max_iter:
        # --- Evaluate population and update archive ---
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
                # Instead of calling archive.remove(rem), rebuild the archive filtering out dominated entries.
                new_archive = []
                for entry in archive:
                    should_remove = False
                    for rem in removal_list:
                        # Compare both the solution and objective vector using np.array_equal.
                        if np.array_equal(entry[0], rem[0]) and np.array_equal(entry[1], rem[1]):
                            should_remove = True
                            break
                    if not should_remove:
                        new_archive.append(entry)
                archive = new_archive
                archive.append((X[i, :].copy(), f_val.copy()))
        
        # --- Select a leader ("rabbit") randomly from the archive ---
        if archive:
            rabbit = random.choice(archive)[0]
        else:
            rabbit = X[0, :].copy()
        
        E1 = 2 * (1 - (t / Max_iter))  # Decreasing factor
        
        # --- Update positions of hawks ---
        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            if abs(Escaping_Energy) >= 1:  # Exploration phase
                q = random.random()
                rand_index = random.randint(0, SearchAgents_no - 1)
                X_rand = X[rand_index, :].copy()
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * np.abs(X_rand - 2 * random.random() * X[i, :])
                else:
                    X[i, :] = (rabbit - np.mean(X, axis=0)) - random.random() * ((ub - lb) * random.random() + lb)
            else:  # Exploitation phase
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
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :]) + \
                             np.multiply(np.random.randn(dim), Levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0)) + \
                             np.multiply(np.random.randn(dim), Levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
        convergence_history.append(len(archive))
        t += 1
    
    return archive, convergence_history


# =============================================================================
# Visualization: Plot a Gantt chart for a schedule.
# =============================================================================
def plot_gantt(schedule, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks = []
    yticklabels = []
    for i, task in enumerate(schedule):
        ax.broken_barh([(task["start"], task["duration"])],
                       (i * 10, 9),
                       facecolors='tab:blue')
        yticks.append(i * 10 + 5)
        yticklabels.append(f'Task {task["task_id"]}: {task["task_name"]}\n(Workers: {task["workers"]})')
        ax.text(task["start"] + task["duration"] / 2, i * 10 + 5,
                f'{task["start"]:.1f}-{task["finish"]:.1f}',
                ha='center', va='center', color='white', fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Tasks")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Script: Run MOHHO and display the Pareto Front analysis.
# =============================================================================
if __name__ == '__main__':
    # Decision space: one worker allocation per task (dimension = number of tasks)
    dim = len(tasks)
    lb = np.array([task["min"] for task in tasks])
    ub = np.array([task["max"] for task in tasks])
    
    # For an initial baseline schedule we use the midpoint of [min, max] for each task.
    baseline_x = (lb + ub) / 2.0
    baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks)
    print("Baseline makespan (hours):", baseline_makespan)
    plot_gantt(baseline_schedule, "Baseline Schedule")
    
    # Set MOHHO parameters
    SearchAgents_no = 30
    Max_iter = 50
    
    # Run the Multi-objective HHO
    pareto_archive, convergence = MOHHO(multi_objective, lb, ub, dim, SearchAgents_no, Max_iter)
    
    # Extract objective values for Pareto analysis:
    pareto_objs = np.array([f for (_, f) in pareto_archive])
    makespans = pareto_objs[:, 0]
    costs = pareto_objs[:, 1]
    avg_utils = -pareto_objs[:, 2]  # (convert back to positive utilization)
    
    # Display Pareto solutions:
    print("\nPareto Front (non-dominated solutions):")
    for idx, (sol, f_val) in enumerate(pareto_archive):
        print(f"Solution {idx+1}: Workers = {np.round(sol, 0)}, Makespan = {f_val[0]:.2f}, "
              f"Cost = {f_val[1]:.2f}, Avg Utilization = {(-f_val[2]):.2f}")
    
    # -----------------------------------------------------------------------------
    # Plot the Pareto front: makespan vs. cost, with color indicating average utilization.
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(makespans, costs, c=avg_utils, cmap='viridis', s=80, edgecolor='k')
    plt.xlabel("Project Makespan (hours)")
    plt.ylabel("Total Cost")
    plt.title("Pareto Front: Makespan vs. Cost\n(Color ~ Average Resource Utilization)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Average Utilization (0-1)")
    plt.grid(True)
    plt.show()
    
    # Optionally, pick one Pareto solution to display its detailed schedule.
    # (Here we select the solution with the lowest makespan.)
    best_index = np.argmin(makespans)
    best_solution = pareto_archive[best_index][0]
    best_schedule, best_makespan = compute_schedule(best_solution, tasks)
    print("\nSelected schedule from Pareto front (lowest makespan):")
    print("Makespan (hours):", best_makespan)
    plot_gantt(best_schedule, "Optimized Schedule (Selected Pareto Solution)")

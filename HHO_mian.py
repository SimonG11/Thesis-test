import numpy as np
import math, random, time
import matplotlib.pyplot as plt

# ======================
# Define the 10-task schedule data
# ======================
tasks = [
    {"id": 1, "task_name": "Requirements Gathering", "base_effort": 80, "min": 2, "max": 5, "dependencies": []},
    {"id": 2, "task_name": "System Design", "base_effort": 100, "min": 3, "max": 6, "dependencies": [1]},
    {"id": 3, "task_name": "Module 1 Development", "base_effort": 150, "min": 3, "max": 7, "dependencies": [2]},
    {"id": 4, "task_name": "Module 2 Development", "base_effort": 150, "min": 3, "max": 7, "dependencies": [2]},
    {"id": 5, "task_name": "Integration", "base_effort": 120, "min": 2, "max": 5, "dependencies": [3,4]},
    {"id": 6, "task_name": "Testing", "base_effort": 100, "min": 2, "max": 5, "dependencies": [5]},
    {"id": 7, "task_name": "User Acceptance Testing", "base_effort": 80, "min": 2, "max": 4, "dependencies": [6]},
    {"id": 8, "task_name": "Documentation", "base_effort": 60, "min": 1, "max": 3, "dependencies": [2]},
    {"id": 9, "task_name": "Training", "base_effort": 50, "min": 1, "max": 3, "dependencies": [7,8]},
    {"id": 10, "task_name": "Deployment", "base_effort": 70, "min": 2, "max": 4, "dependencies": [7,9]}
]

# ======================
# Helper: Compute schedule timing given a vector of worker assignments (one per task)
# ======================
def compute_schedule(x, tasks):
    """
    x: decision vector of dimension 10; each element is the (continuous) worker assignment
       for the corresponding task. We round and clip x to the task limits.
    Returns a list with schedule info for each task and the overall project makespan.
    """
    schedule = []
    finish_times = {}  # map task id -> finish time
    for task in tasks:
        tid = task["id"]
        # Get the allocated workers (round and enforce [min, max])
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(task["max"], alloc))
        # Compute the adjusted effort and duration.
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
        # Start time is the maximum finish time among dependencies (or 0 if none)
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

# ======================
# Define the objective (cost) function: minimize makespan + penalty (if too many tasks run concurrently)
# ======================
def schedule_cost(x):
    schedule, makespan = compute_schedule(x, tasks)
    # Compute maximum number of concurrent tasks (simple event-based overlap check)
    events = []
    for item in schedule:
        events.append((item["start"], +1))
        events.append((item["finish"], -1))
    events.sort(key=lambda e: e[0])
    concurrency = 0
    max_concurrency = 0
    for t, delta in events:
        concurrency += delta
        max_concurrency = max(max_concurrency, concurrency)
    # If more than 5 tasks are active concurrently, add a penalty.
    penalty = 0
    max_parallel = 5
    if max_concurrency > max_parallel:
        penalty = (max_concurrency - max_parallel) * 1000  # penalty factor
    return makespan + penalty

# ======================
# Visualization: Plot a simple Gantt chart for the schedule
# ======================
def plot_gantt(schedule, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks = []
    yticklabels = []
    for i, task in enumerate(schedule):
        ax.broken_barh([(task["start"], task["duration"])],
                       (i*10, 9),
                       facecolors='tab:blue')
        yticks.append(i*10 + 5)
        yticklabels.append(f'Task {task["task_id"]}: {task["task_name"]}\n(Workers: {task["workers"]})')
        ax.text(task["start"] + task["duration"]/2, i*10 + 5,
                f'{task["start"]:.1f}-{task["finish"]:.1f}',
                ha='center', va='center', color='white', fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Tasks")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ======================
# Harris Hawks Optimization (HHO) implementation (adapted from the research code)
# ======================
# (For brevity, we include the main parts; note that many details are taken from the research.)
def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2) /
             (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)) )**(1/beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.power(np.abs(v), 1/beta))
    return step

class solution:
    pass

def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Initialize the locations of hawks
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    convergence_curve = np.zeros(Max_iter)
    
    # Initialize the “rabbit” (best solution) record:
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float("inf")
    
    t = 0
    timerStart = time.time()
    s = solution()
    
    while t < Max_iter:
        for i in range(SearchAgents_no):
            # Check and correct boundaries
            X[i, :] = np.clip(X[i, :], lb, ub)
            fitness = objf(X[i, :])
            if fitness < Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()
        E1 = 2 * (1 - (t / Max_iter))  # decreasing energy
        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            if abs(Escaping_Energy) >= 1:  # Exploration phase
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :].copy()
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * abs(X_rand - 2 * random.random() * X[i, :])
                else:
                    X[i, :] = (Rabbit_Location - X.mean(axis=0)) - random.random() * ((ub - lb) * random.random() + lb)
            else:  # Exploitation phase
                r = random.random()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - X[i, :])
                if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    if objf(X1) < objf(X[i, :]):
                        X[i, :] = X1.copy()
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :]) + np.multiply(np.random.randn(dim), Levy(dim))
                        if objf(X2) < objf(X[i, :]):
                            X[i, :] = X2.copy()
                if r < 0.5 and abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(axis=0))
                    if objf(X1) < objf(X[i, :]):
                        X[i, :] = X1.copy()
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(axis=0)) + np.multiply(np.random.randn(dim), Levy(dim))
                        if objf(X2) < objf(X[i, :]):
                            X[i, :] = X2.copy()
        convergence_curve[t] = Rabbit_Energy
        # Uncomment the following line to see progress:
        # print(f"Iteration {t}, best cost = {Rabbit_Energy}")
        t += 1
    
    timerEnd = time.time()
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location
    s.convergence = convergence_curve
    s.executionTime = timerEnd - timerStart
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    return s

# ======================
# Main script: Setup, baseline schedule, HHO optimization, and visualization
# ======================
# Decision space: each task’s worker count is our variable.
dim = len(tasks)
# Create vector lower and upper bounds (as arrays)
lb = np.array([task["min"] for task in tasks])
ub = np.array([task["max"] for task in tasks])

# Baseline: let’s use the midpoint (rounded) as the starting worker assignments.
baseline_x = (lb + ub) / 2.0
baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks)
print("Baseline makespan (hours):", baseline_makespan)
plot_gantt(baseline_schedule, "Baseline Schedule")

# Run Harris Hawks Optimization on the schedule cost function
SearchAgents_no = 30
Max_iter = 50

best_solution = HHO(schedule_cost, lb, ub, dim, SearchAgents_no, Max_iter)
print("Optimized makespan (hours):", best_solution.best)
print("Optimized worker assignments (per task):", np.round(best_solution.bestIndividual, 0))

# Compute the optimized schedule from the best solution
optimized_schedule, optimized_makespan = compute_schedule(best_solution.bestIndividual, tasks)
plot_gantt(optimized_schedule, "Optimized Schedule")

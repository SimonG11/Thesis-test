#!/usr/bin/env python3
"""
Comparison of Multi–Objective PSO, HHO and PDJI-MOPSO on a 30–Task Project Scheduling Problem.
This script:
  • Implements a multi–objective PSO (with particle classes and non–dominated sorting).
  • Implements a modified multi–objective HHO (MOHHO_with_progress) that records the best makespan.
  • Implements an improved multi–objective PSO variant (PDJI-MOPSO) with proportional distribution,
    jump–improved and disturbance operations.
  • Defines a scheduling problem where each decision vector specifies worker allocations.
  • Defines three objectives: makespan, total cost, and negative average utilization.
  • Runs all three algorithms for a fixed number of iterations.
  • Plots convergence curves, Pareto fronts (in objective space), and Gantt charts for the best schedules.
  
Author: Simon Gottschalk
Date: 2025-02-11
"""

import numpy as np
import matplotlib.pyplot as plt
import random, math, time, copy

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

# =============================================================================
# -------------------------- Scheduling Problem Definition ------------------
# =============================================================================

# Define 30 tasks – each task has a base effort, minimum/maximum worker allocations, and dependencies.
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
    {"id": 11, "task_name": "Post-Deployment Support", "base_effort": 40,  "min": 1, "max": 3, "dependencies": [10]},
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
    {"id": 24, "task_name": "Customer Support Setup", "base_effort": 50,  "min": 1, "max": 3, "dependencies": [23]},
    {"id": 25, "task_name": "Product Launch",        "base_effort": 70,  "min": 2, "max": 4, "dependencies": [24]},
    {"id": 26, "task_name": "Post-Launch Review",    "base_effort": 40,  "min": 1, "max": 3, "dependencies": [25]},
    {"id": 27, "task_name": "Customer Feedback Analysis", "base_effort": 30, "min": 1, "max": 2, "dependencies": [26]},
    {"id": 28, "task_name": "Product Improvement",   "base_effort": 20,  "min": 1, "max": 2, "dependencies": [27]},
    {"id": 29, "task_name": "Final Product Review",  "base_effort": 25,  "min": 1, "max": 2, "dependencies": [28]},
    {"id": 30, "task_name": "Project Closure Meeting", "base_effort": 15, "min": 1, "max": 2, "dependencies": [29]},
]

def compute_schedule(x, tasks):
    """
    Given a decision vector x (worker allocations for each task), compute:
      - Start and finish times based on dependencies.
      - Duration computed from an adjusted effort.
    Returns:
      - schedule: list of dictionaries (one per task)
      - makespan: overall project finish time.
    """
    schedule = []
    finish_times = {}
    for task in tasks:
        tid = task["id"]
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(task["max"], alloc))
        new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (alloc - 1))
        duration = new_effort / alloc
        start_time = max([finish_times[dep] for dep in task["dependencies"]]) if task["dependencies"] else 0
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
# ----------------------- Objective Functions -------------------------------
# =============================================================================

def objective_makespan(x):
    """Minimize project makespan."""
    _, ms = compute_schedule(x, tasks)
    return ms

def objective_total_cost(x):
    """Minimize total cost (wage_rate = 50 per man–hour)."""
    wage_rate = 50
    total_cost = 0
    for task in tasks:
        tid = task["id"]
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(task["max"], alloc))
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
        alloc = int(round(x[tid - 1]))
        alloc = max(task["min"], min(task["max"], alloc))
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
        yticklabels.append(f"Task {task['task_id']}: {task['task_name']}\n(Workers: {task['workers']})")
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
# ------------------------- PSO Implementation (Original) -------------------
# =============================================================================

class particle_single():
    """
    Particle for single–objective optimization.
    (Also used as the base for multi–objective particles.)
    """
    def __init__(self, obj_func, attribute_number, constr=[], 
                 vmax=np.array(np.nan), l_bound=np.array(np.nan), u_bound=np.array(np.nan),
                 integer=np.array(np.nan), position=np.array(np.nan), velocity=np.array(np.nan)):
        self.obj_function = obj_func
        if type(constr) != list:
            constr = [constr]
        self.constraints = constr
        self.att = attribute_number
        
        # Initialize position
        if np.all(np.isnan(position)) == False:
            self.position = position
        else:
            try:
                init_pos = []
                for i in range(self.att):
                    if integer[i] == False:
                        init_pos.append(random.uniform(l_bound[i], u_bound[i]))
                    else:
                        init_pos.append(random.randint(l_bound[i], u_bound[i]))
                self.position = np.array(init_pos)
            except Exception as e:
                print('Error initializing position:', e)
        self.obj_value = self.calc_obj_value()
        
        # Initialize velocity
        if np.all(np.isnan(velocity)) == False:
            self.velocity = velocity
        else:
            try:
                self.velocity = np.array([random.uniform(-vmax[i], vmax[i]) for i in range(self.att)])
            except Exception as e:
                print('Error initializing velocity:', e)
        self.best_p = np.nan

    def __repr__(self):
        return f"Particle: pos={self.position}, vel={self.velocity}, obj={self.obj_value}"
    
    def set_position(self, new_pos):
        self.position = new_pos
        self.obj_value = self.calc_obj_value()
        
    def set_velocity(self, new_v):
        self.velocity = new_v
        
    def get_obj_value(self):
        return self.obj_value
    
    def init_p_best(self):
        self.best_p = particle_single(self.obj_function, self.att, constr=self.constraints,
                                       position=self.position.copy(), velocity=self.velocity.copy())
        
    def compare_p_best(self):
        if self.obj_value < self.best_p.obj_value:
            self.best_p.set_position(self.position)
        
    def compare(self, part2):
        return self.obj_value < part2.obj_value
        
    def plot(self, best_p, x_coord, y_coord):
        if best_p:
            plt.plot(self.best_p.position[x_coord], self.best_p.obj_value, 'k.')
        else:
            plt.plot(self.position[x_coord], self.obj_value, 'k.')
        
    def calc_obj_value(self):
        if not self.constraints:
            return self.obj_function(self.position)
        else:
            penalty = sum([con(self.position) for con in self.constraints])
            return self.obj_function(self.position) + penalty

class particle_multi(particle_single):
    """
    Particle for multi–objective optimization.
    In addition to the particle_single attributes, it stores:
      - obj_functions (list), obj_values (vector), and domination info (S, n, rank, crowding distance)
    """
    def __init__(self, obj_func, attribute_number, constr=[], 
                 vmax=np.array(np.nan), l_bound=np.array(np.nan), u_bound=np.array(np.nan),
                 integer=np.array(np.nan), position=np.array(np.nan), velocity=np.array(np.nan)):
        self.obj_functions = obj_func
        self.constraints = constr
        self.att = attribute_number
        
        # Initialize position
        if np.all(np.isnan(position)) == False:
            self.position = position
        else:
            try:
                init_pos = []
                for i in range(self.att):
                    if integer[i] == False:
                        init_pos.append(random.uniform(l_bound[i], u_bound[i]))
                    else:
                        init_pos.append(random.randint(l_bound[i], u_bound[i]))
                self.position = np.array(init_pos)
            except Exception as e:
                print('Error initializing position:', e)
        self.obj_values = self.calc_obj_value()
        
        # Initialize velocity
        if np.all(np.isnan(velocity)) == False:
            self.velocity = velocity
        else:
            try:
                self.velocity = np.array([random.uniform(-vmax[i], vmax[i]) for i in range(self.att)])
            except Exception as e:
                print('Error initializing velocity:', e)
        self.best_p = np.nan
        self.S = []
        self.n = np.nan
        self.rank = np.nan
        self.distance = np.nan

    def __repr__(self):
        return (f"Multi–objective particle: pos={self.position}, vel={self.velocity}, "
                f"obj={self.obj_values}, rank={self.rank}, dist={self.distance}")
        
    def set_position(self, new_pos):
        self.position = new_pos
        self.obj_values = self.calc_obj_value()
    
    def get_obj_value(self):
        return self.obj_values
    
    def init_p_best(self):
        self.best_p = particle_multi(self.obj_functions, self.att, constr=self.constraints,
                                      position=self.position.copy(), velocity=self.velocity.copy())
        self.best_p.rank  = self.rank
        self.best_p.distance = self.distance
        
    def compare_p_best(self):
        if self.compare_rank_dist(self.rank, self.distance, self.best_p.rank, self.best_p.distance):
            self.best_p.set_position(self.position)
            self.best_p.rank = self.rank
            self.best_p.distance = self.distance
        
    def compare(self, part2):
        return self.compare_rank_dist(self.rank, self.distance, part2.rank, part2.distance)
    
    def plot(self, best_p, x_coord, y_coord):
        if best_p:
            if self.best_p.rank == 0:
                plt.plot(self.best_p.obj_values[x_coord], self.best_p.obj_values[y_coord], 'r*')
            else:
                plt.plot(self.best_p.obj_values[x_coord], self.best_p.obj_values[y_coord], 'k.')
        else:
            if self.rank == 0:
                plt.plot(self.obj_values[x_coord], self.obj_values[y_coord], 'r*')
            else:
                plt.plot(self.obj_values[x_coord], self.obj_values[y_coord], 'k.')

    def compare_rank_dist(self, rank_1, distance_1, rank_2, distance_2):
        if rank_1 == rank_2:
            if distance_1 == distance_2:
                return bool(random.randint(0, 1))
            else: 
                return distance_1 > distance_2
        else:
            return rank_1 < rank_2
    
    def dominates(self, part2):
        dom = True
        for i in range(len(self.obj_values)):
            dom = dom and (self.obj_values[i] <= part2.obj_values[i])
        strict = any(self.obj_values[i] < part2.obj_values[i] for i in range(len(self.obj_values)))
        return dom and strict
    
    def calc_obj_value(self):
        if not self.constraints:
            return np.array([func(self.position) for func in self.obj_functions])
        else:
            penalty = sum([con(self.position) for con in self.constraints])
            return np.array([func(self.position) for func in self.obj_functions]) + penalty

class pso:
    """
    Multi–objective Particle Swarm Optimizer.
    """
    def __init__(self, att, l_b, u_b, obj_func, constraints=[], c=2.1304, s=1.0575, w=0.4091, pop=156, vm=np.nan, integer=False):
        if np.isnan(vm).all():
            vm = np.array([u_b[i] - l_b[i] for i in range(att)])
        if not (isinstance(vm, np.ndarray) or isinstance(vm, list)):
            vm = np.array([vm for i in range(att)])
        if len(vm) != att:
            vm = np.append(vm, [vm[len(vm)-1] for i in range(len(l_b), att)])
        if type(integer) != list:
            integer = np.array([integer for i in range(att)])
        self.c_param = c
        self.s_param = s
        self.v_weight = w
        self.l_bound = l_b
        self.u_bound = u_b
        self.integer = integer
        self.vmax = vm
        if type(obj_func) != list:
            self.multi = False
            self.swarm = [particle_single(obj_func, att, constraints, vm, l_b, u_b, integer) for _ in range(pop)]
        else:
            self.multi = True
            self.swarm = [particle_multi(obj_func, att, constraints, vm, l_b, u_b, integer) for _ in range(pop)]
            self.comp_swarm = self.swarm.copy()
        if self.multi:
            self.non_dom_sort()
        for part in self.swarm:
            part.init_p_best()
        self.set_g_best()
        
    def __repr__(self):
        if self.multi:
            return (f"Multi–objective PSO with {len(self.swarm)} particles, "
                    f"{len(self.swarm[0].position)} attributes, and {len(self.swarm[0].obj_functions)} objectives")
        else:
            return (f"Single–objective PSO with {len(self.swarm)} particles and {len(self.swarm[0].position)} attributes")
        
    def non_dom_sort(self):
        # Fast non–dominated sort.
        F = []
        F1 = []
        for p in self.comp_swarm:
            Sp = []
            n_p = 0
            for q in self.comp_swarm:
                if p.dominates(q):
                    Sp.append(q)
                elif q.dominates(p):
                    n_p += 1
            p.S = Sp
            p.n = n_p
            if n_p == 0:
                F1.append(p)
                p.rank = 0
        F.append(F1)
        i = 0
        while F[i]:
            H = []
            for p in F[i]:
                for q in p.S:
                    q.n -= 1
                    if q.n == 0:
                        H.append(q)
                        q.rank = i + 1
            i += 1
            F.append(H)
        F.pop()  # remove empty last front.
        # Compute crowding distance.
        for Fi in F:
            l = len(Fi)
            if l == 0:
                continue
            for parti in Fi:
                parti.distance = 0
            for m in range(len(Fi[0].obj_functions)):
                m_obj = [x.obj_values[m] for x in Fi]
                if l > 1:
                    sorted_indices = np.argsort(m_obj)
                    Fi_sorted = [Fi[j] for j in sorted_indices]
                else:
                    Fi_sorted = Fi
                Fi_sorted[0].distance = np.inf
                Fi_sorted[-1].distance = np.inf
                for j in range(1, l - 1):
                    Fi_sorted[j].distance += (Fi_sorted[j+1].obj_values[m] - Fi_sorted[j-1].obj_values[m])
                    
    def set_g_best(self):
        self.g_best = self.swarm[0]
        for part in self.swarm[1:]:
            if part.compare(self.g_best):
                self.g_best = copy.deepcopy(part)
        
    def plot(self, best_p=True, x_coord=0, y_coord=1):
        for partic in self.swarm:
            partic.plot(best_p, x_coord, y_coord)
        if self.multi:
            plt.xlabel(r'$f_%i(x)$' % x_coord)
            plt.ylabel(r'$f_%i(x)$' % y_coord)
            plt.title('Pareto Front (Current Swarm)')
        else:
            plt.xlabel(r'$x_%i$' % x_coord)
            plt.ylabel(r'$f(x)$')
            plt.title('Objective vs. Decision Variable')
        plt.show()
     
    def moving(self, steps, time_termination):
        t0 = time.time()
        for _ in range(steps):
            if time_termination != -1 and time.time() - t0 > time_termination:
                break
            if self.multi:
                self.comp_swarm = [self.g_best]
            for part in self.swarm:
                r1 = random.random()
                r2 = random.random()
                new_v = (self.v_weight * part.velocity +
                         self.c_param * r1 * (part.best_p.position - part.position) +
                         self.s_param * r2 * (self.g_best.position - part.position))
                new_v = np.array([max(new_v[i], -self.vmax[i]) for i in range(len(new_v))])
                new_v = np.array([min(new_v[i], self.vmax[i]) for i in range(len(new_v))])
                new_p = part.position + new_v
                for i in range(len(new_p)):
                    if self.integer[i]:
                        new_p[i] = int(round(new_p[i]))
                # Enforce bounds (wrap–around if needed)
                new_p = np.array([new_p[i] if new_p[i] > self.l_bound[i]
                                  else self.u_bound[i] - abs(self.l_bound[i]-new_p[i]) % (self.u_bound[i]-self.l_bound[i])
                                  for i in range(len(new_p))])
                new_p = np.array([new_p[i] if new_p[i] < self.u_bound[i]
                                  else self.l_bound[i] + abs(self.u_bound[i]-new_p[i]) % (self.u_bound[i]-self.l_bound[i])
                                  for i in range(len(new_p))])
                part.set_velocity(new_v)
                part.set_position(new_p)
                if self.multi:
                    self.comp_swarm.append(part)
                    self.comp_swarm.append(part.best_p)
            if self.multi:
                self.non_dom_sort()
                self.g_best = copy.deepcopy(self.comp_swarm[0])
                new_swarm = self.comp_swarm[1:-1:2]
                self.swarm = copy.deepcopy(new_swarm)
            for part in self.swarm:
                if self.multi:
                    part.compare_p_best()
                if part.compare(self.g_best):
                    self.g_best = copy.deepcopy(part)
                    
    def get_solution(self, whole_particle=False):
        solution = []
        if self.multi:
            for part in self.swarm:
                if part.rank == 0:
                    solution.append(part if whole_particle else part.get_obj_value())
                if part.best_p.rank == 0 and not np.array_equal(part.position, part.best_p.position):
                    solution.append(part.best_p if whole_particle else part.best_p.get_obj_value())
        else:
            solution = self.g_best if whole_particle else self.g_best.get_obj_value()
        return solution

# =============================================================================
# ----------------------- HHO Implementation -----------------------------
# =============================================================================

def MOHHO_with_progress(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Modified Multi–Objective HHO that:
      - Uses similar operators as standard HHO.
      - Records the best makespan (first objective) at each iteration.
      - Returns an archive of Pareto–optimal solutions and a list of best makespan per iteration.
    """
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    archive = []
    progress = []
    t = 0
    while t < Max_iter:
        # Evaluate each agent and update archive.
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
        # Choose a random leader ("rabbit") from the archive.
        if archive:
            rabbit = random.choice(archive)[0]
        else:
            rabbit = X[0, :].copy()
        E1 = 2 * (1 - (t / Max_iter))
        # Update positions.
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
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :]) + \
                             np.multiply(np.random.randn(dim), levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0)) + \
                             np.multiply(np.random.randn(dim), levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
        best_makespan = np.min([objf(X[i, :])[0] for i in range(SearchAgents_no)])
        progress.append(best_makespan)
        t += 1
    return archive, progress

# =============================================================================
# ------------------------- PDJI-MOPSO Class --------------------------------
# =============================================================================

class PDJI_MOPSO:
    def __init__(self, dim, lb, ub, obj_funcs, pop=100, c2=1.05, w_max=0.9, w_min=0.4,
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
        a1 = random.uniform(0,1)
        a2 = random.uniform(0,1)
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
    
    # Compute a baseline schedule (midpoint allocation)
    baseline_x = (lb + ub) / 2.0
    baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks)
    print("Baseline Makespan (hours):", baseline_makespan)
    plot_gantt(baseline_schedule, "Baseline Schedule")
    
    # ----------------------- Run Original PSO -----------------------
    pso_iterations = 200
    objectives = [objective_makespan, objective_total_cost, objective_neg_utilization]
    P = pso(att=dim, l_b=lb, u_b=ub, obj_func=objectives, constraints=[],
            c=2.13, s=1.05, w=0.41, pop=100, integer=True)
    pso_progress = []
    for i in range(pso_iterations):
        P.moving(steps=1, time_termination=-1)
        current_best = min([part.obj_values[0] for part in P.swarm] +
                           [part.best_p.obj_values[0] for part in P.swarm])
        pso_progress.append(current_best)
    pso_archive = P.get_solution(whole_particle=True)
    best_particle_pso = min(pso_archive, key=lambda p: p.obj_values[0])
    best_solution_pso = best_particle_pso.position
    best_schedule_pso, best_makespan_pso = compute_schedule(best_solution_pso, tasks)
    print("\n[PSO] Best Makespan (hours):", best_makespan_pso)
    
    # ----------------------- Run HHO -----------------------
    hho_iterations = 200
    SearchAgents_no = 30
    archive_hho, hho_progress = MOHHO_with_progress(multi_objective, lb, ub, dim, SearchAgents_no, hho_iterations)
    if archive_hho:
        best_particle_hho = min(archive_hho, key=lambda entry: entry[1][0])
        best_solution_hho = best_particle_hho[0]
        best_makespan_hho = best_particle_hho[1][0]
        best_schedule_hho, _ = compute_schedule(best_solution_hho, tasks)
        print("[HHO] Best Makespan (hours):", best_makespan_hho)
    else:
        best_schedule_hho = None
        best_makespan_hho = None
        print("[HHO] No feasible solution found.")
    
    # ----------------------- Run PDJI-MOPSO -----------------------
    optimizer = PDJI_MOPSO(dim=dim, lb=lb, ub=ub, obj_funcs=objectives,
                           pop=100, c2=1.05, w_max=0.9, w_min=0.4,
                           disturbance_rate_min=0.1, disturbance_rate_max=0.3, jump_interval=20)
    pdji_progress = optimizer.run(max_iter=200)
    archive_pdji = optimizer.archive
    print(f"Number of archive (non-dominated) solutions found by PDJI-MOPSO: {len(archive_pdji)}")
    best_arch = min(archive_pdji, key=lambda entry: entry[1][0])
    best_solution_pdji = best_arch[0]
    best_schedule_pdji, best_makespan_pdji = compute_schedule(best_solution_pdji, tasks)
    print("\n[PDJI-MOPSO] Best Makespan (hours):", best_makespan_pdji)
    
    # ----------------- Convergence Comparison Plot -----------------
    plt.figure(figsize=(10,6))
    plt.plot(range(pso_iterations), pso_progress, label="Original PSO", linewidth=2)
    plt.plot(range(hho_iterations), hho_progress, label="HHO", linewidth=2)
    plt.plot(range(len(pdji_progress)), pdji_progress, label="PDJI-MOPSO", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Makespan (hours)")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ----------------- Pareto Front Comparison -----------------
    # For Original PSO:
    pso_objs = np.array([p.obj_values for p in pso_archive])
    pso_makespans = pso_objs[:, 0]
    pso_costs = pso_objs[:, 1]
    pso_utils = -pso_objs[:, 2]
    plt.figure(figsize=(8,6))
    sc = plt.scatter(pso_makespans, pso_costs, c=pso_utils, cmap='viridis', s=80, edgecolor='k')
    plt.xlabel("Makespan (hours)")
    plt.ylabel("Total Cost")
    plt.title("Original PSO Pareto Front")
    cbar = plt.colorbar(sc)
    cbar.set_label("Avg Utilization")
    plt.grid(True)
    plt.show()
    
    # For HHO:
    if archive_hho:
        hho_objs = np.array([entry[1] for entry in archive_hho])
        hho_makespans = hho_objs[:, 0]
        hho_costs = hho_objs[:, 1]
        hho_utils = -hho_objs[:, 2]
        plt.figure(figsize=(8,6))
        sc = plt.scatter(hho_makespans, hho_costs, c=hho_utils, cmap='viridis', s=80, edgecolor='k')
        plt.xlabel("Makespan (hours)")
        plt.ylabel("Total Cost")
        plt.title("HHO Pareto Front")
        cbar = plt.colorbar(sc)
        cbar.set_label("Avg Utilization")
        plt.grid(True)
        plt.show()
    
    # For PDJI-MOPSO:
    if archive_pdji:
        pdji_objs = np.array([entry[1] for entry in archive_pdji])
        pdji_makespans = pdji_objs[:, 0]
        pdji_costs = pdji_objs[:, 1]
        pdji_utils = -pdji_objs[:, 2]
        plt.figure(figsize=(8,6))
        sc = plt.scatter(pdji_makespans, pdji_costs, c=pdji_utils, cmap='viridis', s=80, edgecolor='k')
        plt.xlabel("Makespan (hours)")
        plt.ylabel("Total Cost")
        plt.title("PDJI-MOPSO Pareto Front")
        cbar = plt.colorbar(sc)
        cbar.set_label("Avg Utilization")
        plt.grid(True)
        plt.show()
    
    # ----------------- Display Final Schedules via Gantt Charts -----------------
    plot_gantt(best_schedule_pso, f"Optimized Schedule (Original PSO)\nMakespan: {best_makespan_pso:.2f} hrs")
    if best_schedule_hho is not None:
        plot_gantt(best_schedule_hho, f"Optimized Schedule (HHO)\nMakespan: {best_makespan_hho:.2f} hrs")
    plot_gantt(best_schedule_pdji, f"Optimized Schedule (PDJI-MOPSO)\nMakespan: {best_makespan_pdji:.2f} hrs")

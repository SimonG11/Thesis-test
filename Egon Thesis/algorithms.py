# algorithms.py
import numpy as np
import random, math
from typing import List, Tuple, Callable, Optional, Dict, Any
from utils import chaotic_map_initialization, levy, dominates, round_half, clip_round_half, discretize_vector, update_archive_with_crowding, compute_crowding_distance
from objectives import multi_objective
import metrics


# =============================================================================
# ----------------------- Algorithm Implementations -------------------------
# =============================================================================
def MOHHO_with_progress(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Adaptive MOHHO_with_progress implements a Multi-Objective Harris Hawks Optimization
    for the RCPSP problem with several enhancements to improve convergence and diversity.
    Decisions are strictly explored in half-step increments.

    Enhancements and Scientific Justifications:
      1. Chaotic Initialization:
         - Uses a logistic chaotic map to initialize the population, thereby enhancing the spread
           and diversity of the initial solutions.
         - Citation: Sun et al. (2019), "Chaotic Multi-Objective Particle Swarm Optimization Algorithm Incorporating Clone Immunity"
           URL: https://doi.org/10.3390/math7020146
         - Also inspired by Yan et al. (2022), "An Improved Multi-Objective Harris Hawk Optimization with Blank Angle Region Enhanced Search"
           URL: https://doi.org/10.3390/sym14050967

      2. Adaptive Step Size Update (Self-adaptation):
         - Dynamically adjusts the step sizes based on improvements between iterations to balance exploration and exploitation.
         - Citation: Adaptive tuning in metaheuristics (e.g., Brest et al. (2006))
           URL: https://doi.org/10.1109/TEVC.2006.872133

      3. Diversity-driven Injection:
         - Monitors population diversity and, if stagnation is detected (average pairwise distance falls below a threshold),
           replaces the worst-performing solution with a new one to avoid premature convergence.
         - Citation: Yüzgeç & Kuşoğlu (2020) propose diversity-driven strategies in multi-objective optimization.

      4. Archive Management via Crowding Distance:
         - Uses a NSGA-II inspired archive update procedure that leverages crowding distance to maintain a diverse set of non-dominated solutions.
         - Citation: Deb et al. (2002), "Multi-Objective Optimization Using Evolutionary Algorithms"
           URL: https://doi.org/10.1109/4235.996017

    Returns:
        archive: A list of non-dominated solutions (each as a tuple of decision and objective vectors).
        progress: A list recording the best makespan value (first objective) per iteration.
    """
    # Enhanced initialization using chaotic map
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    # Initialize self-adaptive step sizes for each hawk and dimension.
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    t = 0
    diversity_threshold = 0.1 * np.mean(ub - lb)
    while t < max_iter:
        # Non-linear decaying escape energy (using cosine schedule)
        E1 = 2 * math.cos((t / max_iter) * (math.pi / 2))
        for i in range(search_agents_no):
            X[i, :] = discretize_vector(np.clip(X[i, :], lb, ub), lb, ub)
            f_val = objf(X[i, :])
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))
        rabbit = random.choice(archive)[0] if archive else X[0, :].copy()
        for i in range(search_agents_no):
            old_x = X[i, :].copy()
            old_obj = np.linalg.norm(objf(old_x))
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            r = random.random()
            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_index = random.randint(0, search_agents_no - 1)
                X_rand = X[rand_index, :].copy()
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * np.abs(X_rand - 2 * random.random() * X[i, :])
                else:
                    X[i, :] = (rabbit - np.mean(X, axis=0)) - random.random() * ((ub - lb) * random.random() + lb)
            else:
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = rabbit - Escaping_Energy * np.abs(rabbit - X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X[i, :] = (rabbit - X[i, :]) - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :])
                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :])
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :]) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0)) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
            new_x = old_x + step_sizes[i, :] * (X[i, :] - old_x)
            new_x = discretize_vector(np.clip(new_x, lb, ub), lb, ub)
            new_obj = np.linalg.norm(objf(new_x))
            if new_obj < old_obj:
                step_sizes[i, :] *= 0.95
            else:
                step_sizes[i, :] *= 1.05
            X[i, :] = new_x.copy()
        dists = [np.linalg.norm(X[i] - X[j]) for i in range(search_agents_no) for j in range(i+1, search_agents_no)]
        avg_dist = np.mean(dists) if dists else 0
        if avg_dist < diversity_threshold:
            obj_values = [np.linalg.norm(objf(X[i])) for i in range(search_agents_no)]
            worst_idx = np.argmax(obj_values)
            if archive:
                base = random.choice(archive)[0]
                new_hawk = base + np.random.uniform(-0.5, 0.5, size=dim)
                X[worst_idx, :] = discretize_vector(new_hawk, lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)
            else:
                X[worst_idx, :] = discretize_vector(chaotic_map_initialization(lb, ub, dim, 1)[0], lb, ub)
                step_sizes[worst_idx, :] = np.ones(dim)
        best_makespan = np.min([objf(X[i, :])[0] for i in range(search_agents_no)])
        progress.append(best_makespan)
        t += 1
    return archive, progress


class PSO:
    """
    Adaptive MOPSO (Multi-Objective Particle Swarm Optimization) for RCPSP with several enhancements.
    
    Enhancements and Scientific Justifications:
      1. Self-adaptive Inertia Weight Update:
         - Dynamically adjusts the inertia weight based on performance improvements to balance exploration and exploitation.
         - Citation: Zhang et al. (2018), Adaptive MOPSO approaches.
         - URL: https://doi.org/10.1007/s11761-018-0231-7

      2. Periodic Mutation/Disturbance:
         - Introduces random disturbances (mutation) in the particle positions to prevent premature convergence.
         - Citation: Sun et al. (2019), "Chaotic Multi-Objective Particle Swarm Optimization Algorithm Incorporating Clone Immunity"
         - URL: https://doi.org/10.3390/math7020146

      3. Archive Update via Crowding Distance:
         - Maintains an external archive of non-dominated solutions using a NSGA-II style crowding distance measure to preserve diversity.
         - Citation: Deb et al. (2002), "Multi-Objective Optimization Using Evolutionary Algorithms"
         - URL: https://doi.org/10.1109/4235.996017

      4. Hypercube-Based Leader Selection:
         - Divides the objective space into hypercubes and selects leaders based on the inverse density of solutions in each cell,
           promoting diverse search directions.
         - Citation: Coello Coello et al. (2004)
         - URL: https://doi.org/10.1080/03052150410001647966

    This class provides methods to initialize the swarm, update velocities and positions, manage the archive,
    and run the optimization for a specified number of iterations.
    """
    def __init__(self, dim: int, lb: np.ndarray, ub: np.ndarray,
                 obj_funcs: List[Callable[[np.ndarray], float]], pop: int = 30,
                 c2: float = 1.05, w_max: float = 0.9, w_min: float = 0.4,
                 disturbance_rate_min: float = 0.1, disturbance_rate_max: float = 0.3,
                 jump_interval: int = 20) -> None:
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.obj_funcs = obj_funcs
        self.pop = pop
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.iteration = 0
        self.max_iter = 200
        self.vmax = self.ub - self.lb
        self.swarm: List[Dict[str, Any]] = []
        # Initialize positions using allowed half-step values.
        for _ in range(pop):
            pos = np.array([random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5))) for i in range(dim)])
            vel = np.array([random.uniform(-self.vmax[i], self.vmax[i]) for i in range(dim)])
            particle = {
                'position': pos,
                'velocity': vel,
                'pbest': pos.copy(),
                'obj': self.evaluate(pos),
                'w': self.w_max  # Start with maximum inertia weight.
            }
            self.swarm.append(particle)
        self.archive: List[Tuple[np.ndarray, np.ndarray]] = []
        self.disturbance_rate_min = disturbance_rate_min
        self.disturbance_rate_max = disturbance_rate_max
        self.jump_interval = jump_interval

    def evaluate(self, pos: np.ndarray) -> np.ndarray:
        """Evaluate a particle's position using the provided objective functions."""
        if len(self.obj_funcs) == 1:
            return np.array(self.obj_funcs[0](pos))
        else:
            return np.array([f(pos) for f in self.obj_funcs])
        
    def select_leader_hypercube(self) -> List[np.ndarray]:
        """
        Select leader particles using hypercube division of the archive.
        """
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        objs = np.array([entry[1] for entry in self.archive])
        num_bins = 5
        mins = np.min(objs, axis=0)
        maxs = np.max(objs, axis=0)
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        cell_indices = []
        cell_counts = {}
        for entry in self.archive:
            idx = tuple(((entry[1] - mins) / ranges * num_bins).astype(int))
            idx = tuple(min(x, num_bins - 1) for x in idx)
            cell_indices.append(idx)
            cell_counts[idx] = cell_counts.get(idx, 0) + 1
        leaders = []
        weights = [1 / cell_counts[cell_indices[i]] for i in range(len(self.archive))]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        for _ in range(self.pop):
            chosen = np.random.choice(len(self.archive), p=probs)
            leaders.append(self.archive[chosen][0])
        return leaders

    def jump_improved_operation(self) -> None:
        """Perform a jump operation to escape local optima."""
        if len(self.archive) < 2:
            return
        c1, c2 = random.sample(self.archive, 2)
        a1, a2 = random.uniform(0, 1), random.uniform(0, 1)
        oc1 = c1[0] + a1 * (c1[0] - c2[0])
        oc2 = c2[0] + a2 * (c2[0] - c1[0])
        for oc in [oc1, oc2]:
            oc = np.array([clip_round_half(val, self.lb[i], self.ub[i]) for i, val in enumerate(oc)])
            obj_val = self.evaluate(oc)
            self.archive = update_archive_with_crowding(self.archive, (oc, obj_val))

    def disturbance_operation(self, particle: Dict[str, Any]) -> None:
        """Apply a random disturbance to a particle's position to enhance exploration."""
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
                new_pos[d] = clip_round_half(new_pos[d], self.lb[d], self.ub[d])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)

    def move(self) -> None:
        """
        Update the swarm by moving each particle, applying self-adaptive inertia weight updates,
        and periodic disturbance operations.
        """
        self.iteration += 1
        leaders = self.select_leader_hypercube()
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj = np.linalg.norm(self.evaluate(old_pos))
            r2 = random.random()
            guide = leaders[idx]
            # Standard PSO velocity and position update.
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([clip_round_half(new_pos[i], self.lb[i], self.ub[i]) for i in range(self.dim)])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)
            particle['pbest'] = new_pos.copy()
            # Update inertia weight based on performance.
            new_obj = np.linalg.norm(self.evaluate(new_pos))
            if new_obj < old_obj:
                particle['w'] = max(particle['w'] * 0.95, self.w_min)
            else:
                particle['w'] = min(particle['w'] * 1.05, self.w_max)
            self.disturbance_operation(particle)
        self.update_archive()
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()
        positions = np.array([p['position'] for p in self.swarm])
        if len(positions) > 1:
            pairwise_dists = [np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions))]
            avg_distance = np.mean(pairwise_dists)
            if avg_distance < 0.1 * np.mean(self.ub - self.lb):
                idx_to_mutate = random.randint(0, self.pop - 1)
                self.swarm[idx_to_mutate]['position'] = np.array([random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5))) for i in range(self.dim)])
                self.swarm[idx_to_mutate]['obj'] = self.evaluate(self.swarm[idx_to_mutate]['position'])
        self.update_archive()

    def update_archive(self) -> None:
        """Update the external archive using the current swarm particles."""
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def run(self, max_iter: Optional[int] = None) -> List[float]:
        """
        Run the Adaptive MOPSO for a specified number of iterations.
        
        Returns:
            convergence: A list of the best makespan values recorded per iteration.
        """
        if max_iter is None:
            max_iter = self.max_iter
        convergence: List[float] = []
        for _ in range(max_iter):
            self.move()
            best_ms = min(p['obj'][0] for p in self.swarm)
            convergence.append(best_ms)
        return convergence


def MOACO_improved(objf: Callable[[np.ndarray], np.ndarray],
                    tasks: List[Dict[str, Any]], 
                    lb: np.ndarray, ub: np.ndarray, 
                    ant_count: int, max_iter: int,
                    alpha: float = 1.0,
                    beta: float = 2.0,
                    evaporation_rate: float = 0.1,
                    w1: float = 1.0,
                    lambda3: float = 2.0,
                    colony_count: int = 10,
                    heuristic_obj: Optional[Callable[[Dict[str, Any], float, int], List[float]]] = None
                   ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Updated MOACO algorithm with improvements to remove bias toward makespan.
    This version uses Tchebycheff scalarization for heuristic ranking and Pareto-based candidate selection
    during local search.
    """

        # ---------------- Helper Functions ----------------
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Min–max scales each column of 'mat' to the [0,1] interval."""
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            norm[:, d] = (mat[:, d] - mins[d]) / range_val if range_val != 0 else 0.5
        return norm, mins, maxs

    def normalized_crowding_distance(archive):
        """
        Compute the crowding distance for each solution in the archive using normalized objectives.
        """
        if not archive:
            return np.array([])
        objs = np.array([entry[1] for entry in archive], dtype=float)
        norm_objs, _, _ = normalize_matrix(objs)
        num_objs = norm_objs.shape[1]
        distances = np.zeros(len(archive))
        for m in range(num_objs):
            sorted_indices = np.argsort(norm_objs[:, m])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
            m_values = norm_objs[sorted_indices, m]
            m_range = m_values[-1] - m_values[0]
            if m_range == 0:
                continue
            for i in range(1, len(archive) - 1):
                distances[sorted_indices[i]] += (m_values[i+1] - m_values[i-1]) / m_range
        return distances

    def fast_non_dominated_sort(candidates: List[List[float]]) -> List[int]:
        """
        Perform fast non-dominated sorting on candidate objective vectors.
        Returns a list of ranks (lower is better).
        """
        n = len(candidates)
        S = [[] for _ in range(n)]
        domination_count = [0] * n
        ranks = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (all(candidates[j][k] <= candidates[i][k] for k in range(len(candidates[i]))) and
                    any(candidates[j][k] < candidates[i][k] for k in range(len(candidates[i])))):
                    domination_count[i] += 1
                elif (all(candidates[i][k] <= candidates[j][k] for k in range(len(candidates[i]))) and
                      any(candidates[i][k] < candidates[j][k] for k in range(len(candidates[i])))):
                    S[i].append(j)
            if domination_count[i] == 0:
                ranks[i] = 1
        current_front = [i for i in range(n) if domination_count[i] == 0]
        front_number = 1
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = front_number + 1
                        next_front.append(j)
            front_number += 1
            current_front = next_front
        return ranks

    # --- Helper: Compute heuristic for each task using Tchebycheff scalarization ---
    def compute_task_heuristic(task_index: int) -> Dict[float, float]:
        possible_values = list(np.arange(lb[task_index], ub[task_index] + 0.5, 0.5))
        candidate_objs = []
        for v in possible_values:
            candidate = np.array([t["min"] for t in tasks])
            candidate[task_index] = v
            candidate_objs.append(objf(candidate))
        candidate_objs = np.array(candidate_objs)
        # Compute the ideal point (componentwise minimum)
        ideal = np.min(candidate_objs, axis=0)
        # Compute Tchebycheff values with equal weights
        tcheby_vals = [max(abs(candidate_objs[j] - ideal)) for j in range(len(possible_values))]
        task_heuristic = {v: 1.0 / (tcheby_vals[j] + 1e-6) for j, v in enumerate(possible_values)}
        return task_heuristic

    # --- Initialization of pheromones and heuristics for each colony ---
    dim = len(lb)
    colony_pheromones = []  # One pheromone matrix per colony
    colony_heuristics = []  # One heuristic matrix per colony
    for colony_idx in range(colony_count):
        pheromone_matrix = []
        heuristic_matrix = []
        for i in range(dim):
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            pheromone_matrix.append({v: 1.0 for v in possible_values})
            heuristic_matrix.append(compute_task_heuristic(i))
        colony_pheromones.append(pheromone_matrix)
        colony_heuristics.append(heuristic_matrix)

    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    best_global = float('inf')
    no_improvement_count = 0
    stagnation_threshold = 10  # iterations before triggering reinitialization
    eps = 1e-6

    # --- Helper: Pareto-based candidate selection ---
    def select_best_candidate(candidates: List[np.ndarray], cand_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ranks = fast_non_dominated_sort(cand_objs.tolist())
        first_front_indices = [i for i, rank in enumerate(ranks) if rank == 1]
        if len(first_front_indices) == 1:
            best_idx = first_front_indices[0]
        else:
            # Compute crowding distances for candidates in the first front
            front_objs = np.array([cand_objs[i] for i in first_front_indices])
            cd = np.zeros(len(front_objs))
            num_objs = front_objs.shape[1]
            for m in range(num_objs):
                sorted_indices = np.argsort(front_objs[:, m])
                cd[sorted_indices[0]] = cd[sorted_indices[-1]] = float('inf')
                m_range = front_objs[sorted_indices[-1], m] - front_objs[sorted_indices[0], m]
                if m_range == 0:
                    continue
                for j in range(1, len(front_objs) - 1):
                    cd[sorted_indices[j]] += (front_objs[sorted_indices[j+1], m] - front_objs[sorted_indices[j-1], m]) / m_range
            best_in_front = np.argmax(cd)
            best_idx = first_front_indices[best_in_front]
        return candidates[best_idx], cand_objs[best_idx]

    # --- Main Iteration Loop ---
    for iteration in range(max_iter):
        colony_solutions = []  # store solutions from all colonies
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            for _ in range(ants_per_colony):
                solution = []
                # Construct solution: for each task, select allocation based on pheromone and heuristic
                for i in range(dim):
                    possible_values = list(pheromone[i].keys())
                    probs = []
                    for v in possible_values:
                        tau = pheromone[i][v]
                        h_val = heuristic[i][v]
                        probs.append((tau ** alpha) * (h_val ** beta))
                    total = sum(probs)
                    if not np.isfinite(total) or total <= 0:
                        probs = [1.0 / len(probs)] * len(probs)
                    else:
                        probs = [p / total for p in probs]
                    r = random.random()
                    cumulative = 0.0
                    chosen = possible_values[-1]
                    for idx, v in enumerate(possible_values):
                        cumulative += probs[idx]
                        if r <= cumulative:
                            chosen = v
                            break
                    solution.append(chosen)
                # Local search: perturb each task’s allocation by ±0.5
                candidates = [solution]
                for i in range(dim):
                    for delta in [-0.5, 0.5]:
                        neighbor = solution.copy()
                        neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
                        candidates.append(neighbor)
                candidates = [np.array(c) for c in candidates]
                cand_objs = np.array([objf(c) for c in candidates], dtype=float)
                best_candidate, best_obj = select_best_candidate(candidates, cand_objs)
                colony_solutions.append((best_candidate.tolist(), best_obj.tolist()))
        # --- Archive update ---
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), np.array(obj_val)))
        # --- Pheromone Evaporation ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.array(all_values)
            var_pheromone = np.var(all_values)
            current_evap_rate = evaporation_rate * 1.5 if var_pheromone < 0.001 else evaporation_rate
            for i in range(dim):
                for v in pheromone[i]:
                    pheromone[i][v] *= (1 - current_evap_rate)
        # --- Pheromone Deposit Update ---
        crowding = normalized_crowding_distance(archive)
        max_cd = np.max(crowding) if len(crowding) > 0 else 1.0
        if not np.isfinite(max_cd) or max_cd <= 0:
            max_cd = 1.0
        decay_factor = 1.0 - (iteration / max_iter)
        for idx, (sol, obj_val) in enumerate(archive):
            deposit = w1 * lambda3 * (crowding[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i, v in enumerate(sol):
                    colony_pheromones[colony_idx][i][v] += deposit
        # --- Multi-Colony Reinitialization and Merge ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.array(all_values)
            if np.var(all_values) < 0.001:
                for i in range(dim):
                    possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
                    pheromone[i] = {v: 1.0 for v in possible_values}
        merged_pheromone = []
        for i in range(dim):
            merged = {}
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            for v in possible_values:
                val = sum(colony_pheromones[colony_idx][i].get(v, 0) for colony_idx in range(colony_count)) / colony_count
                merged[v] = val
            merged_pheromone.append(merged)
        for colony_idx in range(colony_count):
            colony_pheromones[colony_idx] = [merged_pheromone[i].copy() for i in range(dim)]
        # --- Record Progress ---
        if archive:
            objs = np.array([entry[1] for entry in archive])
            ideal = np.min(objs, axis=0)
            # Use Tchebycheff scalarization to compute a balanced score
            tcheby_scores = [max(abs(entry[1] - ideal)) for entry in archive]
            current_best = min(tcheby_scores)
        else:
            current_best = float('inf')
        progress.append(current_best)
        # --- Stagnation Handling ---
        if iteration > 0 and progress[-1] >= progress[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if no_improvement_count >= stagnation_threshold:
            for colony_idx in range(colony_count):
                num_to_reinit = max(1, ants_per_colony // 10)
                for _ in range(num_to_reinit):
                    new_solution = [random.choice(list(np.arange(lb[i], ub[i] + 0.5, 0.5))) for i in range(dim)]
                    archive = update_archive_with_crowding(archive, (np.array(new_solution), objf(np.array(new_solution))))
            no_improvement_count = 0
    return archive, progress
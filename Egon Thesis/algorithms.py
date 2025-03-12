# algorithms.py
import numpy as np
import random, math
from typing import List, Tuple, Callable, Optional, Dict, Any
from utils import chaotic_map_initialization, levy, dominates, round_half, clip_round_half, discretize_vector, update_archive_with_crowding, compute_crowding_distance
from objectives import multi_objective
import metrics
from tqdm import tqdm


# =============================================================================
# ----------------------- Algorithm Implementations -------------------------
# =============================================================================
def MOHHO_with_progress(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Adaptive MOHHO_with_progress implements a Multi-Objective Harris Hawks Optimization.
    Returns:
        archive: List of non-dominated solutions.
        progress: List of progress metric values per iteration.
    """
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            norm[:, d] = (mat[:, d] - mins[d]) / range_val if range_val != 0 else 0.5
        return norm, mins, maxs

    def normalize_obj(obj, mins, maxs):
        obj = np.array(obj, dtype=float)
        norm_obj = np.zeros_like(obj)
        for i in range(len(obj)):
            range_val = maxs[i] - mins[i]
            norm_obj[i] = (obj[i] - mins[i]) / range_val if range_val != 0 else 0.5
        return norm_obj

    # --- Initialization ---
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    diversity_threshold = 0.1 * np.mean(ub - lb)
    
    # Main iteration loop with progress bar.
    for t in tqdm(range(max_iter), desc="MOHHO Progress", unit="iter"):
        # Update archive with current solutions.
        for i in range(search_agents_no):
            X[i, :] = discretize_vector(np.clip(X[i, :], lb, ub), lb, ub)
            f_val = objf(X[i, :])
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))
        
        pop_objs = [objf(X[i, :]) for i in range(search_agents_no)]
        pop_objs_mat = np.array(pop_objs)
        _, pop_mins, pop_maxs = normalize_matrix(pop_objs_mat)
        
        rabbit = random.choice(archive)[0] if archive else X[0, :].copy()
        
        for i in range(search_agents_no):
            old_x = X[i, :].copy()
            old_obj = np.linalg.norm(normalize_obj(objf(old_x), pop_mins, pop_maxs))
            E0 = 2 * random.random() - 1
            E1 = 2 * math.cos((t / max_iter) * (math.pi / 2))
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
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :]) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0)) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()
            
            new_x = old_x + step_sizes[i, :] * (X[i, :] - old_x)
            new_x = discretize_vector(np.clip(new_x, lb, ub), lb, ub)
            new_obj = np.linalg.norm(normalize_obj(objf(new_x), pop_mins, pop_maxs))
            if new_obj < old_obj:
                step_sizes[i, :] *= 0.95
            else:
                step_sizes[i, :] *= 1.05
            X[i, :] = new_x.copy()
        
        normalized_objs = [normalize_obj(objf(X[i, :]), pop_mins, pop_maxs) for i in range(search_agents_no)]
        ideal = np.min(np.array(normalized_objs), axis=0)
        tcheby_values = [max(abs(n_obj - ideal)) for n_obj in normalized_objs]
        progress_metric = min(tcheby_values)
        progress.append(progress_metric)
    
    return archive, progress

# --------------------------- MOPSO Algorithm -------------------------
class PSO:
    """
    Adaptive MOPSO (Multi-Objective Particle Swarm Optimization) for RCPSP.
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
        for _ in range(pop):
            pos = np.array([random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5)))
                            for i in range(dim)])
            vel = np.array([random.uniform(-self.vmax[i], self.vmax[i]) for i in range(dim)])
            particle = {
                'position': pos,
                'velocity': vel,
                'pbest': pos.copy(),
                'obj': self.evaluate(pos),
                'w': self.w_max
            }
            self.swarm.append(particle)
        self.archive: List[Tuple[np.ndarray, np.ndarray]] = []
        self.disturbance_rate_min = disturbance_rate_min
        self.disturbance_rate_max = disturbance_rate_max
        self.jump_interval = jump_interval

    def evaluate(self, pos: np.ndarray) -> np.ndarray:
        if len(self.obj_funcs) == 1:
            return np.array(self.obj_funcs[0](pos))
        else:
            return np.array([f(pos) for f in self.obj_funcs])
    
    @staticmethod
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            norm[:, d] = (mat[:, d] - mins[d]) / range_val if range_val != 0 else 0.5
        return norm, mins, maxs

    @staticmethod
    def normalize_obj(obj, mins, maxs):
        obj = np.array(obj, dtype=float)
        norm_obj = np.zeros_like(obj)
        for i in range(len(obj)):
            range_val = maxs[i] - mins[i]
            norm_obj[i] = (obj[i] - mins[i]) / range_val if range_val != 0 else 0.5
        return norm_obj

    def select_leader_hypercube(self, norm_mins: np.ndarray, norm_maxs: np.ndarray) -> List[np.ndarray]:
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        objs = np.array([entry[1] for entry in self.archive])
        norm_objs = np.array([self.normalize_obj(obj, norm_mins, norm_maxs) for obj in objs])
        num_bins = 5
        mins = np.min(norm_objs, axis=0)
        maxs = np.max(norm_objs, axis=0)
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        cell_indices = []
        cell_counts = {}
        for norm_obj in norm_objs:
            idx = tuple(((norm_obj - mins) / ranges * num_bins).astype(int))
            idx = tuple(min(x, num_bins - 1) for x in idx)
            cell_indices.append(idx)
            cell_counts[idx] = cell_counts.get(idx, 0) + 1
        weights = [1 / cell_counts[cell_indices[i]] for i in range(len(self.archive))]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        leaders = []
        for _ in range(self.pop):
            chosen = np.random.choice(len(self.archive), p=probs)
            leaders.append(self.archive[chosen][0])
        return leaders

    def disturbance_operation(self, particle: Dict[str, Any]) -> None:
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
        self.iteration += 1
        raw_objs = np.array([self.evaluate(p['position']) for p in self.swarm])
        norm_objs, norm_mins, norm_maxs = self.normalize_matrix(raw_objs)
        ideal = np.min(norm_objs, axis=0)
        leaders = self.select_leader_hypercube(norm_mins, norm_maxs)
        
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj_raw = self.evaluate(old_pos)
            old_norm = self.normalize_obj(old_obj_raw, norm_mins, norm_maxs)
            old_scalar = np.max(np.abs(old_norm - ideal))
            
            r2 = random.random()
            guide = leaders[idx]
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([clip_round_half(new_pos[i], self.lb[i], self.ub[i]) for i in range(self.dim)])
            particle['position'] = new_pos
            new_obj_raw = self.evaluate(new_pos)
            new_norm = self.normalize_obj(new_obj_raw, norm_mins, norm_maxs)
            new_scalar = np.max(np.abs(new_norm - ideal))
            particle['obj'] = new_obj_raw
            particle['pbest'] = new_pos.copy()
            
            if new_scalar < old_scalar:
                particle['w'] = max(particle['w'] * 0.95, self.w_min)
            else:
                particle['w'] = min(particle['w'] * 1.05, self.w_max)
            
            self.disturbance_operation(particle)
        
        self.update_archive()
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()
        
        positions = np.array([p['position'] for p in self.swarm])
        if len(positions) > 1:
            pairwise_dists = [np.linalg.norm(positions[i] - positions[j])
                              for i in range(len(positions)) for j in range(i+1, len(positions))]
            avg_distance = np.mean(pairwise_dists)
            if avg_distance < 0.1 * np.mean(self.ub - self.lb):
                idx_to_mutate = random.randint(0, self.pop - 1)
                self.swarm[idx_to_mutate]['position'] = np.array(
                    [random.choice(list(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5)))
                     for i in range(self.dim)])
                self.swarm[idx_to_mutate]['obj'] = self.evaluate(self.swarm[idx_to_mutate]['position'])
        
        self.update_archive()

    def jump_improved_operation(self) -> None:
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

    def update_archive(self) -> None:
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def run(self, max_iter: Optional[int] = None) -> List[float]:
        if max_iter is None:
            max_iter = self.max_iter
        convergence: List[float] = []
        for _ in tqdm(range(max_iter), desc="PSO Progress", unit="iter"):
            self.move()
            best_ms = min(p['obj'][0] for p in self.swarm)
            convergence.append(best_ms)
        return convergence

# --------------------------- MOACO Algorithm -------------------------


def compute_task_heuristic(task_index: int, tasks: List[Dict[str, any]], objf: Callable[[np.ndarray], np.ndarray],
                           lb: np.ndarray, ub: np.ndarray) -> Dict[float, float]:
    """
    For the given task index, compute a heuristic dictionary mapping each possible discrete value
    (in 0.5 increments between lb and ub) to a heuristic value.
    Uses Tchebycheff scalarization based on candidate evaluations.
    """
    possible_values = list(np.arange(lb[task_index], ub[task_index] + 0.5, 0.5))
    candidate_objs = []
    for v in possible_values:
        candidate = np.array([t["min"] for t in tasks], dtype=float)
        candidate[task_index] = v
        candidate_objs.append(objf(candidate))
    candidate_objs = np.array(candidate_objs)
    ideal = np.min(candidate_objs, axis=0)
    # Use maximum absolute deviation (Tchebycheff) as measure.
    tcheby_vals = [max(abs(candidate_objs[j] - ideal)) for j in range(len(possible_values))]
    task_heuristic = {v: 1.0 / (tcheby_vals[j] + 1e-6) for j, v in enumerate(possible_values)}
    return task_heuristic

# ---------------- Helper Functions (assumed to exist or defined similarly) ----------------

def normalized_crowding_distance(archive: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Compute the crowding distance for each solution in the archive.
    (Assume this is implemented as in your code.)
    """
    if not archive:
        return np.array([])
    objs = np.array([entry[1] for entry in archive], dtype=float)
    # Normalize using min-max per objective.
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
        for i in range(1, len(archive)-1):
            distances[sorted_indices[i]] += (m_values[i+1] - m_values[i-1]) / m_range
    return distances

def normalize_matrix(mat: np.ndarray, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min–max scales each column of mat to [0, 1]."""
    mat = np.array(mat, dtype=float)
    mins = mat.min(axis=0)
    maxs = mat.max(axis=0)
    ranges = np.maximum(maxs - mins, epsilon)
    norm = (mat - mins) / ranges
    return norm, mins, maxs

def fast_non_dominated_sort(candidates: List[List[float]]) -> List[int]:
    """
    Fast non-dominated sorting for candidate objective vectors.
    Returns a list of ranks.
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

def select_best_candidate(candidates: List[np.ndarray], cand_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the best candidate solution among those in the first front using crowding distance.
    """
    ranks = fast_non_dominated_sort(cand_objs.tolist())
    first_front_indices = [i for i, rank in enumerate(ranks) if rank == 1]
    if len(first_front_indices) == 1:
        best_idx = first_front_indices[0]
    else:
        front_objs = np.array([cand_objs[i] for i in first_front_indices])
        cd = np.zeros(len(front_objs))
        num_objs = front_objs.shape[1]
        for m in range(num_objs):
            sorted_indices = np.argsort(front_objs[:, m])
            cd[sorted_indices[0]] = cd[sorted_indices[-1]] = float('inf')
            m_range = front_objs[sorted_indices[-1], m] - front_objs[sorted_indices[0], m]
            if m_range == 0:
                continue
            for j in range(1, len(front_objs)-1):
                cd[sorted_indices[j]] += (front_objs[sorted_indices[j+1], m] - front_objs[sorted_indices[j-1], m]) / m_range
        best_in_front = np.argmax(cd)
        best_idx = first_front_indices[best_in_front]
    return candidates[best_idx], cand_objs[best_idx]

def evaluate_candidates_vectorized(solution: List[float], dim: int, lb: np.ndarray, ub: np.ndarray,
                                   objf: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate candidate neighbors for a given solution by perturbing each dimension by ±0.5,
    and evaluate all candidates at once using vectorized operations if possible.
    
    Returns:
       best_candidate: The candidate (as a NumPy array) with the best objective (based on non-dominated sorting + crowding).
       best_obj: Its corresponding objective vector.
    """
    # Base candidate: original solution.
    candidates = [solution]
    # For each dimension, generate two neighbors: one by adding 0.5 and one by subtracting 0.5.
    for i in range(dim):
        for delta in [-0.5, 0.5]:
            neighbor = solution.copy()
            neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
            candidates.append(neighbor)
    candidates_arr = np.array(candidates)  # Shape: (1+2*dim, dim)
    
    # Try vectorized evaluation.
    try:
        cand_objs = objf(candidates_arr)
    except Exception:
        cand_objs = np.array([objf(cand) for cand in candidates_arr])
    
    best_candidate, best_obj = select_best_candidate([np.array(c) for c in candidates], cand_objs)
    return best_candidate, best_obj

# ------------------------- MOACO_improved Function -------------------------

def MOACO_improved(objf: Callable[[np.ndarray], np.ndarray],
                   tasks: List[Dict[str, any]], 
                   lb: np.ndarray, ub: np.ndarray, 
                   ant_count: int, max_iter: int,
                   alpha: float = 1.0,
                   beta: float = 2.0,
                   evaporation_rate: float = 0.1,
                   w1: float = 1.0,
                   lambda3: float = 2.0,
                   colony_count: int = 10) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    MOACO_improved implements a multi-objective Ant Colony Optimization for the RCPSP.
    
    Enhancements:
      - Tchebycheff scalarization is used for heuristic ranking.
      - Candidate solutions are generated with discrete increments (0.5) and neighbors are created in a vectorized way.
      - Adaptive pheromone evaporation and multi-colony pheromone updates are applied.
      - Archive management uses a crowding-distance mechanism.
      
    Returns:
      archive: A list of non-dominated solutions (each as (decision_vector, objective_vector)).
      progress: A list of convergence scores (Tchebycheff-based) per iteration.
    """
    dim = len(lb)
    # Precompute possible discrete values and corresponding heuristics and initialize pheromones for each colony.
    colony_possible_values = []
    colony_pheromones = []
    colony_heuristics = []
    for _ in range(colony_count):
        possible_matrix = []
        pheromone_matrix = []
        heuristic_matrix = []
        for i in range(dim):
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            possible_matrix.append(possible_values)
            pheromone_matrix.append({v: 1.0 for v in possible_values})
            heuristic_matrix.append(compute_task_heuristic(i, tasks, objf, lb, ub))
        colony_possible_values.append(possible_matrix)
        colony_pheromones.append(pheromone_matrix)
        colony_heuristics.append(heuristic_matrix)
    
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    best_global = float('inf')
    no_improvement_count = 0
    stagnation_threshold = 10
    eps = 1e-6

    for iteration in tqdm(range(max_iter), desc="MOACO Iterations", unit="iter"):
        colony_solutions = []
        # Generate candidate solutions for each colony.
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            possible_values = colony_possible_values[colony_idx]
            for _ in range(ants_per_colony):
                # Construct a solution using probability distributions.
                solution = []
                for i in range(dim):
                    poss = possible_values[i]
                    probs = []
                    for v in poss:
                        tau = pheromone[i][v]
                        h_val = heuristic[i][v]
                        probs.append((tau ** alpha) * (h_val ** beta))
                    total = sum(probs)
                    if total <= 0:
                        probs = [1.0 / len(probs)] * len(probs)
                    else:
                        probs = [p / total for p in probs]
                    r = random.random()
                    cumulative = 0.0
                    chosen = poss[-1]
                    for idx, v in enumerate(poss):
                        cumulative += probs[idx]
                        if r <= cumulative:
                            chosen = v
                            break
                    solution.append(chosen)
                # Generate candidate neighbors using vectorized evaluation.
                best_candidate, best_obj = evaluate_candidates_vectorized(solution, dim, lb, ub, objf)
                colony_solutions.append((best_candidate, best_obj))
        
        # Update archive with all colony solutions.
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), np.array(obj_val)))
        
        # Adaptive pheromone evaporation.
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            for i in range(dim):
                for v in pheromone[i]:
                    pheromone[i][v] *= (1 - evaporation_rate)
        
        # Pheromone deposition using normalized crowding distance.
        cd = normalized_crowding_distance(archive)
        max_cd = np.max(cd) if len(cd) > 0 else 1.0
        decay_factor = 1.0 - (iteration / max_iter)
        for idx, (sol, obj_val) in enumerate(archive):
            deposit = w1 * lambda3 * (cd[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i, v in enumerate(sol):
                    colony_pheromones[colony_idx][i][v] += deposit
        
        # Merge pheromones across colonies.
        merged_pheromone = []
        for i in range(dim):
            merged = {}
            poss = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            for v in poss:
                merged[v] = sum(colony_pheromones[colony_idx][i].get(v, 0) for colony_idx in range(colony_count)) / colony_count
            merged_pheromone.append(merged)
        for colony_idx in range(colony_count):
            colony_pheromones[colony_idx] = [merged_pheromone[i].copy() for i in range(dim)]
        
        # Evaluate current progress via a Tchebycheff-like metric.
        if archive:
            objs_archive = np.array([entry[1] for entry in archive])
            ideal = np.min(objs_archive, axis=0)
            tcheby_scores = [max(abs(entry[1] - ideal)) for entry in archive]
            current_best = min(tcheby_scores)
        else:
            current_best = float('inf')
        progress.append(current_best)
        
        if iteration > 0 and progress[-1] >= progress[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        if no_improvement_count >= stagnation_threshold:
            # Diversity injection: reinitialize a few solutions.
            for colony_idx in range(colony_count):
                num_to_reinit = max(1, ants_per_colony // 10)
                for _ in range(num_to_reinit):
                    new_solution = [random.choice(list(np.arange(lb[i], ub[i] + 0.5, 0.5))) for i in range(dim)]
                    archive = update_archive_with_crowding(archive, (np.array(new_solution), objf(np.array(new_solution))))
            no_improvement_count = 0

    return archive, progress


def MOACO_improved1(objf: Callable[[np.ndarray], np.ndarray],
                   tasks: List[Dict[str, Any]], 
                   lb: np.ndarray, ub: np.ndarray, 
                   ant_count: int, max_iter: int,
                   alpha: float = 1.0,
                   beta: float = 2.0,
                   evaporation_rate: float = 0.1,
                   w1: float = 1.0,
                   lambda3: float = 2.0,
                   colony_count: int = 10,
                  ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    MOACO_improved implements a multi-objective Ant Colony Optimization for RCPSP with several enhancements.

    Base algorithm concept from Distributed Optimization by Ant Colonies
    https://www.researchgate.net/publication/216300484_Distributed_Optimization_by_Ant_Colonies

    This version employs:
      1. Tchebycheff Scalarization for Heuristic Ranking:
         - For each task, candidate solutions are evaluated using Tchebycheff scalarization,
           ensuring balanced consideration of all objectives (https://doi.org/10.48550/arXiv.2402.19078).
         - This technique is standard in multi-objective optimization (see Deb et al., 2002:
           "Multi-Objective Optimization Using Evolutionary Algorithms" https://doi.org/10.1109/4235.996017).

      2. Pareto-Based Candidate Selection during Local Search:
         - Instead of aggregating normalized objectives (which can introduce bias), candidates are
           ranked using fast non-dominated sorting and crowding distance.
         - This approach follows the NSGA-II methodology (Deb et al., 2002) and related indicator-based methods 
           (e.g., Zitzler & Künzli, 2004).

      3. Adaptive Pheromone Evaporation and Multi-Colony Pheromone Update:
         - The evaporation rate is adjusted based on pheromone variance to prevent premature convergence.
         - Multi-colony updates are applied to encourage diverse search (Angus & Woodward, 2009,
           https://doi.org/10.1007/s11721-008-0022-4) and adaptive evaporation is inspired by Zhao et al. (2018,
           https://doi.org/10.3390/sym10040104).

      4. Archive Management using Crowding Distance:
         - Archive updates use a NSGA-II inspired crowding distance mechanism (Deb et al., 2002) to
           maintain solution diversity.
           
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
        This function is inspired by the crowding distance mechanism in NSGA-II (Deb et al., 2002).
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
        Based on the sorting procedure used in NSGA-II (Deb et al., 2002).
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
        """
        For the task at index 'task_index', compute heuristic values for each possible allocation
        using Tchebycheff scalarization. This balances the influence of all objectives.
        Reference: Deb et al. (2002).
        """
        possible_values = list(np.arange(lb[task_index], ub[task_index] + 0.5, 0.5))
        candidate_objs = []
        for v in possible_values:
            candidate = np.array([t["min"] for t in tasks])
            candidate[task_index] = v
            candidate_objs.append(objf(candidate))
        candidate_objs = np.array(candidate_objs)
        # Compute the ideal point (componentwise minimum)
        ideal = np.min(candidate_objs, axis=0)
        # Compute Tchebycheff values with equal weights (balanced contribution)
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
    stagnation_threshold = 10  # iterations before triggering reinitialization (diversity injection)
    eps = 1e-6

    # --- Helper: Pareto-based candidate selection ---
    def select_best_candidate(candidates: List[np.ndarray], cand_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the best candidate based on Pareto dominance and crowding distance.
        This selection mechanism uses fast non-dominated sorting and is inspired by the NSGA-II approach (Deb et al., 2002).
        """
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
            # Adaptive evaporation based on variance (prevents premature convergence)
            # Reference: Zhao et al. (2018)
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
        # --- Multi-Colony Pheromone Reinitialization and Merge ---
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
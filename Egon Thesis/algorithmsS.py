import numpy as np
import random, math, time
from typing import List, Tuple, Callable, Optional, Dict, Any
from tqdm import tqdm

# Import standardized functions from your utils module.
from utils import (
    chaotic_map_initialization,
    levy,
    clip_round_half,
    discretize_vector,
    update_archive_with_crowding,
    dominates,
    compute_crowding_distance  # already available in your utils
)

# Import NonDominatedSorting from pymoo (highly optimized, e.g., via Cython).
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# --- Standardized Normalization Routines ---
def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized normalization of each column in a matrix to the [0,1] range.
    """
    mat = mat.astype(float)
    mins = np.min(mat, axis=0)
    maxs = np.max(mat, axis=0)
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    norm = (mat - mins) / ranges
    norm = np.where((maxs - mins) == 0, 0.5, norm)
    return norm, mins, maxs

def normalize_obj(obj: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """
    Vectorized normalization of a single objective vector.
    """
    obj = np.array(obj, dtype=float)
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    norm_obj = (obj - mins) / ranges
    norm_obj = np.where((maxs - mins) == 0, 0.5, norm_obj)
    return norm_obj

# =============================================================================
# --------------------------- MOHHO Algorithm -------------------------------
# =============================================================================
def MOHHO(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int, time_limit: float = None
                        ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Multi-Objective Harris Hawks Optimization (MOHHO) with progress tracking.
    
    Enhancements:
      1. Chaotic Initialization using `chaotic_map_initialization`.
      2. Adaptive Step Size & Escaping Strategy using precomputed random numbers and Levy flights.
      3. Diversity-driven Injection: reinitialize candidate upon stagnation.
      4. Archive Management via Crowding Distance using `update_archive_with_crowding`.
      
    Performance improvements:
      • Entire population's objectives are computed once per iteration (via np.apply_along_axis).
      • The normalized population is cached and reused, eliminating redundant calls to objf().
      • Random numbers for candidate updates are precomputed in bulk.
      
    Note: For further speed gains, consider JIT compiling the inner update loop with Numba.
    
    Parameters:
        objf: Function mapping decision vector to objective vector.
        lb, ub: Lower and upper bounds.
        dim: Problem dimensionality.
        search_agents_no: Number of candidate solutions.
        max_iter: Maximum iterations.
        time_limit: Optional time limit in seconds.
    
    Returns:
        archive: List of non-dominated solutions.
        progress: Tchebycheff score progress per iteration.
    """
    start_time = time.time()
    # Initialize population.
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    no_improvement_count = 0

    for t in tqdm(range(max_iter), desc="MOHHO Progress", unit="iter", leave=False):
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

        # Update each candidate by clipping and discretizing.
        for i in range(search_agents_no):
            X[i, :] = discretize_vector(np.clip(X[i, :], lb, ub), lb, ub)
        
        # Vectorized evaluation of entire population.
        pop_objs = np.apply_along_axis(objf, 1, X)
        # Update archive once using the computed pop_objs.
        for i in range(search_agents_no):
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), pop_objs[i].copy()))
        
        # Compute normalization parameters only once.
        norm_objs, pop_mins, pop_maxs = normalize_matrix(pop_objs)
        ideal = np.min(norm_objs, axis=0)
        rabbit = random.choice(archive)[0] if archive else X[0, :].copy()

        # Precompute arrays of random numbers for candidates.
        r_array = np.random.uniform(size=search_agents_no)
        E0_array = 2 * np.random.uniform(size=search_agents_no) - 1
        E1_val = 2 * math.cos((t / max_iter) * (math.pi / 2))
        Escaping_Energy_array = E1_val * E0_array

        # Update each candidate (inner loop).
        for i in range(search_agents_no):
            old_x = X[i, :].copy()
            f_old = pop_objs[i]  # Cached objective value from beginning of iteration.
            old_norm = normalize_obj(f_old, pop_mins, pop_maxs)
            old_scalar = np.max(np.abs(old_norm - ideal))
            Escaping_Energy = Escaping_Energy_array[i]
            r = r_array[i]

            if abs(Escaping_Energy) >= 1:
                X_rand = X[np.random.randint(0, search_agents_no), :].copy()
                if np.random.uniform() < 0.5:
                    X[i, :] = X_rand - np.random.uniform() * np.abs(X_rand - 2 * np.random.uniform() * X[i, :])
                else:
                    X[i, :] = (rabbit - np.mean(X, axis=0)) - np.random.uniform() * ((ub - lb) * np.random.uniform() + lb)
            else:
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = rabbit - Escaping_Energy * np.abs(rabbit - X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    jump_strength = 2 * (1 - np.random.uniform())
                    X[i, :] = (rabbit - X[i, :]) - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :])
                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    jump_strength = 2 * (1 - np.random.uniform())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :])
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(f_old, pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :]) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(f_old, pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    jump_strength = 2 * (1 - np.random.uniform())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(f_old, pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0)) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < np.linalg.norm(normalize_obj(f_old, pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()

            new_x = old_x + step_sizes[i, :] * (X[i, :] - old_x)
            new_x = discretize_vector(np.clip(new_x, lb, ub), lb, ub)
            f_new = objf(new_x)
            new_obj = np.linalg.norm(normalize_obj(f_new, pop_mins, pop_maxs))
            step_sizes[i, :] *= 0.95 if new_obj < old_scalar else 1.05
            X[i, :] = new_x.copy()
        
        # Update progress metric (Tchebycheff scalarization).
        norm_objs_new, _, _ = normalize_matrix(np.apply_along_axis(objf, 1, X))
        ideal_new = np.min(norm_objs_new, axis=0)
        tcheby_values = np.max(np.abs(norm_objs_new - ideal_new), axis=1)
        progress_metric = np.min(tcheby_values)
        progress.append(progress_metric)
        
        # Diversity-driven injection if stagnation.
        if t > 0 and progress[-1] >= progress[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if no_improvement_count >= 10:
            X[np.random.randint(0, search_agents_no), :] = np.array(
                [random.choice(np.arange(lb[i], ub[i] + 0.5, 0.5)) for i in range(dim)]
            )
            no_improvement_count = 0

    return archive, progress

# =============================================================================
# --------------------------- MOPSO Algorithm (Class PSO) ---------------------
# =============================================================================
class PSO:
    """
    Multi-Objective Particle Swarm Optimization (MOPSO) with adaptive inertia,
    periodic mutation, and hypercube-based leader selection.
    
    Enhancements:
      1. Self-adaptive Inertia Weight Update based on Tchebycheff scalarization.
      2. Periodic Mutation/Disturbance to escape local optima.
      3. Archive Management using NSGA-II style crowding (via update_archive_with_crowding).
      4. Hypercube-Based Leader Selection for low-density regions.
      
    Performance improvements:
      • Vectorized evaluation of the swarm's objectives.
      • Precomputation of random numbers for the update loop.
      • Vectorized pairwise distance calculation for diversity checking.
      
    Uses standardized routines from the utils module.
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
        # Initialize swarm positions (discrete using half-steps) and velocities.
        for _ in range(pop):
            pos = np.array([random.choice(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5))
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
        return normalize_matrix(mat)
    
    @staticmethod
    def normalize_obj(obj: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
        return normalize_obj(obj, mins, maxs)
    
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
            # Precompute random indices and values.
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
        
        # Vectorize evaluation of swarm positions.
        positions = np.array([p['position'] for p in self.swarm])
        raw_objs = np.apply_along_axis(self.evaluate, 1, positions)
        norm_objs, norm_mins, norm_maxs = self.normalize_matrix(raw_objs)
        ideal = np.min(norm_objs, axis=0)
        leaders = self.select_leader_hypercube(norm_mins, norm_maxs)
        
        # Precompute random numbers for the update.
        r_array = np.random.uniform(size=self.pop)
        
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            # Use cached objective from raw_objs if available.
            old_obj_raw = raw_objs[idx]
            old_norm = self.normalize_obj(old_obj_raw, norm_mins, norm_maxs)
            old_scalar = np.max(np.abs(old_norm - ideal))
            
            r2 = r_array[idx]
            guide = leaders[idx]
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            particle['velocity'] = np.clip(new_v, -self.vmax, self.vmax)
            new_pos = particle['position'] + particle['velocity']
            # Apply discretization on new position.
            new_pos = np.array([clip_round_half(new_pos[i], self.lb[i], self.ub[i]) for i in range(self.dim)])
            particle['position'] = new_pos
            new_obj_raw = self.evaluate(new_pos)
            new_norm = self.normalize_obj(new_obj_raw, norm_mins, norm_maxs)
            new_scalar = np.max(np.abs(new_norm - ideal))
            particle['obj'] = new_obj_raw
            particle['pbest'] = new_pos.copy()
            
            # Adaptive inertia weight update.
            if new_scalar < old_scalar:
                particle['w'] = max(particle['w'] * 0.95, self.w_min)
            else:
                particle['w'] = min(particle['w'] * 1.05, self.w_max)
            
            self.disturbance_operation(particle)
        
        # Update archive once after the swarm update.
        self.update_archive()
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()
        
        # Use vectorized pairwise distance calculation for diversity check.
        positions = np.array([p['position'] for p in self.swarm])
        if positions.shape[0] > 1:
            # Compute pairwise Euclidean distances (upper triangle).
            diff = positions[:, None, :] - positions[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            # Take only the upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(positions.shape[0], k=1)
            avg_dist = np.mean(dists[triu_indices])
            if avg_dist < 0.1 * np.mean(self.ub - self.lb):
                idx_to_mutate = random.randint(0, self.pop - 1)
                new_position = np.array([random.choice(np.arange(self.lb[i], self.ub[i] + 0.5, 0.5))
                                         for i in range(self.dim)])
                self.swarm[idx_to_mutate]['position'] = new_position
                self.swarm[idx_to_mutate]['obj'] = self.evaluate(new_position)
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
    
    def run(self, max_iter: Optional[int] = None, time_limit: float = None) -> List[float]:
        start_time = time.time()
        if max_iter is None:
            max_iter = self.max_iter
        convergence: List[float] = []
        for _ in tqdm(range(max_iter), desc="PSO Progress", unit="iter", leave=False):
            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break
            self.move()
            # Use vectorized extraction of best objective.
            swarm_objs = np.array([p['obj'] for p in self.swarm])
            best_ms = np.min(swarm_objs[:, 0])
            convergence.append(best_ms)
        return convergence

# =============================================================================
# --------------------------- MOACO Algorithm -------------------------------
# =============================================================================
def MOACO(objf: Callable[[np.ndarray], np.ndarray],
                   tasks: List[Dict[str, Any]], 
                   lb: np.ndarray, ub: np.ndarray, 
                   ant_count: int, max_iter: int,
                   alpha: float = 1.0,
                   beta: float = 2.0,
                   evaporation_rate: float = 0.1,
                   w1: float = 1.0,
                   lambda3: float = 2.0,
                   colony_count: int = 10,
                   time_limit: float = None
                  ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Multi-Objective Ant Colony Optimization (MOACO) with improvements for RCPSP.
    
    Enhancements:
      1. Tchebycheff Scalarization for Heuristic Ranking.
      2. Pareto-Based Candidate Selection using pymoo’s NonDominatedSorting and vectorized crowding distance.
      3. Adaptive Pheromone Evaporation & Multi-Colony Update.
      4. Archive Management using NSGA-II style crowding (via update_archive_with_crowding).
    
    Performance improvements:
      • Pheromone matrices are stored as NumPy arrays with precomputed index mappings.
      • Candidate solution generation and neighbor production are vectorized per colony.
      • Batch evaluation of candidate solutions reduces redundant objective function calls.
      • Archive update is performed in a consolidated loop.
    
    Note: Further speed gains may be obtained by JIT compiling inner loops with Numba if the objective
          function and helper routines are nopython–compatible.
    
    Parameters:
        objf: Objective function.
        tasks: List of task dictionaries (each with a "min" key).
        lb, ub: Lower and upper bounds.
        ant_count: Total number of ants.
        max_iter: Maximum iterations.
        alpha, beta: Pheromone-heuristic balance parameters.
        evaporation_rate: Base pheromone evaporation rate.
        w1, lambda3: Pheromone deposit parameters.
        colony_count: Number of colonies.
        time_limit: Optional time limit (in seconds).
    
    Returns:
        archive: List of non-dominated solutions.
        progress: Tchebycheff score progress per iteration.
    """
    start_time = time.time()
    
    def normalize(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return normalize_matrix(mat)
    
    nds = NonDominatedSorting()
    
    def select_best_candidate(candidates: List[np.ndarray], cand_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fronts = nds.do(cand_objs, only_non_dominated_front=False)
        first_front = fronts[0]
        if len(first_front) == 1:
            best_idx = first_front[0]
        else:
            dummy_archive = [(None, cand_objs[i]) for i in first_front]
            cd = compute_crowding_distance(dummy_archive)
            best_in_front = np.argmax(cd)
            best_idx = first_front[best_in_front]
        return candidates[best_idx], cand_objs[best_idx]
    
    def compute_task_heuristic(task_index: int) -> np.ndarray:
        # Precompute possible discrete values.
        vals = np.arange(lb[task_index], ub[task_index] + 0.5, 0.5)
        candidate_objs = []
        for v in vals:
            candidate = np.array([t["min"] for t in tasks])
            candidate[task_index] = v
            candidate_objs.append(objf(candidate))
        candidate_objs = np.array(candidate_objs)
        norm_objs, _, _ = normalize(candidate_objs)
        norm_ideal = np.min(norm_objs, axis=0)
        tcheby_vals = np.max(np.abs(norm_objs - norm_ideal), axis=1)
        return 1.0 / (tcheby_vals + 1e-6)
    
    # Precompute possible values and index mapping.
    dim = len(lb)
    possible_vals = [np.arange(lb[i], ub[i] + 0.5, 0.5) for i in range(dim)]
    idx_mapping = [{v: idx for idx, v in enumerate(possible_vals[i])} for i in range(dim)]
    
    # Initialize pheromone matrices as NumPy arrays.
    colony_pheromones = []
    colony_heuristics = []
    for _ in range(colony_count):
        pheromone_matrix = [np.ones(len(possible_vals[i])) for i in range(dim)]
        heuristic_matrix = [compute_task_heuristic(i) for i in range(dim)]
        colony_pheromones.append(pheromone_matrix)
        colony_heuristics.append(heuristic_matrix)
    
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    no_improvement_count = 0
    stagnation_threshold = 10
    eps = 1e-6
    
    for iteration in tqdm(range(max_iter), desc="MOACO Progress", unit="iter"):
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break
        colony_solutions = []
        # For each colony, generate all candidate solutions in a vectorized fashion.
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            for _ in range(ants_per_colony):
                # For each ant, sample each dimension at once.
                solution = np.empty(dim)
                for i in range(dim):
                    probs = (pheromone[i] ** alpha) * (heuristic[i] ** beta)
                    total = probs.sum()
                    if not np.isfinite(total) or total <= 0:
                        probs = np.ones_like(probs) / len(probs)
                    else:
                        probs = probs / total
                    idx = np.random.choice(len(probs), p=probs)
                    solution[i] = possible_vals[i][idx]
                # Generate neighbors (1 + 2*dim candidates) for local search.
                candidates = [solution.copy()]
                for i in range(dim):
                    for delta in [-0.5, 0.5]:
                        neighbor = solution.copy()
                        neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
                        candidates.append(neighbor)
                candidates = [np.array(c) for c in candidates]
                cand_objs = np.array([objf(c) for c in candidates], dtype=float)
                best_candidate, best_obj = select_best_candidate(candidates, cand_objs)
                colony_solutions.append((best_candidate, best_obj))
        # Update archive in a batch.
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (sol, obj_val))
        
        # Adaptive pheromone evaporation (vectorized over each colony and dimension).
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_vals = np.concatenate([p for p in pheromone])
            var_pheromone = np.var(all_vals)
            current_evap_rate = evaporation_rate * 1.5 if var_pheromone < 0.001 else evaporation_rate
            for i in range(dim):
                pheromone[i] *= (1 - current_evap_rate)
        
        # Deposit pheromones based on archive crowding.
        crowding = compute_crowding_distance([(None, entry[1]) for entry in archive]) if archive else np.array([])
        max_cd = np.max(crowding) if crowding.size > 0 else 1.0
        for idx, (sol, obj_val) in enumerate(archive):
            if not np.isfinite(max_cd) or max_cd <= 0:
                max_cd = 1.0
            decay_factor = 1.0 - (iteration / max_iter)
            deposit = w1 * lambda3 * (crowding[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i in range(dim):
                    index = idx_mapping[i][sol[i]]
                    colony_pheromones[colony_idx][i][index] += deposit
        
        # Reset pheromones if variance is too low.
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_vals = np.concatenate([p for p in pheromone])
            if np.var(all_vals) < 0.001:
                for i in range(dim):
                    colony_pheromones[colony_idx][i] = np.ones_like(colony_pheromones[colony_idx][i])
        
        # Merge pheromones across colonies.
        merged_pheromone = []
        for i in range(dim):
            avg_phero = np.mean([colony_pheromones[colony_idx][i] for colony_idx in range(colony_count)], axis=0)
            merged_pheromone.append(avg_phero)
        for colony_idx in range(colony_count):
            for i in range(dim):
                colony_pheromones[colony_idx][i] = merged_pheromone[i].copy()
        
        # Progress metric via Tchebycheff scalarization.
        if archive:
            objs = np.array([entry[1] for entry in archive])
            ideal = np.min(objs, axis=0)
            tcheby_scores = np.max(np.abs(objs - ideal), axis=1)
            current_best = np.min(tcheby_scores)
        else:
            current_best = float('inf')
        progress.append(current_best)
        
        if iteration > 0 and progress[-1] >= progress[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if no_improvement_count >= stagnation_threshold:
            for colony_idx in range(colony_count):
                num_to_reinit = max(1, ants_per_colony // 10)
                for _ in range(num_to_reinit):
                    new_solution = np.array([random.choice(possible_vals[i]) for i in range(dim)])
                    archive = update_archive_with_crowding(archive, (new_solution, objf(new_solution)))
            no_improvement_count = 0

    return archive, progress

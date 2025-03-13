# algorithms.py
import numpy as np
import random, math
from typing import List, Tuple, Callable, Optional, Dict, Any
from utils import chaotic_map_initialization, levy, dominates, round_half, clip_round_half, discretize_vector, update_archive_with_crowding, compute_crowding_distance
from objectives import multi_objective
import metrics
from tqdm import tqdm
import time


# =============================================================================
# ----------------------- Algorithm Implementations -------------------------
# =============================================================================
# --------------------------- MOHHO Algorithm -------------------------
def MOHHO_with_progress(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int, time_limit: float = None) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Adaptive MOHHO_with_progress implements a Multi-Objective Harris Hawks Optimization for the RCPSP
    problem with four key enhancements aimed at improving convergence, diversity, and solution quality.
    The search moves are restricted to half-step increments.

    Enhancements and Scientific Justifications:
      1. Chaotic Initialization:
         - Initializes the population using a logistic chaotic map to ensure a diverse and wide spread of
           initial candidate solutions.
         - References:
              - Sun et al. (2019): https://doi.org/10.3390/math7020146
              - Yan et al. (2022): https://doi.org/10.3390/sym14050967
      2. Adaptive Step Size Update & Escaping Strategy:
         - Dynamically adjusts the step sizes based on improvements between iterations and uses an escaping
           energy concept to allow occasional large jumps (via random or Levy flight moves).
         - Reference: Brest et al. (2006): https://doi.org/10.1109/TEVC.2006.872133
      3. Diversity-driven Injection:
         - Monitors the diversity of the population (e.g., via pairwise distances) and, upon detecting stagnation,
           reinitializes one or more solutions to inject diversity.
         - Reference: Strategies proposed by Yüzgeç & Kuşoğlu (2020)
      4. Archive Management via Crowding Distance:
         - Maintains an external archive of non-dominated solutions, using a NSGA-II inspired crowding distance
           mechanism to preserve diversity along the Pareto front.
         - Reference: Deb et al. (2002): https://doi.org/10.1109/4235.996017

    Returns:
        archive: A list of non-dominated solutions (each a tuple containing a decision vector and its objective vector).
        progress: A list recording progress metrics (best makespan values) per iteration.
    """

    start_time = time.time()


    # ----- Helper Functions for Normalization -----
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Scales the columns of a matrix to the [0,1] range.
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            norm[:, d] = (mat[:, d] - mins[d]) / range_val if range_val != 0 else 0.5
        return norm, mins, maxs

    def normalize_obj(obj, mins, maxs):
        # Normalizes a single objective vector using provided minimum and maximum values.
        obj = np.array(obj, dtype=float)
        norm_obj = np.zeros_like(obj)
        for i in range(len(obj)):
            range_val = maxs[i] - mins[i]
            norm_obj[i] = (obj[i] - mins[i]) / range_val if range_val != 0 else 0.5
        return norm_obj

    # ----- Enhancement 1: Chaotic Initialization -----
    # Generate a diverse initial population using a logistic chaotic map.
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    step_sizes = np.ones((search_agents_no, dim))
    archive: List[Tuple[np.ndarray, np.ndarray]] = []  # Archive to store non-dominated solutions.
    progress: List[float] = []
    diversity_threshold = 0.1 * np.mean(ub - lb)

    no_improvement_count = 0

    # Main iterative optimization loop:
    for t in tqdm(range(max_iter), desc="MOHHO Progress", unit="iter", leave=False, position=2):
            # Check if time limit is reached
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break
        # ----- Enhancement 4: Archive Management via Crowding Distance -----
        # Update the archive with the current population.
        for i in range(search_agents_no):
            X[i, :] = discretize_vector(np.clip(X[i, :], lb, ub), lb, ub)
            f_val = objf(X[i, :])
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))

        # Normalize current objective values for proper scaling.
        pop_objs = [objf(X[i, :]) for i in range(search_agents_no)]
        pop_objs_mat = np.array(pop_objs)
        _, pop_mins, pop_maxs = normalize_matrix(pop_objs_mat)

        # Choose the best solution (termed "rabbit") from the archive.
        rabbit = random.choice(archive)[0] if archive else X[0, :].copy()

        # Process each candidate (agent) individually.
        for i in range(search_agents_no):
            old_x = X[i, :].copy()
            old_obj = np.linalg.norm(normalize_obj(objf(old_x), pop_mins, pop_maxs))
            # Compute the "escaping energy" used to determine if a jump is needed.
            E0 = 2 * random.random() - 1
            E1 = 2 * math.cos((t / max_iter) * (math.pi / 2))
            Escaping_Energy = E1 * E0
            r = random.random()

            # ----- Enhancement 2: Adaptive Step Size Update & Escaping Strategy -----
            # If escaping energy is high, perform a large random jump; otherwise, use the best solution (rabbit)
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
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < \
                       np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - X[i, :]) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < \
                           np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(normalize_obj(objf(X1), pop_mins, pop_maxs)) < \
                       np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(jump_strength * rabbit - np.mean(X, axis=0)) + np.random.randn(dim) * levy(dim)
                        if np.linalg.norm(normalize_obj(objf(X2), pop_mins, pop_maxs)) < \
                           np.linalg.norm(normalize_obj(objf(X[i, :]), pop_mins, pop_maxs)):
                            X[i, :] = X2.copy()

            # Continue Enhancement 2: Update the step size based on whether the new solution is better.
            new_x = old_x + step_sizes[i, :] * (X[i, :] - old_x)
            new_x = discretize_vector(np.clip(new_x, lb, ub), lb, ub)
            new_obj = np.linalg.norm(normalize_obj(objf(new_x), pop_mins, pop_maxs))
            if new_obj < old_obj:
                step_sizes[i, :] *= 0.95  # Reduce step size when improvement is observed.
            else:
                step_sizes[i, :] *= 1.05  # Increase step size to promote exploration.
            X[i, :] = new_x.copy()

        # ----- Compute Progress Metric -----
        # Use a Tchebycheff scalarization to compute a progress measure.
        normalized_objs = [normalize_obj(objf(X[i, :]), pop_mins, pop_maxs) for i in range(search_agents_no)]
        ideal = np.min(np.array(normalized_objs), axis=0)
        tcheby_values = [max(abs(n_obj - ideal)) for n_obj in normalized_objs]
        progress_metric = min(tcheby_values)
        progress.append(progress_metric)

        # ----- Enhancement 3: Diversity-driven Injection -----
        # If no improvement is detected over several iterations, reinitialize a candidate to inject diversity.
        if t > 0 and progress[-1] >= progress[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if no_improvement_count >= 10:
            idx_to_reinit = random.randint(0, search_agents_no - 1)
            X[idx_to_reinit, :] = np.array([random.choice(list(np.arange(lb[i], ub[i] + 0.5, 0.5)))
                                            for i in range(dim)])
            no_improvement_count = 0

    return archive, progress

# --------------------------- MOPSO Algorithm -------------------------
class PSO:
    """
    Adaptive MOPSO implements a Multi-Objective Particle Swarm Optimization for the RCPSP problem,
    incorporating four main enhancements to balance exploration and maintain a diverse Pareto archive.

    Enhancements and Scientific Justifications:
      1. Self-adaptive Inertia Weight Update:
         - Dynamically adjusts the inertia weight (w) based on the quality of each particle's move.
         - Reference: Zhang et al. (2018) - https://doi.org/10.1007/s11761-018-0231-7
      2. Periodic Mutation/Disturbance:
         - Randomly perturbs particle positions to help escape local optima.
         - Reference: Sun et al. (2019) - https://doi.org/10.3390/math7020146
      3. Archive Update via Crowding Distance:
         - Maintains an external archive of non-dominated solutions using a NSGA-II style crowding distance mechanism.
         - Reference: Deb et al. (2002) - https://doi.org/10.1109/4235.996017
      4. Hypercube-Based Leader Selection:
         - Divides the objective space into hypercubes and selects leaders from sparsely populated regions.
         - Reference: Coello Coello et al. (2004) - https://doi.org/10.1080/03052150410001647966

    This class initializes a swarm, updates particles’ velocities and positions, manages the archive,
    and executes the multi-objective optimization.
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
        # Initialize swarm: positions are selected from half-step increments.
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
        self.archive: List[Tuple[np.ndarray, np.ndarray]] = []  # [Enhancement 3]
        self.disturbance_rate_min = disturbance_rate_min
        self.disturbance_rate_max = disturbance_rate_max
        self.jump_interval = jump_interval

    def evaluate(self, pos: np.ndarray) -> np.ndarray:
        # Evaluate the objective(s) for the given position.
        if len(self.obj_funcs) == 1:
            return np.array(self.obj_funcs[0](pos))
        else:
            return np.array([f(pos) for f in self.obj_funcs])
    
    @staticmethod
    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Normalize each column of a matrix to the [0,1] range.
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
        # Normalize a single objective vector.
        obj = np.array(obj, dtype=float)
        norm_obj = np.zeros_like(obj)
        for i in range(len(obj)):
            range_val = maxs[i] - mins[i]
            norm_obj[i] = (obj[i] - mins[i]) / range_val if range_val != 0 else 0.5
        return norm_obj

    def select_leader_hypercube(self, norm_mins: np.ndarray, norm_maxs: np.ndarray) -> List[np.ndarray]:
        # [Enhancement 4] Hypercube-Based Leader Selection:
        # Partition the normalized objective space into hypercubes and choose leaders from underpopulated cells.
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
        # [Enhancement 2] Periodic Mutation/Disturbance:
        # Randomly perturb the particle's position to help escape local optima.
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
        # Normalize the current swarm's objective values.
        raw_objs = np.array([self.evaluate(p['position']) for p in self.swarm])
        norm_objs, norm_mins, norm_maxs = self.normalize_matrix(raw_objs)
        ideal = np.min(norm_objs, axis=0)
        # [Enhancement 4] Leader Selection using hypercube-based strategy.
        leaders = self.select_leader_hypercube(norm_mins, norm_maxs)
        
        # Update each particle.
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj_raw = self.evaluate(old_pos)
            old_norm = self.normalize_obj(old_obj_raw, norm_mins, norm_maxs)
            old_scalar = np.max(np.abs(old_norm - ideal))
            
            r2 = random.random()
            guide = leaders[idx]
            # Standard PSO velocity update with inertia and leader guidance.
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
            
            # [Enhancement 1 & 2] Self-adaptive Inertia Weight Update:
            # If the move results in improvement (i.e. lower Tchebycheff score), reduce the inertia weight;
            # otherwise, increase it to promote further exploration.
            if new_scalar < old_scalar:
                particle['w'] = max(particle['w'] * 0.95, self.w_min)
            else:
                particle['w'] = min(particle['w'] * 1.05, self.w_max)
            
            self.disturbance_operation(particle)  # [Enhancement 2] Apply mutation.
        
        self.update_archive()  # [Enhancement 3] Archive management.
        # [Optional] Jump Operation: inject diversity every few iterations.
        if self.iteration % self.jump_interval == 0:
            self.jump_improved_operation()
        
        # Check diversity: if the average pairwise distance is too low, reinitialize one particle.
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
        # [Enhancement 2] Jump Operation:
        # Combine two archived solutions to generate a new candidate, further enhancing diversity.
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
        # [Enhancement 3] Archive Management:
        # Update the archive with each particle's current position and objective value.
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def run(self, max_iter: Optional[int] = None, time_limit: float = None) -> List[float]:
        start_time = time.time()
        # Run the PSO optimization for a specified number of iterations.
        if max_iter is None:
            max_iter = self.max_iter
        convergence: List[float] = []
        for _ in tqdm(range(max_iter), desc="PSO Progress", unit="iter"):
                    # Check if time limit is reached
            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break
            self.move()
            best_ms = min(p['obj'][0] for p in self.swarm)
            convergence.append(best_ms)
        return convergence

# --------------------------- MOACO Algorithm -------------------------
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
                   time_limit: float = None
                  ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    MOACO_improved implements a Multi-Objective Ant Colony Optimization for the RCPSP problem,
    incorporating four key enhancements to improve convergence, diversity, and solution quality.

    Base Algorithm Reference:
      - Distributed Optimization by Ant Colonies
        URL: https://www.researchgate.net/publication/216300484_Distributed_Optimization_by_Ant_Colonies

    Enhancements and Scientific Justifications:
      1. Tchebycheff Scalarization for Heuristic Ranking:
         - Ranks candidate solutions using Tchebycheff scalarization to ensure balanced consideration of all objectives.
         - Reference: Deb et al. (2002) - https://doi.org/10.1109/4235.996017
      2. Pareto-Based Candidate Selection during Local Search:
         - Uses fast non-dominated sorting and crowding distance to select the best candidate from local neighborhoods.
         - References: NSGA-II (Deb et al., 2002) and Zitzler & Künzli (2004)
      3. Adaptive Pheromone Evaporation and Multi-Colony Pheromone Update:
         - Adjusts the evaporation rate based on pheromone variance and integrates information from multiple colonies.
         - References:
             - Angus & Woodward (2009): https://doi.org/10.1007/s11721-008-0022-4
             - Zhao et al. (2018): https://doi.org/10.3390/sym10040104
      4. Archive Management using Crowding Distance:
         - Maintains a diverse archive of non-dominated solutions using a NSGA-II inspired crowding distance mechanism.
         - Reference: Deb et al. (2002) - https://doi.org/10.1109/4235.996017

    Returns:
        archive: List of non-dominated solutions (each a tuple of decision and objective vectors).
        progress: List of progress metric values (Tchebycheff scores) per iteration.
    """

    start_time = time.time()


    def normalize_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Normalize each column of a matrix to the [0,1] interval.
        mat = np.array(mat, dtype=float)
        mins = mat.min(axis=0)
        maxs = mat.max(axis=0)
        norm = np.zeros_like(mat)
        for d in range(mat.shape[1]):
            range_val = maxs[d] - mins[d]
            norm[:, d] = (mat[:, d] - mins[d]) / range_val if range_val != 0 else 0.5
        return norm, mins, maxs

    def normalized_crowding_distance(archive):
        # Compute the crowding distance for each solution in the archive using normalized objectives.
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
        # Perform non-dominated sorting on candidate solutions.
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

    def compute_task_heuristic(task_index: int) -> Dict[float, float]:
        """
        For the task at index 'task_index', compute heuristic values for each possible allocation value
        using Tchebycheff scalarization on normalized objectives.
        """
        possible_values = list(np.arange(lb[task_index], ub[task_index] + 0.5, 0.5))
        candidate_objs = []
        for v in possible_values:
            candidate = np.array([t["min"] for t in tasks])
            candidate[task_index] = v
            candidate_objs.append(objf(candidate))
        candidate_objs = np.array(candidate_objs)
        # Normalize the candidate objective values across the possible allocations
        norm_objs, obj_mins, obj_maxs = normalize_matrix(candidate_objs)
        # Compute the normalized ideal (componentwise minimum)
        norm_ideal = np.min(norm_objs, axis=0)
        # For each candidate, use the maximum deviation from the ideal as the Tchebycheff score
        tcheby_vals = [max(abs(norm_objs[j] - norm_ideal)) for j in range(len(possible_values))]
        # Higher heuristic value (inverse of deviation) is better.
        task_heuristic = {v: 1.0 / (tcheby_vals[j] + 1e-6) for j, v in enumerate(possible_values)}
        return task_heuristic

    # ----- Enhancement 3: Initialization of Colonies -----
    # Create multiple colonies to promote diverse exploration.
    dim = len(lb)
    colony_pheromones = []
    colony_heuristics = []
    for colony_idx in range(colony_count):
        pheromone_matrix = []
        heuristic_matrix = []
        for i in range(dim):
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            # Initialize pheromone values uniformly.
            pheromone_matrix.append({v: 1.0 for v in possible_values})
            # Compute heuristic values based on Tchebycheff scalarization.
            heuristic_matrix.append(compute_task_heuristic(i))
        colony_pheromones.append(pheromone_matrix)
        colony_heuristics.append(heuristic_matrix)

    archive: List[Tuple[np.ndarray, np.ndarray]] = []  # Archive for non-dominated solutions.
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    no_improvement_count = 0
    stagnation_threshold = 10
    eps = 1e-6

    def select_best_candidate(candidates: List[np.ndarray], cand_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Select the best candidate based on non-dominated sorting and crowding distance.
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
                for j in range(1, len(front_objs) - 1):
                    cd[sorted_indices[j]] += (front_objs[sorted_indices[j+1], m] - front_objs[sorted_indices[j-1], m]) / m_range
            best_in_front = np.argmax(cd)
            best_idx = first_front_indices[best_in_front]
        return candidates[best_idx], cand_objs[best_idx]

    # Main iterative loop for MOACO:
    for iteration in tqdm(range(max_iter), desc="MOACO Progress", unit="iter"):
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break
        colony_solutions = []
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            # For each ant, construct a solution.
            for _ in range(ants_per_colony):
                solution = []
                for i in range(dim):
                    possible_values = list(pheromone[i].keys())
                    probs = []
                    # ----- Enhancement 1: Tchebycheff Scalarization for Heuristic Ranking -----
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
                # Generate neighboring solutions to refine the candidate.
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
        # ----- Enhancement 4: Archive Management -----
        # Update the archive with new candidate solutions.
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), np.array(obj_val)))
        # ----- Enhancement 3: Adaptive Pheromone Evaporation & Multi-Colony Update -----
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
        # Deposit pheromones based on archive crowding distance.
        for idx, (sol, obj_val) in enumerate(archive):
            crowding = normalized_crowding_distance(archive)
            max_cd = np.max(crowding) if len(crowding) > 0 else 1.0
            if not np.isfinite(max_cd) or max_cd <= 0:
                max_cd = 1.0
            decay_factor = 1.0 - (iteration / max_iter)
            deposit = w1 * lambda3 * (crowding[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i, v in enumerate(sol):
                    colony_pheromones[colony_idx][i][v] += deposit
        # Reset pheromones if they become too homogeneous.
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
        # Merge pheromones across colonies to share learned information.
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
        # Compute progress metric via Tchebycheff scores.
        if archive:
            objs = np.array([entry[1] for entry in archive])
            ideal = np.min(objs, axis=0)
            tcheby_scores = [max(abs(entry[1] - ideal)) for entry in archive]
            current_best = min(tcheby_scores)
        else:
            current_best = float('inf')
        progress.append(current_best)
        # ----- Enhancement 3: Diversity-driven Injection -----
        # Check for stagnation and reinitialize a candidate if necessary.
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

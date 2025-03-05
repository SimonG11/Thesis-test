# algorithms.py
import numpy as np
import random, math
from typing import List, Tuple, Callable, Optional, Dict, Any
from utils import chaotic_map_initialization, levy, dominates, round_half, clip_round_half, discretize_vector
from metrics import update_archive_with_crowding, compute_crowding_distance
from objectives import multi_objective


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
            return np.array([self.obj_funcs[0](pos)])
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
                    tasks: List[Dict[str, Any]], workers: Dict[str, int],
                    lb: np.ndarray, ub: np.ndarray, ant_count: int, max_iter: int,
                    alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.1,
                    Q: float = 100.0, P: float = 0.6, w1: float = 1.0, w2: float = 1.0,
                    sigma_share: float = 1.0, lambda3: float = 2.0, lambda4: float = 5.0,
                    colony_count: int = 10) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    MOACO_improved implements a multi-objective Ant Colony Optimization for RCPSP with several enhancements.

    MOACO_improved implements a multi-objective ACO for RCPSP with several enhancements.
    Base algorithm concept from Distributed Optimization by Ant Colonies
    https://www.researchgate.net/publication/216300484_Distributed_Optimization_by_Ant_Colonies

    Enhancements and their scientific justifications:
    1. Chaotic Initialization:
       - Uses a logistic chaotic map to improve the diversity of the initial population.
       - Citation: Sun et al. (2019) "Chaotic Multi-Objective Particle Swarm Optimization Algorithm Incorporating Clone Immunity"
       - https://doi.org/10.3390/math7020146

    2. Adaptive Evaporation:
       - Increases the evaporation rate when the variance of pheromone values is low, preventing premature convergence.
       - Citation: Zhao et al. (2018) (adaptive evaporation approaches in ACO)
       - https://doi.org/10.3390/sym10040104

    3. Ranking-Based Pheromone Deposit Using Crowding Distance:
       - Computes the crowding distance of archive solutions (inspired by NSGA-II [Deb, 2002]) and deposits pheromone proportional to the normalized crowding distance.
       - A decay factor is applied so that deposits diminish over time, shifting the search from exploration to exploitation.
       - Citations: Deb, K. (2002) "Multi-Objective Optimization Using Evolutionary Algorithms" and indicator-based methods (e.g., Zitzler & Künzli, 2004).
       - https://doi.org/10.1109/4235.996017
       - https://doi.org/10.1007/978-3-540-30217-9_84

    4. Multi-Colony Pheromone Updates and Periodic Reinitialization:
       - Maintains separate pheromone matrices per colony and merges them periodically.
       - Helps explore multiple regions of the search space.
       - Citation: Angus & Woodward (2009) for multi-colony ACO approaches.
       - https://doi-org.miman.bib.bth.se/10.1007/s11721-008-0022-4

    5. Local Search and Diversity Injection:
       - Employs extended local search (±1 and ±2 perturbations) and diversity-driven injection of new ants when stagnation is detected.
       - Citation: López-Ibáñez et al. (2012) for extended local search, and Yüzgeç & Kuşoğlu (2020) for diversity-driven injection.
       - https://doi-org.miman.bib.bth.se/10.1007/s11721-012-0070-7
       - https://bseujert.bilecik.edu.tr/index.php/bseujert/article/view/14/11

    Returns:
        archive: A list of non-dominated solutions (decision and objective vectors).
        progress: A list recording the best objective value (e.g., makespan) per iteration.
    """
    dim = len(lb)
    # Initialize pheromone and heuristic information for each colony.
    colony_pheromones = []
    colony_heuristics = []
    for _ in range(colony_count):
        pheromone = []
        heuristic = []
        for i in range(dim):
            possible_values = list(np.arange(lb[i], ub[i] + 0.5, 0.5))
            pheromone.append({v: 1.0 for v in possible_values})
            h_dict = {}
            task = tasks[i]
            for v in possible_values:
                new_effort = task["base_effort"] * (1 + (1.0 / task["max"]) * (v - 1))
                duration = new_effort / v
                h_dict[v] = np.nan_to_num(1.0 / duration, nan=0.0, posinf=0.0, neginf=0.0)
            heuristic.append(h_dict)
        colony_pheromones.append(pheromone)
        colony_heuristics.append(heuristic)
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    best_global = float('inf')
    no_improvement_count = 0
    stagnation_threshold = 10  # Inject diversity if no improvement for 10 iterations.
    eps = 1e-6

    for iteration in range(max_iter):
        colony_solutions = []
        # --- Solution Construction and Extended Local Search ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            for _ in range(ants_per_colony):
                solution: List[float] = []
                for i in range(dim):
                    possible_values = list(pheromone[i].keys())
                    probs = []
                    for v in possible_values:
                        tau = np.nan_to_num(pheromone[i][v], nan=0.0, posinf=0.0, neginf=0.0)
                        h_val = np.nan_to_num(heuristic[i][v], nan=0.0, posinf=0.0, neginf=0.0)
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
                # Extended Local Search: try ±0.5 perturbations.
                neighbors = []
                for i in range(dim):
                    for delta in [-0.5, 0.5]:
                        neighbor = solution.copy()
                        neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
                        neighbors.append(neighbor)
                def compare_objs(obj_a, obj_b):
                    if dominates(obj_a, obj_b):
                        return True
                    elif not dominates(obj_b, obj_a) and np.sum(obj_a) < np.sum(obj_b):
                        return True
                    return False
                best_neighbor = solution
                best_obj = objf(np.array(solution))
                for neighbor in neighbors:
                    n_obj = objf(np.array(neighbor))
                    if compare_objs(n_obj, best_obj):
                        best_obj = n_obj
                        best_neighbor = neighbor
                # If no improvement with ±0.5, try ±1 perturbations.
                if best_neighbor == solution:
                    extended_neighbors = []
                    for i in range(dim):
                        for delta in [-2.0, 2.0]:
                            neighbor = solution.copy()
                            neighbor[i] = clip_round_half(neighbor[i] + delta, lb[i], ub[i])
                            extended_neighbors.append(neighbor)
                    for neighbor in extended_neighbors:
                        n_obj = objf(np.array(neighbor))
                        if compare_objs(n_obj, best_obj):
                            best_obj = n_obj
                            best_neighbor = neighbor
                solution = best_neighbor
                obj_val = objf(np.array(solution))
                colony_solutions.append((solution, obj_val))
        # Update archive using NSGA-II crowding distance mechanism.
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), obj_val))
        
        # --- Adaptive Evaporation ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.nan_to_num(np.array(all_values), nan=0.0, posinf=0.0, neginf=0.0)
            var_pheromone = np.var(all_values)
            if var_pheromone < 0.001:
                evap_rate_current = min(0.9, evaporation_rate * 1.5)
            else:
                evap_rate_current = evaporation_rate
            for i in range(dim):
                for v in pheromone[i]:
                    pheromone[i][v] *= (1 - evap_rate_current)
        
        # --- Ranking-based Deposit Update Using Crowding Distance ---
        crowding = compute_crowding_distance(archive)
        max_cd = np.max(crowding) if len(crowding) > 0 else 1.0
        if not np.isfinite(max_cd) or max_cd <= 0:
            max_cd = 1.0
        decay_factor = 1.0 - (iteration / max_iter)
        for idx, (sol, obj_val) in enumerate(archive):
            deposit = w1 * lambda3 * (crowding[idx] / (max_cd + eps)) * decay_factor
            for colony_idx in range(colony_count):
                for i, v in enumerate(sol):
                    colony_pheromones[colony_idx][i][v] += deposit
        
        # --- Multi-Colony Pheromone Reinitialization ---
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            all_values = np.nan_to_num(np.array(all_values), nan=0.0, posinf=0.0, neginf=0.0)
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
        
        # --- Progress Update & Diversity Injection ---
        current_best = min(obj_val[0] for _, obj_val in colony_solutions)
        progress.append(current_best)
        if current_best < best_global:
            best_global = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= stagnation_threshold:
            for colony_idx in range(colony_count):
                num_to_reinit = max(1, ants_per_colony // 10)
                for _ in range(num_to_reinit):
                    new_solution = [random.choice(list(np.arange(lb[i], ub[i] + 0.5, 0.5))) for i in range(dim)]
                    archive = update_archive_with_crowding(archive, (np.array(new_solution), objf(np.array(new_solution))))
            no_improvement_count = 0
    return archive, progress


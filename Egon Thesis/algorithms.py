# algorithms.py
import numpy as np
import random, math
from typing import List, Tuple, Callable, Optional
from utils import chaotic_map_initialization, levy
from metrics import update_archive_with_crowding
from objectives import multi_objective

def MOHHO_with_progress(objf: Callable[[np.ndarray], np.ndarray],
                        lb: np.ndarray, ub: np.ndarray, dim: int,
                        search_agents_no: int, max_iter: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Adaptive MOHHO with chaotic initialization and adaptive escape energy.
    """
    X = chaotic_map_initialization(lb, ub, dim, search_agents_no)
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    t = 0
    while t < max_iter:
        for i in range(search_agents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            f_val = objf(X[i, :])
            archive = update_archive_with_crowding(archive, (X[i, :].copy(), f_val.copy()))
        rabbit = random.choice(archive)[0] if archive else X[0, :].copy()
        E1 = 2 * (1 - (t / max_iter))
        for i in range(search_agents_no):
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
        best_makespan = np.min([objf(X[i, :])[0] for i in range(search_agents_no)])
        progress.append(best_makespan)
        t += 1
    return archive, progress

class PSO:
    """
    Adaptive MOPSO with adaptive inertia, periodic mutation, and crowding-based archive updates.
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
        self.swarm: List[dict] = []
        for _ in range(pop):
            pos = np.array([random.randint(int(self.lb[i]), int(self.ub[i])) for i in range(dim)])
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
            return np.array([self.obj_funcs[0](pos)])
        else:
            return np.array([f(pos) for f in self.obj_funcs])

    def update_archive(self) -> None:
        for particle in self.swarm:
            pos = particle['position'].copy()
            obj_val = particle['obj'].copy()
            from metrics import update_archive_with_crowding
            self.archive = update_archive_with_crowding(self.archive, (pos, obj_val))

    def proportional_distribution(self) -> List[np.ndarray]:
        from metrics import compute_crowding_distance
        if not self.archive:
            return [random.choice(self.swarm)['position'] for _ in range(self.pop)]
        distances = compute_crowding_distance(self.archive)
        total = np.sum(distances)
        if total == 0 or math.isinf(total) or math.isnan(total):
            probs = [1.0 / len(distances)] * len(distances)
        else:
            probs = [d / total for d in distances]
        guides = []
        for _ in range(self.pop):
            r = random.random()
            cum_prob = 0.0
            chosen_idx = len(probs) - 1
            for idx, p in enumerate(probs):
                cum_prob += p
                if r <= cum_prob:
                    chosen_idx = idx
                    break
            guides.append(self.archive[chosen_idx][0])
        return guides

    def jump_improved_operation(self) -> None:
        if len(self.archive) < 2:
            return
        c1, c2 = random.sample(self.archive, 2)
        a1, a2 = random.uniform(0, 1), random.uniform(0, 1)
        oc1 = c1[0] + a1 * (c1[0] - c2[0])
        oc2 = c2[0] + a2 * (c2[0] - c1[0])
        for oc in [oc1, oc2]:
            oc = np.array([int(np.clip(val, self.lb[i], self.ub[i])) for i, val in enumerate(oc)])
            obj_val = self.evaluate(oc)
            from metrics import update_archive_with_crowding
            self.archive = update_archive_with_crowding(self.archive, (oc, obj_val))

    def disturbance_operation(self, particle: dict) -> None:
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

    def move(self) -> None:
        from metrics import update_archive_with_crowding
        self.iteration += 1
        leaders = self.proportional_distribution()
        for idx, particle in enumerate(self.swarm):
            old_pos = particle['position'].copy()
            old_obj = np.linalg.norm(self.evaluate(old_pos))
            r2 = random.random()
            guide = leaders[idx]
            new_v = particle['w'] * particle['velocity'] + self.c2 * r2 * (guide - particle['position'])
            new_v = np.array([np.clip(new_v[i], -self.vmax[i], self.vmax[i]) for i in range(self.dim)])
            particle['velocity'] = new_v
            new_pos = particle['position'] + new_v
            new_pos = np.array([int(np.clip(round(new_pos[i]), self.lb[i], self.ub[i])) for i in range(self.dim)])
            particle['position'] = new_pos
            particle['obj'] = self.evaluate(new_pos)
            particle['pbest'] = new_pos.copy()
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
                particle = self.swarm[idx_to_mutate]
                particle['position'] = np.array([random.randint(int(self.lb[i]), int(self.ub[i])) for i in range(self.dim)])
                particle['obj'] = self.evaluate(particle['position'])
        self.update_archive()

    def run(self, max_iter: Optional[int] = None) -> List[float]:
        if max_iter is None:
            max_iter = self.max_iter
        convergence: List[float] = []
        for _ in range(max_iter):
            self.move()
            best_ms = min(p['obj'][0] for p in self.swarm)
            convergence.append(best_ms)
        return convergence

def MOACO_improved(objf: Callable[[np.ndarray], np.ndarray],
                    tasks: List[dict], workers: dict,
                    lb: np.ndarray, ub: np.ndarray, ant_count: int, max_iter: int,
                    alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.1,
                    Q: float = 100.0, P: float = 0.6, w1: float = 1.0, w2: float = 1.0,
                    sigma_share: float = 1.0, lambda3: float = 2.0, lambda4: float = 5.0,
                    colony_count: int = 2) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    dim = len(lb)
    colony_pheromones = []
    colony_heuristics = []
    for _ in range(colony_count):
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
        colony_pheromones.append(pheromone)
        colony_heuristics.append(heuristic)
    archive: List[Tuple[np.ndarray, np.ndarray]] = []
    progress: List[float] = []
    ants_per_colony = ant_count // colony_count
    for iteration in range(max_iter):
        colony_solutions = []
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            heuristic = colony_heuristics[colony_idx]
            for _ in range(ants_per_colony):
                solution: List[int] = []
                for i in range(dim):
                    possible_values = list(pheromone[i].keys())
                    probs = []
                    for v in possible_values:
                        tau = pheromone[i][v]
                        h_val = heuristic[i][v]
                        probs.append((tau ** alpha) * (h_val ** beta))
                    total = sum(probs)
                    probs = [p / total if total > 0 else 1 / len(probs) for p in probs]
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
                best_obj = objf(np.array(solution))
                for neighbor in neighbors:
                    n_obj = objf(np.array(neighbor))
                    if n_obj[0] < best_obj[0]:
                        best_obj = n_obj
                        best_neighbor = neighbor
                solution = best_neighbor
                obj_val = objf(np.array(solution))
                colony_solutions.append((solution, obj_val))
        for sol, obj_val in colony_solutions:
            archive = update_archive_with_crowding(archive, (np.array(sol), obj_val))
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            for i in range(dim):
                for v in pheromone[i]:
                    pheromone[i][v] *= (1 - evaporation_rate)
            for sol, obj_val in archive:
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
                                count += (1 - d / sigma_share)
                        niche_counts.append(count)
                    min_index = np.argmin(niche_counts)
                    chosen_sol, chosen_obj = archive[min_index]
                    distances = [np.linalg.norm(chosen_obj - other_obj)
                                 for (other_sol, other_obj) in archive
                                 if not np.array_equal(chosen_sol, other_sol)]
                    mu = min(distances) if distances else 0
                    deposit = w2 * (lambda4 if mu > 0 else lambda3)
                for i, v in enumerate(sol):
                    pheromone[i][v] += deposit
        for colony_idx in range(colony_count):
            pheromone = colony_pheromones[colony_idx]
            all_values = []
            for i in range(dim):
                all_values.extend(list(pheromone[i].values()))
            if np.var(all_values) < 0.001:
                for i in range(dim):
                    possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
                    pheromone[i] = {v: 1.0 for v in possible_values}
        merged_pheromone = []
        for i in range(dim):
            merged = {}
            possible_values = list(range(int(lb[i]), int(ub[i]) + 1))
            for v in possible_values:
                val = sum(colony_pheromones[colony_idx][i].get(v, 0) for colony_idx in range(colony_count)) / colony_count
                merged[v] = val
            merged_pheromone.append(merged)
        for colony_idx in range(colony_count):
            colony_pheromones[colony_idx] = [merged_pheromone[i].copy() for i in range(dim)]
        best_ms = min(obj_val[0] for _, obj_val in colony_solutions)
        progress.append(best_ms)
    return archive, progress

import numpy as np
import random, math
from typing import List, Tuple, Dict

DAY_HOURS_FULL = 8    # Full working day in billable hours.
DAY_HOURS_HALF = DAY_HOURS_FULL / 2

def initialize_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

def round_half(x: float) -> float:
    return round(x * 2) / 2.0

def clip_round_half(x: float, lb: float, ub: float) -> float:
    return round_half(np.clip(x, lb, ub))

def discretize_vector(vec: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    # Uses NumPy vectorized operations for clipping and rounding to half steps.
    return np.round(np.clip(vec, lb, ub) * 2) / 2

def dominates(obj_a: np.ndarray, obj_b: np.ndarray, epsilon: float = 1e-6) -> bool:
    less_equal = np.all(obj_a <= obj_b + epsilon)
    strictly_less = np.any(obj_a < obj_b - epsilon)
    return less_equal and strictly_less

def get_true_pareto_points(points: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    nondom_indices = nds.do(points, only_non_dominated_front=True)
    return points[nondom_indices]

def levy(dim: int) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.abs(v) ** (1 / beta))

def find_earliest_start(earliest: float, duration: float, allocated: int,
                        scheduled_tasks: list, capacity: int, resource: str, epsilon: float = 1e-6) -> float:
    tasks_r = [t for t in scheduled_tasks if t.get("resource") == resource]
    if not tasks_r:
        return earliest

    candidate_times = {earliest}
    for task in tasks_r:
        if task["start"] >= earliest:
            candidate_times.add(task["start"])
        if task["finish"] >= earliest:
            candidate_times.add(task["finish"])
    candidate_times = sorted(candidate_times)

    for t in candidate_times:
        events = [t, t + duration]
        for task in tasks_r:
            if task["finish"] > t and task["start"] < t + duration:
                events.extend([task["start"], task["finish"]])
        events = sorted(set(events))
        feasible = True
        for i in range(len(events) - 1):
            mid = (events[i] + events[i+1]) / 2.0
            usage = sum(task["workers"] for task in tasks_r if task["start"] <= mid < task["finish"])
            if usage + allocated > capacity:
                feasible = False
                break
        if feasible:
            return t

    last_finish = max(task["finish"] for task in tasks_r)
    return last_finish + epsilon

def chaotic_map_initialization(lb: np.ndarray, ub: np.ndarray, dim: int, n_agents: int) -> np.ndarray:
    r = 4.0
    x = np.random.rand(n_agents, dim)
    for _ in range(10):
        x = r * x * (1 - x)
    return lb + x * (ub - lb)

def compute_extremes(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                     margin: float = 0.05) -> Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
    try:
        all_objs = np.concatenate([np.array([obj for _, obj in archive]) for archive in archives if archive])
    except ValueError:
        raise ValueError("No objective data found in archives.")
    
    extremes = []
    for m in range(2):
        observed_min = np.min(all_objs[:, m])
        observed_max = np.max(all_objs[:, m])
        range_val = observed_max - observed_min
        ideal_bound = observed_min - margin * range_val
        upper_bound = observed_max + margin * range_val
        extremes.append((ideal_bound, upper_bound))
    
    extremes.append((-1, 0))
    
    new_extremes = {
        (0,1): ((extremes[0][0], extremes[1][1]), (extremes[0][1], extremes[1][1])),
        (0,2): ((extremes[0][0], extremes[2][0]), (extremes[0][1], extremes[2][1])),
        (1,2): ((extremes[1][0], extremes[2][0]), (extremes[1][1], extremes[2][0]))
    }
    
    return new_extremes

def compute_fixed_reference(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
    union_archive = [entry for alg in archives_all for archive in archives_all[alg] for entry in archive]
    if not union_archive:
        raise ValueError("No archive entries found.")
    objs = np.array([entry[1] for entry in union_archive])
    return np.max(objs, axis=0)

def compute_combined_ideal(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
    union_archive = [entry for alg in archives_all for archive in archives_all[alg] for entry in archive]
    if not union_archive:
        raise ValueError("No archive entries found.")
    objs = np.array([entry[1] for entry in union_archive])
    return np.min(objs, axis=0)

def same_entry(entry1: Tuple[np.ndarray, np.ndarray], entry2: Tuple[np.ndarray, np.ndarray]) -> bool:
    return np.array_equal(entry1[0], entry2[0]) and np.array_equal(entry1[1], entry2[1])

def compute_crowding_distance(archive: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not archive:
        return np.array([])
    objs = np.array([entry[1] for entry in archive])
    num_objs = objs.shape[1]
    distances = np.zeros(len(archive))
    for m in range(num_objs):
        sorted_idx = np.argsort(objs[:, m])
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = float('inf')
        m_values = objs[sorted_idx, m]
        m_range = m_values[-1] - m_values[0]
        if m_range == 0:
            continue
        distances[sorted_idx[1:-1]] += (m_values[2:] - m_values[:-2]) / m_range
    return distances

def update_archive_with_crowding(archive: List[Tuple[np.ndarray, np.ndarray]],
                                 new_entry: Tuple[np.ndarray, np.ndarray],
                                 max_archive_size: int = 100,
                                 epsilon: float = 1e-6) -> List[Tuple[np.ndarray, np.ndarray]]:
    _, obj_new = new_entry
    dominated_flag = False
    removal_list = []
    for (sol_arch, obj_arch) in archive:
        if dominates(obj_arch, obj_new, epsilon):
            dominated_flag = True
            break
        if dominates(obj_new, obj_arch, epsilon):
            removal_list.append((sol_arch, obj_arch))
    if not dominated_flag:
        archive = [entry for entry in archive if not any(same_entry(entry, rem) for rem in removal_list)]
        archive.append(new_entry)
        if len(archive) > max_archive_size:
            distances = compute_crowding_distance(archive)
            min_index = np.argmin(distances)
            archive.pop(min_index)
    return archive

def get_global_non_dominated(solutions: List[Tuple[int, np.ndarray, np.ndarray]], 
                             epsilon: float = 1e-6) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    objs = np.array([sol[2] for sol in solutions])
    nondom_indices = nds.do(objs, only_non_dominated_front=True)
    return [solutions[i] for i in nondom_indices]

def convert_hours_to_billable_days(duration: float) -> float:
    if duration <= 0:
        return 0.0
    full_days = int(duration // DAY_HOURS_FULL)
    remainder = duration - full_days * DAY_HOURS_FULL
    if remainder == 0:
        extra = 0.0
    elif remainder <= DAY_HOURS_HALF:
        extra = 0.5
    else:
        extra = 1.0
    return full_days + extra

def compute_billable_hours(duration: float) -> float:
    billable_days = convert_hours_to_billable_days(duration)
    full_days = int(billable_days)
    half_day = 1 if (billable_days - full_days) >= 0.5 else 0
    return full_days * DAY_HOURS_FULL + half_day * DAY_HOURS_HALF

def compute_billable_cost(duration: float, allocation: float, wage_rate: float) -> float:
    F = compute_billable_hours(duration)
    full_cost = F * wage_rate
    if allocation < 1:
        if duration < 2:
            return full_cost / 2
        else:
            if F <= DAY_HOURS_FULL:
                return full_cost
            else:
                discount = (DAY_HOURS_HALF * wage_rate) / 2
                return full_cost - discount
    else:
        full_workers = int(allocation)
        fractional = allocation - full_workers
        if fractional >= 0.5:
            return full_workers * full_cost + full_cost / 2
        else:
            return full_workers * full_cost

import numpy as np
import random, math
from typing import List, Tuple, Dict


# Named constants for clarity.
DAY_HOURS_FULL = 8    # Full working day in billable hours.
DAY_HOURS_HALF = DAY_HOURS_FULL / 2    # Half working day in billable hours.


def initialize_seed(seed: int = 42) -> None:
    """Initialize random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def round_half(x: float) -> float:
    """
    Round a number to the nearest half step (e.g., 1, 1.5, 2, 2.5, ...).
    This function ensures that worker allocations remain in the allowed discrete set.
    """
    return round(x * 2) / 2.0


def clip_round_half(x: float, lb: float, ub: float) -> float:
    """Clip a value between lb and ub and round it to the nearest half step."""
    return round_half(np.clip(x, lb, ub))


def discretize_vector(vec: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Discretize each element of a vector to half steps within given bounds."""
    return np.array([clip_round_half(val, lb[i], ub[i]) for i, val in enumerate(vec)])


def dominates(obj_a: np.ndarray, obj_b: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if solution A dominates solution B in a minimization context.
    Two objective values are considered equal if their difference is less than epsilon.
    
    A dominates B if every objective in A is <= corresponding objective in B,
    with at least one strictly less (by more than epsilon).
    """
    less_equal = np.all(obj_a <= obj_b + epsilon)
    strictly_less = np.any(obj_a < obj_b - epsilon)
    return less_equal and strictly_less


def get_true_pareto_points(points: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Given a set of points (each row is an objective vector), return only the non-dominated points.
    
    Parameters:
      points: A NumPy array of shape (n_points, n_objectives).
      epsilon: Tolerance for comparing objectives.
    
    Returns:
      A NumPy array containing only the non-dominated (true Pareto front) points.
    """
    n = points.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(points[j], points[i], epsilon):
                dominated[i] = True
                break
    return points[~dominated]


def levy(dim: int) -> np.ndarray:
    """
    Compute a Levy flight step for a given dimensionality.
    """
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.power(np.abs(v), 1 / beta))


def find_earliest_start(earliest: float, duration: float, allocated: int,
                        scheduled_tasks: list, capacity: int, resource: str, epsilon: float = 1e-6) -> float:
    """
    Determine the earliest feasible start time for a task given resource constraints.
    """
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
    """
    Initialize the population using a logistic chaotic map.
    """
    r = 4.0
    population = np.zeros((n_agents, dim))
    for i in range(n_agents):
        x = np.random.rand(dim)
        for _ in range(10):
            x = r * x * (1 - x)
        population[i, :] = lb + x * (ub - lb)
    return population


def compute_extremes(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                     margin: float = 0.05) -> List[Tuple[float, float]]:
    """
    Compute extreme bounds for each objective from a list of archives.
    
    For objectives 0 (cost) and 1 (makespan), combine all objective vectors,
    find the minimum and maximum values, and extend these by a given margin.
    For objective 2 (resource utilization, which is negated), return the fixed extremes (-1, 0).
    
    Parameters:
        archives: A list of archives, each being a list of (decision_vector, objective_vector) tuples.
        margin: A fraction (e.g., 0.05 for 5%) by which to extend the observed range.
    
    Returns:
        A list of three tuples: [(ideal0, upper0), (ideal1, upper1), (ideal2, upper2)],
        where ideal_m is the lower (ideal) bound for objective m and upper_m is the upper bound.
    """
    # Collect all objective vectors for cost and makespan (objectives 0 and 1).
    all_objs = []
    for archive in archives:
        for _, obj in archive:
            all_objs.append(obj)
    if not all_objs:
        raise ValueError("No objective data found in archives.")
    
    all_objs = np.array(all_objs)  # shape (N, 3)
    
    extremes = []
    # For objective 0 (cost) and objective 1 (makespan)
    for m in range(2):
        observed_min = np.min(all_objs[:, m])
        observed_max = np.max(all_objs[:, m])
        range_val = observed_max - observed_min
        # Extend the bounds by margin * range
        ideal_bound = observed_min - margin * range_val
        upper_bound = observed_max + margin * range_val
        extremes.append((ideal_bound, upper_bound))
    
    # For objective 2 (resource utilization, which is negated), we fix extremes.
    extremes.append((-1, 0))

    new_extremes = {(0,1) : ((extremes[0][0], extremes[1][1]), (extremes[0][1], extremes[1][1])),
                   (0,2) : ((extremes[0][0], extremes[2][0]),(extremes[0][1], extremes[2][1])),
                   (1,2) : ((extremes[1][0], extremes[2][0]),(extremes[1][1], extremes[2][0]))
                   }
    
    return new_extremes


def compute_fixed_reference(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
    """
    Compute a fixed reference point based on the union of all solution archives.
    """
    union_archive = []
    for alg in archives_all:
        for archive in archives_all[alg]:
            union_archive.extend(archive)
    if not union_archive:
        raise ValueError("No archive entries found.")
    objs = np.array([entry[1] for entry in union_archive])
    ref_point = np.max(objs, axis=0)
    return ref_point


def compute_combined_ideal(archives_all: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]]) -> np.ndarray:
    """
    Compute the combined ideal point from multiple archives.
    """
    union_archive = []
    for alg in archives_all:
        for archive in archives_all[alg]:
            union_archive.extend(archive)
    if not union_archive:
        raise ValueError("No archive entries found.")
    objs = np.array([entry[1] for entry in union_archive])
    ideal = np.min(objs, axis=0)
    return ideal


def same_entry(entry1: Tuple[np.ndarray, np.ndarray], entry2: Tuple[np.ndarray, np.ndarray]) -> bool:
    """Return True if two archive entries are identical."""
    return np.array_equal(entry1[0], entry2[0]) and np.array_equal(entry1[1], entry2[1])


def compute_crowding_distance(archive: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Compute the crowding distance for each solution in the archive.
    """
    if not archive:
        return np.array([])
    objs = np.array([entry[1] for entry in archive])
    num_objs = objs.shape[1]
    distances = np.zeros(len(archive))
    for m in range(num_objs):
        sorted_indices = np.argsort(objs[:, m])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
        m_values = objs[sorted_indices, m]
        m_range = m_values[-1] - m_values[0]
        if m_range == 0:
            continue
        for i in range(1, len(archive) - 1):
            distances[sorted_indices[i]] += (m_values[i+1] - m_values[i-1]) / m_range
    return distances


def update_archive_with_crowding(archive: List[Tuple[np.ndarray, np.ndarray]],
                                 new_entry: Tuple[np.ndarray, np.ndarray],
                                 max_archive_size: int = 100,
                                 epsilon: float = 1e-6) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Update the non-dominated archive with a new entry while preserving diversity.
    """
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
    """
    Given a list of solutions tagged with an algorithm index, return only the non-dominated ones.
    
    Each solution is a tuple (alg_index, decision_vector, objective_vector).
    The function loops over every pair and marks a solution as dominated if any other solution dominates it.
    
    Returns a list of solutions that are non-dominated globally.
    """
    n = len(solutions)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                # Compare the objective vectors.
                if dominates(solutions[j][2], solutions[i][2], epsilon):
                    dominated[i] = True
                    break
    return [solutions[i] for i in range(n) if not dominated[i]]


def convert_hours_to_billable_days(duration: float) -> float:
    """
    Convert a duration (in hours) to billable days.
    Business Rules:
      - Non-positive durations return 0.0 days.
      - Durations ≤ DAY_HOURS_HALF (4 hours) count as a half day (0.5).
      - Durations > DAY_HOURS_HALF and ≤ DAY_HOURS_FULL (8 hours) count as a full day (1.0).
      - For durations > DAY_HOURS_FULL:
            Compute full days = duration // DAY_HOURS_FULL.
            Let remainder = duration - (full_days * DAY_HOURS_FULL).
            If remainder is ≤ DAY_HOURS_HALF, add 0.5 day; otherwise, add 1.0 day.
    """
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
    """
    Convert a duration (in hours) to billable hours.
    Billing Conversion:
      - Each full day is billed as DAY_HOURS_FULL hours.
      - Each half day is billed as DAY_HOURS_HALF hours.
    """
    billable_days = convert_hours_to_billable_days(duration)
    full_days = int(billable_days)
    half_day = 1 if (billable_days - full_days) >= 0.5 else 0
    return full_days * DAY_HOURS_FULL + half_day * DAY_HOURS_HALF


def compute_billable_cost(duration: float, allocation: float, wage_rate: float) -> float:
    """
    Compute the total labor cost given a task's duration (in hours), worker allocation, and wage rate.
    """
    F = compute_billable_hours(duration)
    full_cost = F * wage_rate
    if allocation < 1:
        # Pure half worker.
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

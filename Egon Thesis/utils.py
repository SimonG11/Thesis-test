# utils.py
import numpy as np
import random, math
from typing import List, Tuple


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


def convertDurationtodays(duration):
    days = 0 
    while duration >= 0:
        if duration > 4:
            duration -= 8
            days +=1
        else:
            duration -= 4
            days +=0.5
    return days


def convertDurationtodaysCost(duration, alloc):
    ActualAcualEffort = 0
    if int(alloc) == alloc:    
        while duration >= 0:
            if duration > 4:
                duration -= 8
                ActualAcualEffort += 8
            else:
                duration -= 4
                #days +=0.5
                ActualAcualEffort +=4
        return ActualAcualEffort * alloc 
    else:
        alloc -= 0.5
        halfduration = duration / 2 
        while duration >= 0:
            if duration > 4:
                duration -= 8
                ActualAcualEffort += 8
            else:
                duration -= 4
                #days +=0.5
                ActualAcualEffort +=4
        ActualAcualEffort *= alloc
        ActualAcualEffort += ((halfduration // 4)*4)
        if (halfduration % 4) != 0:
            ActualAcualEffort += 4
        return ActualAcualEffort
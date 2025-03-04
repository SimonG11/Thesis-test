# metrics.py
import numpy as np
from typing import List, Tuple, Dict
from utils import dominates


def approximate_hypervolume(archive: List[Tuple[np.ndarray, np.ndarray]],
                            reference_point: np.ndarray,
                            num_samples: int = 1000) -> float:
    """
    Approximate the hypervolume of the archive via Monte Carlo sampling.
    """
    if not archive:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    mins = np.min(objs, axis=0)
    samples = np.random.uniform(low=mins, high=reference_point, size=(num_samples, len(reference_point)))
    count = sum(1 for sample in samples if any(np.all(sol <= sample) for sol in objs))
    vol = np.prod(reference_point - mins)
    return (count / num_samples) * vol


def absolute_hypervolume_fixed(archive: List[Tuple[np.ndarray, np.ndarray]],
                               reference_point: np.ndarray,
                               global_lower_bound: np.ndarray,
                               num_samples: int = 1000) -> float:
    """
    Approximate the absolute hypervolume using a fixed global lower bound.
    """
    if not archive:
        return 0.0
    lb = global_lower_bound
    ub = reference_point
    samples = np.random.uniform(low=lb, high=ub, size=(num_samples, len(ub)))
    objs = np.array([entry[1] for entry in archive])
    dominated_count = sum(1 for sample in samples if any(np.all(sol <= sample) for sol in objs))
    return (dominated_count / num_samples) * 100


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


def same_entry(entry1: Tuple[np.ndarray, np.ndarray], entry2: Tuple[np.ndarray, np.ndarray]) -> bool:
    """Return True if two archive entries are identical."""
    return np.array_equal(entry1[0], entry2[0]) and np.array_equal(entry1[1], entry2[1])


def update_archive_with_crowding(archive: List[Tuple[np.ndarray, np.ndarray]],
                                 new_entry: Tuple[np.ndarray, np.ndarray],
                                 max_archive_size: int = 50,
                                 epsilon: float = 1e-6) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Update the non-dominated archive with a new entry while preserving diversity.
    """
    sol_new, obj_new = new_entry
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


def compute_generational_distance(archive: List[Tuple[np.ndarray, np.ndarray]],
                                  true_pareto: np.ndarray) -> float:
    """
    Compute the Generational Distance (GD) between the archive and the true Pareto front.
    """
    if not archive or true_pareto.size == 0:
        return float('inf')
    objs = np.array([entry[1] for entry in archive])
    distances = [np.min(np.linalg.norm(true_pareto - sol, axis=1)) for sol in objs]
    return np.mean(distances)


def compute_spread(archive: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Compute the spread (diversity) of the archive.
    """
    if len(archive) < 2:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    dists = [np.linalg.norm(objs[i] - objs[j]) for i in range(len(objs)) for j in range(i+1, len(objs))]
    return np.mean(dists)


def compute_spread_3obj_with_extremes(
    archive: List[Tuple[np.ndarray, np.ndarray]], 
    extremes: List[Tuple[float, float]],
    epsilon: float = 1e-6
) -> float:
    """
    Compute the Δ spread metric for an archive with three objectives, incorporating extreme values.
    
    For each objective m (m = 0, 1, 2):
      - Let extremes[m] = (ideal_m, upper_m), where ideal_m is the ideal (best) bound and upper_m is the upper bound.
      - Extract the sorted objective values: f1, f2, ..., f_n.
      - Compute:
          d_f = f1 - ideal_m      (gap from ideal to best solution)
          d_l = upper_m - f_n      (gap from worst solution to upper bound)
          d_i = f(i+1) - f(i) for i = 1, …, n-1
          d_avg = average of all d_i
      - Then compute:
          Δ_m = (d_f + d_l + Σ |d_i - d_avg|) / (d_f + d_l + (n-1) * d_avg)
    
    The overall spread is the average of Δ_m for m = 0, 1, 2.
    
    Parameters:
        archive: A list of tuples (decision_vector, objective_vector). Each objective_vector is a 3-element NumPy array.
        extremes: A list of 3 tuples. For each objective m, extremes[m] = (ideal_m, upper_m).
        epsilon: Tolerance for numerical comparisons.
    
    Returns:
        A float representing the average spread over the three objectives.
    """
    # If there is only one solution, no spread exists.
    if len(archive) < 2:
        return 0.0
    
    # Extract objective vectors (n x 3 array).
    objs = np.array([entry[1] for entry in archive])
    n = objs.shape[0]
    delta_sum = 0.0

    for m in range(3):
        # Get ideal and upper for objective m.
        ideal_m, upper_m = extremes[m]
        
        # Extract the m-th objective values and sort them.
        sorted_vals = np.sort(objs[:, m])
        
        # Compute gap from ideal to the best solution.
        d_f = sorted_vals[0] - ideal_m
        
        # Compute gap from the worst solution to the upper bound.
        d_l = upper_m - sorted_vals[-1]
        
        # Compute distances between consecutive sorted values.
        dists = [abs(sorted_vals[i+1] - sorted_vals[i]) for i in range(n - 1)]
        d_avg = np.mean(dists) if dists else 0.0
        
        # Compute the sum of absolute deviations from the average gap.
        deviation_sum = sum(abs(d - d_avg) for d in dists)
        
        # Avoid division by zero.
        if d_avg == 0:
            delta_m = 0.0
        else:
            delta_m = (d_f + d_l + deviation_sum) / (d_f + d_l + (n - 1) * d_avg)
        
        delta_sum += delta_m

    # Return the average spread over all three objectives.
    return delta_sum / len(extremes)


def compute_coverage(setA: List[Tuple[np.ndarray, np.ndarray]], 
                     setB: List[Tuple[np.ndarray, np.ndarray]], 
                     epsilon: float = 1e-6) -> float:
    """
    Compute the pairwise set coverage (C-metric) between two Pareto front approximations.
    
    Given two archives setA and setB (each a list of (decision_vector, objective_vector) tuples),
    C(setA, setB) is defined as the fraction of solutions in setB that are dominated by at least one
    solution in setA.
    
    Returns:
        A float between 0 and 1 representing the coverage of setA over setB.
    """
    if len(setB) == 0:
        return 0.0
    
    dominated_count = 0
    for decision_b, obj_b in setB:
        for decision_a, obj_a in setA:
            if dominates(obj_a, obj_b, epsilon):
                dominated_count += 1
                break  # No need to check other solutions in setA
    return dominated_count / len(setB)


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


def normalized_hypervolume_fixed(archive: List[Tuple[np.ndarray, np.ndarray]], fixed_ref: np.ndarray) -> float:
    """
    Compute the normalized hypervolume as a percentage using a fixed reference point.
    """
    if not archive:
        return 0.0
    objs = np.array([entry[1] for entry in archive])
    ideal = np.min(objs, axis=0)
    total_volume = np.prod(fixed_ref - ideal)
    if total_volume == 0:
        return 0.0
    hv = approximate_hypervolume(archive, reference_point=fixed_ref)
    return (hv / total_volume) * 100.0

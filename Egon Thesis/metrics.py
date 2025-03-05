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


def compute_spread_2d(points_2d: np.ndarray, 
                      extremes: Tuple[Tuple[float, float], Tuple[float, float]], 
                      epsilon: float = 1e-6) -> float:
    """
    Compute the Δ spread metric for a set of 2D points.
    
    For a set of points in a 2D objective space, and given extremes as:
      - ideal: (ideal_x, ideal_y)
      - upper: (upper_x, upper_y)
    we do the following:
      1. Sort the points by the first coordinate.
      2. Compute Euclidean distances between consecutive sorted points.
      3. Compute d_f = distance from ideal to the first point, and 
         d_l = distance from the last point to the upper bound.
      4. Compute the average of the consecutive distances d_avg.
      5. Compute the sum of absolute deviations: deviation_sum = Σ |d_i – d_avg|.
      6. Return the spread as:
           Δ = (d_f + d_l + deviation_sum) / (d_f + d_l + (N - 1) * d_avg)
         If (N - 1) * d_avg is 0, return 0.
         
    A lower Δ indicates a more uniform distribution.
    """
    N = points_2d.shape[0]
    if N < 2:
        return 0.0
    # Sort points by the first coordinate (you could also use lexicographic order).
    sorted_indices = np.argsort(points_2d[:, 0])
    sorted_points = points_2d[sorted_indices]
    
    # Compute Euclidean distances between consecutive sorted points.
    dists = [np.linalg.norm(sorted_points[i+1] - sorted_points[i]) for i in range(N - 1)]
    d_avg = np.mean(dists) if dists else 0.0
    
    # Compute boundary distances:
    ideal, upper = extremes  # ideal and upper are 2D tuples
    d_f = np.linalg.norm(sorted_points[0] - np.array(ideal))
    d_l = np.linalg.norm(np.array(upper) - sorted_points[-1])
    print(extremes, sorted_points[0])
    print(d_f, d_l)
    deviation_sum = sum(abs(d - d_avg) for d in dists)
    print(deviation_sum)
    denominator = d_f + d_l + (N - 1) * d_avg
    print(denominator)
    if denominator < epsilon:
        return 0.0
    return (d_f + d_l + deviation_sum) / denominator


def compute_spread_3d_by_projections(archive: List[Tuple[np.ndarray, np.ndarray]], 
                                     extremes_proj: dict, 
                                     epsilon: float = 1e-6) -> float:
    """
    Compute a 3D spread metric by projecting the archive's objective vectors into three 2D spaces,
    computing a 2D spread metric for each, and averaging the results.
    
    Parameters:
      archive: List of (decision_vector, objective_vector) tuples. Each objective_vector is a 3-element array.
      extremes_proj: A dictionary where keys are 2-tuples representing the projection indices, e.g.
                    (0,1), (0,2), (1,2), and each value is a tuple of extreme pairs:
                    ((ideal_i, ideal_j), (upper_i, upper_j)).
                    
      For example:
          {
             (0,1): ((ideal0, ideal1), (upper0, upper1)),
             (0,2): ((ideal0, ideal2), (upper0, upper2)),
             (1,2): ((ideal1, ideal2), (upper1, upper2))
          }
    
    Returns:
      The average spread across the three 2D projections.
    """
    if len(archive) < 2:
        return 0.0
    objs = np.array([entry[1] for entry in archive])  # shape (N, 3)
    projections = [(0,1), (0,2), (1,2)]
    spread_values = []
    
    for proj in projections:
        i, j = proj
        # Extract the 2D projection.
        points_2d = objs[:, [i, j]]
        if proj not in extremes_proj:
            raise ValueError(f"Extremes not provided for projection {proj}")
        ext = extremes_proj[proj]
        spread_2d = compute_spread_2d(points_2d, ext, epsilon)
        spread_values.append(spread_2d)
    
    return np.mean(spread_values)


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
                # No need to check other solutions in setA
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

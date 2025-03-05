import numpy as np
from typing import List, Tuple

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
    print(d_f, d_l)
    deviation_sum = sum(abs(d - d_avg) for d in dists)
    denominator = d_f + d_l + (N - 1) * d_avg
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

# --------------------- Example Testing ---------------------
if __name__ == "__main__":
    # Example archive: three solutions, each with three objectives.
    archive = [
        (np.array([0.1, 0.2]), np.array([1, 2, -1])),
        (np.array([0.3, 0.4]), np.array([1.5, 1.5, -0.5])),
        (np.array([0.5, 0.6]), np.array([2, 1, 0]))
    ]
    
    # Define extreme bounds for each 2D projection.
    # For projection (0,1): for instance, cost and makespan.
    # Here, assume:
    #   ideal for (0,1) is (99.5, 239.5), upper is (110.5, 250.5).
    # For projection (0,2): (cost, resource utilization)
    #   ideal for (0,2) is (99.5,  -1) [because resource utilization is negated, ideal becomes -1],
    #   upper is (110.5,  0).
    # For projection (1,2): (makespan, resource utilization)
    #   ideal for (1,2) is (239.5, -1), upper is (250.5, 0).
    extremes_proj = {
        (0, 1): ((1, 2), (2, 1)),
        (0, 2): ((1, -1), (2, 0)),
        (1, 2): ((1, 0), (2, -1))
    }
    
    spread_avg = compute_spread_3d_by_projections(archive, extremes_proj)
    print("Computed average 2D-projection spread for 3 objectives:", spread_avg)

import numpy as np
from typing import List, Tuple

def absolute_hypervolume_fixed(archive: List[Tuple[np.ndarray, np.ndarray]],
                                  global_lower_bound: np.ndarray,
                                  reference_point: np.ndarray,
                                  num_samples: int = 100) -> float:
    """
    Approximate the absolute hypervolume of the archive using a fixed global lower bound.
    
    In this approach, we define a fixed box that spans from a predetermined lower bound (global_lower_bound)
    to a fixed reference point. The absolute hypervolume is then estimated as the total volume of this box that is
    dominated by the solutions in the archive.
    
    Parameters:
        archive: A list of tuples, where each tuple is (decision_vector, objective_vector).
                 The objective vectors are assumed to be for minimization.
        global_lower_bound: A fixed lower bound for the objective space (e.g., np.array([0, 0, 0])).
        reference_point: A fixed referengice point (upper bound) for the objective space.
        num_samples: Number of random samples used in the Monte Carlo approximation.
    
    Returns:
        The approximate absolute hypervolume (raw dominated volume) as a float.
    
    How it works:
      1. It uses the fixed global_lower_bound and the fixed reference_point to define the full search space.
      2. It generates random samples uniformly from this full box.
      3. For each sample, it checks if any solution in the archive dominates that sample.
      4. The hypervolume is then estimated as the fraction of dominated samples multiplied by the total volume of the box.
    """
    if not archive:
        return 0.0

    # Define the full box using the fixed global lower bound.
    lb = global_lower_bound
    ub = reference_point

    # Generate random samples within the box [lb, ub].
    samples = np.random.uniform(low=lb, high=ub, size=(num_samples, len(ub)))
    
    # Extract the objective vectors from the archive.
    objs = np.array([entry[1] for entry in archive])
    
    # Count samples that are dominated by at least one solution in the archive.
    dominated_count = sum(1 for sample in samples if any(np.all(sol <= sample) for sol in objs))
    
    # Calculate the total volume of the box.
    total_volume = np.prod(ub - lb)
    
    # Estimate the absolute hypervolume.
    hypervolume = (dominated_count / num_samples) * 100
    return hypervolume
def compute_combined_ideal(archives: List[List[Tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    """
    Compute the combined ideal point from multiple archives.
    
    The ideal point is the element-wise minimum of all objective vectors across all archives.
    
    Parameters:
        archives: A list where each element is an archive (a list of (decision_vector, objective_vector) tuples).
    
    Returns:
        A NumPy array representing the combined ideal point.
    """
    # Gather all objective vectors from every archive
    all_objs = []
    for archive in archives:
        for decision, obj in archive:
            all_objs.append(obj)
    
    if not all_objs:
        raise ValueError("No objective data found in the archives.")
    
    all_objs = np.array(all_objs)
    # Compute the element-wise minimum (ideal) across all objective vectors
    ideal = np.min(all_objs, axis=0)
    return ideal
# --------------------- Simple Test Harness ---------------------
if __name__ == "__main__":
    # Create a simple test archive.
    archive = [
        (np.array([0.1, 0.2]), np.array([100, 250])),
        (np.array([0.3, 0.4]), np.array([110, 240])),
        (np.array([0.5, 0.6]), np.array([105, 245]))
    ]
    archive2 = [
        (np.array([0.1, 0.2]), np.array([120, 250])),
        (np.array([0.3, 0.4]), np.array([130, 240])),
        (np.array([0.5, 0.6]), np.array([100, 245]))
    ]
    
    # Define a fixed global lower bound (e.g., best possible or known lower limits).
    global_lower_bound = compute_combined_ideal([archive, archive2])
    
    # Define a fixed reference point that is worse than all solutions.
    reference_point = np.array([130, 260])
    
    # Calculate and print the absolute hypervolume using the fixed box.
    hv_fixed = absolute_hypervolume_fixed(archive, global_lower_bound, reference_point, num_samples=10000)
    print("Absolute Hypervolume (fixed lower bound) =", hv_fixed)
    hv_fixed = absolute_hypervolume_fixed(archive2, global_lower_bound, reference_point, num_samples=10000)
    print("Absolute Hypervolume (fixed lower bound) =", hv_fixed)

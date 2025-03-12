# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional,Dict
from utils import get_true_pareto_points, get_global_non_dominated


def fit_quadratic_surface_to_points(points: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Fit a quadratic surface of the form:
    
        z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    
    to a set of 3D points using least-squares.
    
    Parameters:
        points: A NumPy array of shape (n_points, 3) representing points in 3D.
    
    Returns:
        A tuple of coefficients (a, b, c, d, e, f).
    """
    # Extract x, y, z coordinates from the points.
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    
    # Build the design matrix: each row is [x^2, y^2, x*y, x, y, 1].
    A = np.column_stack([X**2, Y**2, X*Y, X, Y, np.ones_like(X)])
    
    # Solve the least-squares problem.
    coeffs, residuals, rank, s = np.linalg.lstsq(A, -Z, rcond=None)
    return tuple(coeffs)


def plot_quadratic_surface(quad_coeffs: Tuple[float, float, float, float, float, float],
                           x_limits: Tuple[float, float],
                           y_limits: Tuple[float, float],
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the quadratic surface defined by
         z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    over a grid defined by x_limits and y_limits.
    
    Parameters:
        quad_coeffs: A tuple (a, b, c, d, e, f) defining the surface.
        x_limits: Tuple (x_min, x_max) for the grid.
        y_limits: Tuple (y_min, y_max) for the grid.
        ax: An optional matplotlib 3D axis; if None, a new figure and axis will be created.
    
    Returns:
        The matplotlib axis with the plotted surface.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    a, b, c, d, e, f = quad_coeffs
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    # Create a grid over the given limits.
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 30),
                                 np.linspace(y_min, y_max, 30))
    # Compute z values from the quadratic surface.
    z_grid = a * x_grid**2 + b * y_grid**2 + c * x_grid * y_grid + d * x_grid + e * y_grid + f
    ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.3)
    return ax


def fit_plane_to_points(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a plane z = a*x + b*y + c to a set of points.
    
    Parameters:
        points: An array of shape (n_points, 3) representing points in 3D space.
        
    Returns:
        A tuple (a, b, c) such that the best-fit plane is given by z = a*x + b*y + c.
    """
    # Assume points is an array with columns: x, y, z.
    # We want to solve: z = a*x + b*y + c
    # Formulate the linear system: [x, y, 1] * [a, b, c].T = z.
    X = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    z = points[:, 2]
    # Solve the least squares problem:
    coeffs, residuals, rank, s = np.linalg.lstsq(X, z, rcond=None)
    a, b, c = coeffs
    return a, b, c


def plot_gantt(schedule: List[dict], title: str) -> None:
    """Plot a Gantt chart for the given schedule."""
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks, yticklabels = [], []
    for i, task in enumerate(schedule):
        ax.broken_barh([(task["start"], task["duration"])],
                       (i * 10, 9),
                       facecolors='tab:blue')
        yticks.append(i * 10 + 5)
        yticklabels.append(f"Task {task['task_id']}: {task['task_name']} ({task['resource']})\n(W: {task['workers']})")
        ax.text(task["start"] + task["duration"] / 2, i * 10 + 5,
                f"{task['start']:.1f}-{task['finish']:.1f}",
                ha='center', va='center', color='white', fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Tasks")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_convergence(metrics_dict: dict, metric_name: str) -> None:
    """
    Plot boxplots for a given performance metric across different runs.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    data = list(metrics_dict.values())
    ax.boxplot(data, tick_labels=list(metrics_dict.keys()))
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribution of {metric_name} across runs")
    ax.grid(True)
    plt.show()


def plot_pareto_2d(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                   labels: List[str], markers: List[str], colors: List[str],
                   ref_point: Optional[np.ndarray] = None) -> None:
    """
    Plot 2D Pareto fronts (Makespan vs. Total Cost).
    
    If ref_point is provided, it is plotted as a black 'x'.
    """
    plt.figure(figsize=(8, 6))
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
            plt.scatter(objs[:, 0], objs[:, 1], c=color, marker=marker, s=80, edgecolor='k', label=label)
    if ref_point is not None:
        plt.scatter(ref_point[0], ref_point[1], c='black', marker='x', s=100, label='Fixed Reference')
    plt.xlabel("Makespan (hours)")
    plt.ylabel("Total Cost")
    plt.title("2D Pareto Front (Makespan vs. Total Cost)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pareto_3d_individual(archive: List[Tuple[np.ndarray, np.ndarray]],
                              label: str, marker: str, color: str,
                              plane_coeffs: Tuple[float, float, float],
                              x_limits: Tuple[float, float],
                              y_limits: Tuple[float, float],
                              fixed_ref: np.ndarray) -> None:
    """
    Plot 3D Pareto points for a single algorithm along with a fitted plane and fixed reference.
    """
    if not archive:
        print(f"No points to plot for {label}.")
        return
    # Extract objective points for this archive.
    objs = np.array([entry[1] for entry in archive])
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot scatter points for this algorithm.
    ax.scatter(objs[:, 0], objs[:, 1], -objs[:, 2],
               c=color, marker=marker, s=80, edgecolor='k', label=f'{label} n={len(archive)}')
    
    # Generate grid from provided x_limits and y_limits.
    plot_quadratic_surface(plane_coeffs, x_limits, y_limits, ax)
    
    # Plot fixed reference point.
    if fixed_ref is not None:
        ax.scatter([fixed_ref[0]], [fixed_ref[1]], [-fixed_ref[2]],
                   c='black', marker='x', s=100, label='Fixed Reference')
    
    ax.set_xlabel("Makespan (hours)")
    ax.set_ylabel("Total Cost")
    ax.set_zlabel("Average Utilization")
    ax.set_title(f"3D Pareto Front for {label} with Fitted Plane")
    ax.legend()
    plt.show()


def plot_pareto_3d_combined(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                            labels: List[str], markers: List[str], colors: List[str],
                            plane_coeffs: Tuple[float, float, float],
                            x_limits: Tuple[float, float],
                            y_limits: Tuple[float, float],
                            fixed_ref: Optional[np.ndarray] = None) -> None:
    """
    Plot the combined 3D Pareto fronts (from all algorithms) along with a fitted plane and fixed reference.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each algorithm's Pareto points.
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
            ax.scatter(objs[:, 0], objs[:, 1], -objs[:, 2],
                       c=color, marker=marker, s=80, edgecolor='k', label=f'{label} n={len(archive)}')
    
    # Generate grid from provided x_limits and y_limits.
    plot_quadratic_surface(plane_coeffs, x_limits, y_limits, ax)
    
    # Plot fixed reference point.
    if fixed_ref is not None:
        ax.scatter([fixed_ref[0]], [fixed_ref[1]], [-fixed_ref[2]],
                   c='black', marker='x', s=100, label='Fixed Reference')
    
    ax.set_xlabel("Makespan (hours)")
    ax.set_ylabel("Total Cost")
    ax.set_zlabel("Average Utilization")
    ax.set_title("Combined 3D Pareto Front with Fitted Plane")
    ax.legend()
    plt.show()


def plot_all_pareto_graphs(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                           labels: List[str],
                           markers: List[str],
                           colors: List[str],
                           fixed_ref: Optional[np.ndarray] = None) -> None:
    """
    Plot 4 separate graphs:
      1. One combined 3D Pareto front (with fitted plane and fixed reference).
      2. One individual 3D Pareto plot for each algorithm (with the same fitted plane and fixed reference).
    
    The fitted plane is computed once from the global non-dominated Pareto points, and then adjusted
    so that no Pareto point falls below it.
    """
    # Gather all objective vectors from all archives.
    all_points = []
    for archive in archives:
        if archive:
            all_points.extend([entry[1] for entry in archive])
    if len(all_points) == 0:
        print("No points to plot.")
        return
    all_points = np.array(all_points)
    
    # Remove dominated points globally.
    true_points = get_true_pareto_points(all_points)
    
    # Fit a plane to the global non-dominated points.
    plane_coeffs = None
    if true_points.shape[0] >= 3:
        plane_coeffs = fit_quadratic_surface_to_points(true_points)
    else:
        print("Not enough points to fit a plane.")
        plane_coeffs = (0, 0, 0)
    x_limits = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))  # e.g., (260, 310)
    y_limits = (np.min(all_points[:, 1]), np.max(all_points[:, 1]))
    # Adjust the plane so that none of the true points fall below it.
    # For each point, compute error = z_point - (a*x + b*y + c)
    a, b, c, d, e, f = plane_coeffs
    
    # Use the specified percentile (e.g., 95th) of the positive errors.
    delta = 0.01
    # Subtract this delta from f to lower the entire surface.
    f_adjusted = f + delta
    plane_coeffs = (a, b, c, d, e, f_adjusted)
    
    # Plot the combined graph.
    print("Plotting combined Pareto front...")
    plot_pareto_3d_combined(archives, labels, markers, colors, plane_coeffs, x_limits, y_limits, fixed_ref)
    plot_non_dominated_archives_3d(archives, labels, markers, colors, plane_coeffs, x_limits, y_limits, fixed_ref)
    
    # Plot each algorithm individually.
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        print(f"Plotting individual Pareto front for {label}...")
        plot_pareto_3d_individual(archive, label, marker, color, plane_coeffs, x_limits, y_limits, fixed_ref)


def plot_comparative_bar_chart(results: dict, metric: str, algos: List[str]) -> None:
    means = {algo: np.mean(results[algo][metric]) for algo in algos}
    stds = {algo: np.std(results[algo][metric]) for algo in algos}
    fig, ax = plt.subplots(figsize=(8,6))
    x = np.arange(len(algos))
    ax.bar(x, [means[a] for a in algos], yerr=[stds[a] for a in algos], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_title(f"Comparative {metric} (Mean ± Std)")
    ax.set_ylabel(metric)
    plt.show()


def plot_aggregate_convergence(convergence_data: Dict[str, List[List[float]]], title: str = "Aggregate Convergence Curves") -> None:
    """
    Plot the aggregate convergence curves for each algorithm.
    For each algorithm, the mean and standard deviation over iterations are computed and plotted.
    """
    fig, ax = plt.subplots(figsize=(10,6))
    for algo, curves in convergence_data.items():
        curves_arr = np.array(curves)
        mean_curve = np.mean(curves_arr, axis=0)
        std_curve = np.std(curves_arr, axis=0)
        iterations = np.arange(len(mean_curve))
        ax.plot(iterations, mean_curve, label=algo)
        ax.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best Makespan (hours)")
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_non_dominated_archives_3d(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                                   labels: List[str],
                                   markers: List[str],
                                   colors: List[str],
                                   plane_coeffs: Tuple[float, float, float],
                                   x_limits: Tuple[float, float],
                                   y_limits: Tuple[float, float],
                                   fixed_ref: Optional[np.ndarray] = None) -> None:
    """
    Plot a 3D Pareto front graph showing only the global non-dominated solutions (from the union of all archives).
    
    The function does the following:
      1. Combines all archives (each archive is a list of (decision_vector, objective_vector) tuples)
         and tags each solution with its algorithm index.
      2. Removes any solution that is dominated by any other across all archives.
      3. Groups the non-dominated solutions by algorithm.
      4. Creates a 3D scatter plot using all three objectives:
            - x-axis: first objective (e.g., makespan)
            - y-axis: second objective (e.g., total cost)
            - z-axis: third objective (e.g., resource utilization), here plotted as -z 
              if you want lower resource utilization to appear higher.
      5. Overlays a fitted surface (using the provided plane_coeffs) that is generated
         over the given x_limits and y_limits.
      6. Plots a fixed reference point (if provided).
      7. Annotates each algorithm’s contribution (the count of its non-dominated solutions) in the legend.
    
    Parameters:
      - archives: List of archives; each archive is a list of (decision_vector, objective_vector) tuples.
      - labels: Labels for each algorithm.
      - markers: Marker styles for each algorithm (e.g., 'o', '^', 's').
      - colors: Colors for each algorithm.
      - plane_coeffs: Coefficients (a, b, c) for a fitted plane (or surface) of the form z = a*x + b*y + c.
      - x_limits: Tuple (x_min, x_max) defining the x-axis range for the fitted surface.
      - y_limits: Tuple (y_min, y_max) defining the y-axis range for the fitted surface.
      - fixed_ref: Optional fixed reference point as a 3-element array.
    """
    # 1. Combine all solutions with algorithm indices.
    combined = []
    for alg_idx, archive in enumerate(archives):
        for sol in archive:
            # Each sol is (decision_vector, objective_vector)
            combined.append((alg_idx, sol[0], sol[1]))
    
    # 2. Compute global non-dominated solutions.
    global_nd = get_global_non_dominated(combined)
    
    # 3. Group solutions by originating algorithm.
    groups = {i: [] for i in range(len(archives))}
    for alg_idx, decision, obj in global_nd:
        groups[alg_idx].append((decision, obj))
    
    # 4. Create a 3D scatter plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(len(archives)):
        group = groups[i]
        if not group:
            continue
        
        # For each solution, use all three objective values.
        # We assume that the objective_vector is [makespan, total cost, resource utilization].
        # If lower resource utilization is better, and you want to visualize it inverted,
        # we plot - (resource utilization) on the z-axis.
        objs = np.array([entry[1] for entry in group])
        x = objs[:, 0]  # e.g., makespan
        y = objs[:, 1]  # e.g., total cost
        z = -objs[:, 2] # invert resource utilization so that lower becomes higher
        ax.scatter(x, y, z, marker=markers[i], color=colors[i], s=80, 
                   label=f"{labels[i]} (n={len(group)})")
    
    # 5. Plot the fitted surface.
    # Generate a grid over x_limits and y_limits.
    plot_quadratic_surface(plane_coeffs, x_limits, y_limits, ax)
    
    # 6. Plot the fixed reference point if provided.
    if fixed_ref is not None:
        # Assume fixed_ref is a 3-element array [x, y, z]; plot z as negative.
        ax.scatter([fixed_ref[0]], [fixed_ref[1]], [-fixed_ref[2]], c='black', marker='x', s=100, label='Fixed Reference')
    
    # 7. Label axes and add title/legend.
    ax.set_xlabel("Makespan (hours)")
    ax.set_ylabel("Total Cost")
    ax.set_zlabel("Resource Utilization")
    ax.set_title("Global Non-Dominated 3D Pareto Front by Algorithm")
    ax.legend(loc="best")
    plt.grid(True)
    plt.show()
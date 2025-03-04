# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional,Dict

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

def plot_pareto_3d(archives: List[List[Tuple[np.ndarray, np.ndarray]]],
                   labels: List[str], markers: List[str], colors: List[str],
                   ref_point: Optional[np.ndarray] = None) -> None:
    """
    Plot 3D Pareto fronts (Makespan, Total Cost, Average Utilization).
    
    If ref_point is provided, it is plotted as a black 'x'.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for archive, label, marker, color in zip(archives, labels, markers, colors):
        if archive:
            objs = np.array([entry[1] for entry in archive])
            ax.scatter(objs[:, 0], objs[:, 1], -objs[:, 2], c=color, marker=marker, s=80, edgecolor='k', label=label)
    if ref_point is not None:
        ax.scatter([ref_point[0]], [ref_point[1]], [-ref_point[2]], c='black', marker='x', s=100, label='Fixed Reference')
    ax.set_xlabel("Makespan (hours)")
    ax.set_ylabel("Total Cost")
    ax.set_zlabel("Average Utilization")
    ax.set_title("3D Pareto Front")
    ax.legend()
    plt.show()

# Additional comparative visualizations (optional)
def plot_comparative_bar_chart(results: dict, metric: str, algos: List[str]) -> None:
    import matplotlib.pyplot as plt
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
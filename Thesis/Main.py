# Main.py

import numpy as np
import random
import matplotlib.pyplot as plt

from project_schedule import tasks
from multi_objective import compute_schedule, multi_objective
from HHO import MOHHO
from vizualization import plot_gantt

def main():

    # Decision space: one worker allocation per task (dimension = number of tasks)
    dim = len(tasks)
    lb = np.array([task["min"] for task in tasks])
    ub = np.array([task["max"] for task in tasks])
    
    # For an initial baseline schedule we use the midpoint of [min, max] for each task.
    baseline_x = (lb + ub) / 2.0
    baseline_schedule, baseline_makespan = compute_schedule(baseline_x, tasks)
    print("Baseline makespan (hours):", baseline_makespan)
    plot_gantt(baseline_schedule, "Baseline Schedule")
    
    # Set MOHHO parameters
    SearchAgents_no = 30
    Max_iter = 100
    
    # Run the Multi-objective HHO
    pareto_archive, convergence = MOHHO(multi_objective, lb, ub, dim, SearchAgents_no, Max_iter)
    
    # Extract objective values for Pareto analysis:
    pareto_objs = np.array([f for (_, f) in pareto_archive])
    makespans = pareto_objs[:, 0]
    costs = pareto_objs[:, 1]
    avg_utils = -pareto_objs[:, 2]  # Convert back to positive utilization
    
    # Plot the Pareto front: makespan vs. cost, with color indicating average utilization.
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(makespans, costs, c=avg_utils, cmap='viridis', s=80, edgecolor='k')
    plt.xlabel("Project Makespan (hours)")
    plt.ylabel("Total Cost")
    plt.title("Pareto Front: Makespan vs. Cost\n(Color ~ Average Resource Utilization)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Average Utilization (0-1)")
    plt.grid(True)
    plt.show()
    
    # Optionally, pick one Pareto solution to display its detailed schedule.
    print("--------------------------")
    print("Best makespan solution")
    best_index = np.argmin(makespans)
    best_solution = pareto_archive[best_index][0]
    best_schedule, best_makespan = compute_schedule(best_solution, tasks)
    print("\nSelected schedule from Pareto front (lowest makespan):")
    print("Makespan (hours):", best_makespan)
    print("Best solution:", best_solution)
    plot_gantt(best_schedule, "Optimized Schedule (Selected Pareto Solution)")

if __name__ == '__main__':
    main()

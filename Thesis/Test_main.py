# Main.py

import numpy as np
import random
import matplotlib.pyplot as plt

from project_schedule import tasks
from multi_objective import compute_schedule, multi_objective
from egon import MOHHO
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
    SearchAgents_no = 100
    Max_iter = 100
    
    # Run the Multi-objective HHO
    pareto_archive, convergence = MOHHO(multi_objective, lb, ub, dim, SearchAgents_no, Max_iter)
    
    # Extract objective values for Pareto analysis:
    # Each archive entry is a tuple: (solution vector, objective vector)
    # where objective vector = [makespan, cost, negative average utilization]
    pareto_objs = np.array([f for (_, f) in pareto_archive])
    makespans = pareto_objs[:, 0]
    costs = pareto_objs[:, 1]
    # Convert back to positive average utilization:
    avg_utils = -pareto_objs[:, 2]
    
    # --- Print each Pareto solution's objectives ---
    print("\nPareto Archive Solutions:")
    for idx, (sol, obj) in enumerate(pareto_archive):
        print(f"Solution {idx+1}: Makespan = {obj[0]:.2f} hours, "
              f"Cost = {obj[1]:.2f}, Average Utilization = {-obj[2]:.2f}")
    
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
    
    # --- Identify and print the best solutions according to each objective ---
    # Best makespan (minimization)
    best_makespan_index = np.argmin(makespans)
    best_makespan_solution = pareto_archive[best_makespan_index][0]
    best_schedule, best_makespan = compute_schedule(best_makespan_solution, tasks)
    
    print("\n--------------------------")
    print("Best Makespan Solution:")
    print(f"  Makespan: {best_makespan:.2f} hours")
    print(f"  Cost: {costs[best_makespan_index]:.2f}")
    print(f"  Average Utilization: {avg_utils[best_makespan_index]:.2f}")
    print("  Solution vector:", best_makespan_solution)
    plot_gantt(best_schedule, "Optimized Schedule (Best Makespan Solution)")
    
    # Best cost (minimization)
    best_cost_index = np.argmin(costs)
    best_cost_solution = pareto_archive[best_cost_index][0]
    best_schedule_cost, best_makespan_cost = compute_schedule(best_cost_solution, tasks)
    
    print("\n--------------------------")
    print("Best Cost Solution:")
    print(f"  Cost: {costs[best_cost_index]:.2f}")
    print(f"  Makespan: {makespans[best_cost_index]:.2f} hours")
    print(f"  Average Utilization: {avg_utils[best_cost_index]:.2f}")
    print("  Solution vector:", best_cost_solution)
    plot_gantt(best_schedule_cost, "Optimized Schedule (Best Cost Solution)")
    
    # Best resource allocation (maximizing average utilization)
    best_util_index = np.argmax(avg_utils)
    best_util_solution = pareto_archive[best_util_index][0]
    best_schedule_util, best_makespan_util = compute_schedule(best_util_solution, tasks)
    
    print("\n--------------------------")
    print("Best Resource Allocation Solution:")
    print(f"  Average Utilization: {avg_utils[best_util_index]:.2f}")
    print(f"  Makespan: {makespans[best_util_index]:.2f} hours")
    print(f"  Cost: {costs[best_util_index]:.2f}")
    print("  Solution vector:", best_util_solution)
    plot_gantt(best_schedule_util, "Optimized Schedule (Best Resource Allocation)")
    
    # --- Determine the "recommended" solution: the best trade-off between cost and makespan ---
    # Normalize the makespan and cost values so they are comparable.
    # Adding a small number (1e-6) avoids division by zero.
    makespan_norm = (makespans - np.min(makespans)) / (np.max(makespans) - np.min(makespans) + 1e-6)
    cost_norm = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + 1e-6)
    
    # Compute Euclidean distance from the ideal point (0, 0) in the normalized space.
    tradeoff_metric = np.sqrt(makespan_norm**2 + cost_norm**2)
    recommended_index = np.argmin(tradeoff_metric)
    recommended_solution = pareto_archive[recommended_index][0]
    recommended_schedule, recommended_makespan = compute_schedule(recommended_solution, tasks)
    
    print("\n--------------------------")
    print("Recommended (Balanced) Solution:")
    print(f"  Makespan: {makespans[recommended_index]:.2f} hours")
    print(f"  Cost: {costs[recommended_index]:.2f}")
    print(f"  Average Utilization: {avg_utils[recommended_index]:.2f}")
    print("  Solution vector:", recommended_solution)
    print(f"  (Normalized Distance = {tradeoff_metric[recommended_index]:.4f})")
    plot_gantt(recommended_schedule, "Optimized Schedule (Recommended Solution)")
    
if __name__ == '__main__':
    main()

# HHO.py

import numpy as np
import math
import random

def dominates(obj_a, obj_b):
    """
    Determine if solution a dominates solution b.
    """
    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

def Levy(dim):
    """
    Generate a Levy flight step.
    """
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.power(np.abs(v), 1 / beta))
    return step

def MOHHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Multi-objective Harris Hawks Optimization.
    Returns the archive of non-dominated solutions and the convergence history.
    """
    # Initialize hawks randomly within the bounds.
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    archive = []  # List of tuples: (solution vector, objective vector)
    convergence_history = []  # e.g., store archive size at each iteration

    t = 0
    while t < Max_iter:
        # Evaluate population and update archive
        for i in range(SearchAgents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            f_val = objf(X[i, :])
            dominated_flag = False
            removal_list = []
            for (sol_arch, f_arch) in archive:
                if dominates(f_arch, f_val):
                    dominated_flag = True
                    break
                if dominates(f_val, f_arch):
                    removal_list.append((sol_arch, f_arch))
            if not dominated_flag:
                new_archive = []
                for entry in archive:
                    should_remove = False
                    for rem in removal_list:
                        if np.array_equal(entry[0], rem[0]) and np.array_equal(entry[1], rem[1]):
                            should_remove = True
                            break
                    if not should_remove:
                        new_archive.append(entry)
                archive = new_archive
                archive.append((X[i, :].copy(), f_val.copy()))

        # Select a leader ("rabbit") randomly from the archive
        if archive:
            rabbit = random.choice(archive)[0]
        else:
            rabbit = X[0, :].copy()

        E1 = 2 * (1 - (t / Max_iter))  # Decreasing factor

        # Update positions of hawks
        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            if abs(Escaping_Energy) >= 1:  # Exploration phase
                q = random.random()
                rand_index = random.randint(0, SearchAgents_no - 1)
                X_rand = X[rand_index, :].copy()
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * np.abs(X_rand - 2 * random.random() * X[i, :])
                else:
                    X[i, :] = (rabbit - np.mean(X, axis=0)) - random.random() * ((ub - lb) * random.random() + lb)
            else:  # Exploitation phase
                r = random.random()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = rabbit - Escaping_Energy * np.abs(rabbit - X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X[i, :] = (rabbit - X[i, :]) - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :])
                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :])
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - X[i, :]) + \
                             np.multiply(np.random.randn(dim), Levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0))
                    if np.linalg.norm(objf(X1)) < np.linalg.norm(objf(X[i, :])):
                        X[i, :] = X1.copy()
                    else:
                        X2 = rabbit - Escaping_Energy * np.abs(Jump_strength * rabbit - np.mean(X, axis=0)) + \
                             np.multiply(np.random.randn(dim), Levy(dim))
                        if np.linalg.norm(objf(X2)) < np.linalg.norm(objf(X[i, :])):
                            X[i, :] = X2.copy()
        convergence_history.append(len(archive))
        t += 1

    return archive, convergence_history

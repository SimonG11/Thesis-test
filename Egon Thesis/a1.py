from itertools import product

param_grid = {
    "pop": [50, 100],
        "c2": [1.0, 1.05, 1.1],
        "w_max": [0.8, 0.9],
        "w_min": [0.3, 0.4],
        "disturbance_rate_min": [0.1, 0.15],
        "disturbance_rate_max": [0.3, 0.35],
        "jump_interval": [20, 30],
        "max_iter": [100, 200]
}

for combination in product(*param_grid.values()):
    print(combination)
    params = dict(zip(param_grid.keys(), combination))
    print(params)
"""
Calculate the time complexity of the model.
"""
import numpy as np
from base import *
import time

sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
total_times = []
solve_times = []

for size in sizes:
    time_row = []
    s_time_row = []
    for repeat in range(10):
        start = time.time()
        grid = Grid(size)
        indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]
        grid.rerandomize(lambda: np.random.beta(10, 2)) 
        for i in range(int(0.1*size), int(0.9*size)):
            server = Server(i, (i, 0), 1, 1)
            grid.add_server(server)
            user = User(i, (i, size-1), 0.5, 0.5)
            grid.add_user(user)
        grid.setup_LP()
        mid = time.time()
        flow_results, flow_sum, flow_abs_sum = grid.solve_LP()
        end = time.time()
        if grid.lp_problem.status == pulp.LpStatusInfeasible or grid.lp_problem.status == pulp.LpStatusNotSolved:
            time_row.append(-1)
            s_time_row.append(-1)
        else:
            time_row.append(end-start)
            s_time_row.append(end-mid)
    total_times.append(time_row)
    solve_times.append(s_time_row)

np.save("./Data/total_times.npy", total_times)
np.save("./Data/solve_times.npy", solve_times)

"""
Calculate flow sum and its variance as a measure of speed and jitter.
"""
import numpy as np
from base import *

sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
flow_sum = []
flow_abs_sum = []

for size in sizes:
    fs_row = []
    fas_row = []
    for repeat in range(10):
        grid = Grid(size)
        indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]
        grid.rerandomize(lambda: np.random.beta(10, 2)) 
        for i in range(int(0.1*size), int(0.9*size)):
            server = Server(i, (i, 0), 1)
            grid.add_server(server)
            user = User(i, (i, size-1), 0.5)
            grid.add_user(user)
        grid.setup_LP()
        flow_results, fs, fas = grid.solve_LP()
        if grid.lp_problem.status == pulp.LpStatusInfeasible or grid.lp_problem.status == pulp.LpStatusNotSolved:
            fs_row.append(-1)
            fas_row.append(-1)
        else:
            fs_row.append(np.sum(fs))
            fas_row.append(np.sum(fas))
    flow_sum.append(fs_row)
    flow_abs_sum.append(fas_row)

np.save("./Data/flow_sum.npy", flow_sum)
np.save("./Data/flow_abs_sub.npy", flow_abs_sum)

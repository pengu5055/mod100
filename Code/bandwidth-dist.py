"""
Calculate the bandwidth distribution of the network
for different randomizations of the grid.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from base import *
from scipy.stats import norm


# Use Custom Style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))
cm = custom_cmap

# Solve multiple grids
size = 30
a = 5
b = 10
grid = Grid(size)
indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]

# grid.derandomize(0.7)
data = []
for i in range(100):
    grid.rerandomize(lambda: np.random.beta(a, b)) 
    for i in range(int(0.1*size), int(0.9*size)):
        server = Server(i, (i, 0), 1, 1)
        grid.add_server(server)
        user = User(i, (i, size-1), 0.1, 0.1)
        grid.add_user(user)
    grid.setup_LP()
    flow_results, flow_sum, flow_abs_sum = grid.solve_LP()
    if grid.lp_problem.status == pulp.LpStatusInfeasible or grid.lp_problem.status == pulp.LpStatusNotSolved:
        print("Not Optimal")
        continue
    else:
        servers = grid.get_servers()
        server_bandwidths = [flow_abs_sum[(d[1], d[2])]for d in servers]
        data.append(server_bandwidths)

server_bandwidths = np.concatenate(data)
np.save(f"./Data/server_bandwidths-{a}-{b}-lowuser.npy", server_bandwidths)

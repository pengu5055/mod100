"""
Optimize case where top row is occupied by servers and bottom row is occupied by users.
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

# Initiate Grid
size = 30
grid = Grid(size)
indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]

# grid.derandomize(0.7)
grid.rerandomize(lambda: np.random.beta(10, 2)) 
for i in range(int(0.1*size), int(0.9*size)):
    server = Server(i, (i, 0), 1, 1)
    grid.add_server(server)
    user = User(i, (i, size-1), 0.5, 0.5)
    grid.add_user(user)

size2 = 100
grid2 = Grid(size2)
indices = [(i, j) for i in range(grid2.size) for j in range(grid2.size)]

# grid.derandomize(0.7)
grid2.rerandomize(lambda: np.random.beta(10, 2))
for i in range(int(0.1*size2), int(0.9*size2)):
    server = Server(i, (i, 0), 1, 1)
    grid2.add_server(server)
    user = User(i, (i, size2-1), 0.5, 0.5)
    grid2.add_user(user)

# Setup LP Problem
grid.setup_LP()
grid2.setup_LP()
lp_problem = grid.lp_problem

flow_results, flow_sum, flow_abs_sum = grid.solve_LP()
flow_results2, flow_sum2, flow_abs_sum2 = grid2.solve_LP()


fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout="compressed")
ax = [axs[0]]
grid.plot_grid(ax[0])
grid.plot_flow(fig, ax[0], flow_results)
grid.plot_icons(ax[0])
grid.plot_flow_divergence(fig, ax[0])
grid.plot_wire_bandwidths(fig, ax[0])

ax[0].set_yticks(np.arange(0, size, 10))
ax[0].set_xticks(np.arange(0, size, 10))
ax[0].set_title("Beta PDF Sampled $30\\times 30$ Grid")
ax[0].set_aspect('equal')

ax = [axs[1]]
grid2.plot_grid(ax[0])
grid2.plot_flow(fig, ax[0], flow_results2)
grid2.plot_icons(ax[0])
grid2.plot_flow_divergence(fig, ax[0])
grid2.plot_wire_bandwidths(fig, ax[0])

ax[0].set_yticks(np.arange(0, size, 10))
ax[0].set_xticks(np.arange(0, size, 10))
ax[0].set_title("Beta PDF Sampled $100\\times 100$ Grid")
ax[0].set_aspect('equal')


plt.savefig(f"./Images/optimize-30-100.pdf", dpi=500)
plt.show()

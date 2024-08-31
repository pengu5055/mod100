"""
Test new AStar class implementation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from base import *

# Use Custom Style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))
cm = custom_cmap

# Initiate Grid
size = 10
grid = Grid(size)
indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]

# grid.derandomize(0.7)
grid.rerandomize(lambda: np.random.beta(10, 2)) 
# for i in range(int(0.1*size), int(0.9*size)):
#     server = Server(i, (i, 0), 1)
#     grid.add_server(server)
#     user = User(i, (i, size-1), 0.5)
#     grid.add_user(user)
server = Server(0, (0, 0), 1)
grid.add_server(server)
user = User(0, (size-1, size-1), 0.5)
grid.add_user(user)

# Setup LP Problem
grid.setup_LP()
lp_problem = grid.lp_problem

flow_results, flow_sum, flow_abs_sum = grid.solve_LP()


fig, axs = plt.subplots(1, 1, figsize=(12, 6), layout="compressed")
ax = [axs]
grid.plot_grid(ax[0])
grid.plot_flow(fig, ax[0], flow_results)
grid.plot_icons(ax[0])
grid.plot_flow_divergence(fig, ax[0])
grid.plot_wire_bandwidths(fig, ax[0])

astar = AStar(grid)
path = astar.search(server, user)
if path is not None:
    ax[0].plot([p[0] for p in path], [p[1] for p in path], color="red", linewidth=2, linestyle="--", zorder=15)
ax[0].set_yticks(np.arange(0, size, 10))
ax[0].set_xticks(np.arange(0, size, 10))
ax[0].set_aspect('equal')

plt.savefig(f"./Images/astar-test.pdf", dpi=500)
plt.show()

"""
Scratchpad to test implementations.
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

grid = Grid(50)
grid.derandomize(1)
server = Server(0, (0, 0), 1)
grid.add_server(server)
user = User(0, (40, 34), 0.5)
grid.add_user(user)

astar = AStar(grid)
path = astar.search(server, user)

fig, ax = plt.subplots(1, 1, figsize=(12, 6), layout="compressed")
grid.plot_grid(ax)
grid.plot_icons(ax)
if path is not None:
    x_cords = np.array([p.x for p in path]) + 0.5
    y_cords = np.array([p.y for p in path]) + 0.5
    ax.plot(x_cords, y_cords, color="red", linewidth=2, linestyle="--", zorder=15)
ax.set_yticks(np.arange(0, 5, 1))
ax.set_xticks(np.arange(0, 5, 1))
ax.set_aspect('equal')

plt.show()

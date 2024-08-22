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

grid = Grid(10)
server = Server(0, (5, 5), 0.5, 0.5)
user = User(0, (0, 0), 0.5, 0.5)
grid.add_server(server)
grid.add_user(user)

fig, ax = plt.subplots(figsize=(6, 5), layout="compressed")

# Plot Wires
wires = grid.plot_wires()
cmap = mpl.colors.ListedColormap(colors)
img = ax.imshow(wires, cmap=cmap, vmin=0, vmax=1, zorder=5,
                extent=[0, grid.size, 0, grid.size])
cbar = fig.colorbar(img, ax=ax, orientation="horizontal", pad=0.1)
cbar.set_label("Bandwidth")
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

# Plot Servers
servers = grid.plot_servers()
img = plt.imread("x.png")            
for server in servers:
    ax.imshow(img, extent=[server[1], server[1] + 1, server[2] - 1, server[2]], zorder=10)
# Plot Users
users = grid.plot_users()
for user in users:
    ax.text(user[1], user[2], f"U{user[0]}", ha="center", va="center", color="black")

# Plot Grid
if True:
    for x in np.arange(0.5, grid.size - 0.5, 1):
        for y in np.arange(0.5, grid.size - 0.5, 1):
            ax.plot([x, x], [y, y + 1], color="black", alpha=0.1, lw=2, zorder=5)
            ax.plot([x, x + 1], [y, y], color="black", alpha=0.1, lw=2, zorder=5)
            ax.plot([x, x + 1], [y + 1, y + 1], color="black", alpha=0.1, lw=2, zorder=10)
            ax.plot([x + 1, x + 1], [y, y + 1], color="black", alpha=0.1, lw=2, zorder=10)


ax.set_xticks(np.arange(0.5, grid.size, 1))
ax.set_xticklabels(np.arange(0, grid.size, 1))
ax.set_yticks(np.arange(0.5, grid.size, 1))
ax.set_yticklabels(np.flip(np.arange(0, grid.size, 1)))
ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

plt.show()

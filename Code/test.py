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



grid = Grid(10)
server = Server(0, (5, 5), 0.5, 0.5)
user = User(0, (0, 0), 0.5, 0.5)
grid.add_server(server)
grid.add_user(user)

fig, ax = plt.subplots(figsize=(6, 5), layout="compressed")

# Plot Wires
wires = grid.get_wires()
cmap = cm
cm.set_bad(color="black")
img = ax.imshow(wires, cmap=cmap, vmin=0, vmax=1, zorder=5, origin="lower",
                extent=[0, grid.size, 0, grid.size])
cbar = fig.colorbar(img, ax=ax, orientation="horizontal", pad=0.1)
cbar.set_label("Bandwidth")
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

# Plot Servers
servers = grid.get_servers()
s_ico = plt.imread("server_ico.png")            
for server in servers:
    ax.imshow(s_ico, extent=[server[1], server[1] + 1, server[2], server[2] + 1], zorder=10)
# Plot Users
users = grid.get_users()
u_ico = plt.imread("user_ico.png")
for user in users:
    ax.imshow(u_ico, extent=[user[1], user[1] + 1, user[2], user[2] + 1], zorder=10)

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
ax.set_yticklabels(np.arange(0, grid.size, 1))
ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

plt.show()

print(cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85), return_fmt="hex"))
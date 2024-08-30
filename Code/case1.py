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
size = 10
grid = Grid(size)
indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]

# grid.derandomize(0.7)
grid.rerandomize(lambda: np.random.beta(10, 2)) 
for i in range(int(0.1*size), int(0.9*size)):
    server = Server(i, (i, 0), 1, 1)
    grid.add_server(server)
    user = User(i, (i, size-1), 0.5, 0.5)
    grid.add_user(user)

print(np.mean([grid[i][j].bandwidth for i in range(size) for j in range(size)]))


# Setup LP Problem
grid.setup_LP()
lp_problem = grid.lp_problem

flow_results, flow_sum, flow_abs_sum = grid.solve_LP()


fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="compressed")
ax = [ax]
grid.plot_grid(ax[0])
grid.plot_flow(fig, ax[0], flow_results)
grid.plot_icons(ax[0])
grid.plot_flow_divergence(fig, ax[0])
grid.plot_wire_bandwidths(fig, ax[0])

ax[0].set_yticks(np.arange(0, size, 10))
ax[0].set_xticks(np.arange(0, size, 10))
ax[0].set_title("Optimized Grid")
ax[0].set_aspect('equal')
if False:
    servers = grid.get_servers()
    server_bandwidths = [flow_abs_sum[(d[1], d[2])]for d in servers]
    n, bins = np.histogram(server_bandwidths, bins=15)
    colors = cmr.take_cmap_colors("cmr.tropical", len(bins), cmap_range=(0, 0.85))
    ax[1].bar(bins[:-1], n, width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=4)
    # Fit Normal Distribution
    mu, sigma = np.mean(server_bandwidths), np.std(server_bandwidths)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    y = y * np.max(n) / np.max(y)
    ax[1].plot(x, y, color="black", ls="--", linewidth=2, label="Normal Distribution", zorder=5)
    ax[1].legend()
    ax[1].set_title("Server Bandwidth Distribution")
    ax[1].set_xlabel("Bandwidth")
    ax[1].set_ylabel("Frequency")

# plt.savefig(f"./Images/optimize-case-{size}.pdf", dpi=500)
plt.show()

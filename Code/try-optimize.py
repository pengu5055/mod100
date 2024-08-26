"""
Try to optimize a randomly initiated network grid using PuLP.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from base import *
import pulp

# Use Custom Style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))
cm = custom_cmap

# Initiate Grid
grid = Grid(10)
indices = [(i, j) for i in range(10) for j in range(10)]
server = Server(0, (5, 5), 0.5, 0.5)
user = User(0, (0, 0), 0.3, 0.3)
grid.add_server(server)
grid.add_user(user)

lp_problem = pulp.LpProblem("Network_Flow", pulp.LpMaximize)
flow_vars = pulp.LpVariable.dicts("Flow", indices, lowBound=0, upBound=1, cat="Continuous")

for ind in indices:
    i, j = ind 
    flow_vars[(i, j)].setInitialValue(grid[i][j].bandwidth)

# Objective Function
lp_problem += pulp.lpSum([flow_vars[(i, j)] for i in indices for j in indices if isinstance(grid[i, j], User)])

# Constraint 1: Flow must obey the bandwidth constraints
for i in indices:
    for j in indices:
        if isinstance(grid[i, j], Wire):
            lp_problem += flow_vars[(i, j)] <= grid[i, j].wire_bandwidth, f"Bandwidth_W_{i}_{j}"

# Constraint 2: Flow out of servers must be balanced with flow into users
for i in indices:
    for j in indices:
        if isinstance(grid[i, j], Server):
            lp_problem += pulp.lpSum([flow_vars[(i, j)] for j in indices]) == grid[i, j].server_bandwidth_out, f"Server_Out_{i}_{j}"
        elif isinstance(grid[i, j], User):
            lp_problem += pulp.lpSum([flow_vars[(i, j)] for i in indices]) == grid[i, j].user_bandwidth_in, f"User_In_{i}_{j}"

# Solve the LP Problem
lp_problem.solve(pulp.PULP_CBC_CMD(msg=True, warmStart=True))

# Print the status of the solution
print("Status:", pulp.LpStatus[lp_problem.status])

# Get the optimized flow values
flow_values = np.zeros((grid.size, grid.size))
for i in indices:
    for j in indices:
        if isinstance(grid[i, j], Wire):
            flow_values[i, j] = flow_vars[(i, j)].value()
            
# Plot the optimized network
fig, ax = plt.subplots(figsize=(6, 5), layout="compressed")

# Plot Wires
wires = grid.plot_wires()
cmap = cm
cm.set_bad(color="black")
img = ax.imshow(flow_values, cmap=cmap, vmin=0, vmax=1, zorder=5, origin="lower",
                extent=[0, grid.size, 0, grid.size])
cbar = fig.colorbar(img, ax=ax, orientation="horizontal", pad=0.1)
cbar.set_label("Bandwidth")
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

# Plot Servers
servers = grid.plot_servers()
s_ico = plt.imread("server_ico.png")
for server in servers:
    ax.imshow(s_ico, extent=[server[1], server[1] + 1, server[2], server[2] + 1], zorder=10)

# Plot Users
users = grid.plot_users()
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

plt.show()

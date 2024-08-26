"""
Another attempt at optimization using the new Flow base class.
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
indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]
server = Server(0, (6, 6), 0.5, 0.5)
user = User(0, (3, 3), 0.8, 0.8)
grid.add_server(server)
grid.add_user(user)

# Setup Flow Dictionary with all directions
lp_problem = pulp.LpProblem("Network_Flow", pulp.LpMaximize)
four_flow = {
    "N": pulp.LpVariable.dicts("Flow_N", indices, lowBound=-1, upBound=1, cat="Continuous"),
    "E": pulp.LpVariable.dicts("Flow_E", indices, lowBound=-1, upBound=1, cat="Continuous"),
    "S": pulp.LpVariable.dicts("Flow_S", indices, lowBound=-1, upBound=1, cat="Continuous"),
    "W": pulp.LpVariable.dicts("Flow_W", indices, lowBound=-1, upBound=1, cat="Continuous"),
}

# Constraint 1: Flow must obey the bandwidth constraints
for i, j in indices:
        if isinstance(grid.grid[i][j], Wire):
            for direction in four_flow.keys():
                lp_problem += four_flow[direction][(i, j)] <= grid.grid[i][j].bandwidth, f"Bandwidth_{direction}_{i}_{j}"
        
        if isinstance(grid.grid[i][j], Server):
            lp_problem += pulp.lpSum([four_flow[direction][(i, j)] for direction in four_flow.keys()]) <= grid.grid[i][j].bandwidth, f"Server_Out_{i}_{j}"

        if isinstance(grid.grid[i][j], User):
            lp_problem += pulp.lpSum([four_flow[direction][(i, j)] for direction in four_flow.keys()]) <= grid.grid[i][j].bandwidth, f"User_In_{i}_{j}"
        

# Constraint 2: Sum of flow out of servers must be equal to sum of flow into users
users = grid.get_users()
servers = grid.get_servers()
user_in = 0
server_out = 0

for user_data in users:
    _, x, y, _, _ = user_data
    x = int(x)
    y = int(y)
    user = grid.grid[x][y]
    user_in += pulp.lpSum([four_flow[direction][(x, y)] for direction in four_flow.keys()])

for server_data in servers:
    _, x, y, _, _ = server_data
    x = int(x)
    y = int(y)
    server = grid.grid[x][y]
    server_out += pulp.lpSum([four_flow[direction][(x, y)] for direction in four_flow.keys()])

lp_problem += user_in == server_out, "Flow_Balance"

# Constraint 3: Continuity of flow between wires
for i, j in indices:
    neighbors = {dir: (i + fi[0], j + fi[1]) for dir, fi in FLOW_INDEX.items()}
    for dir, neighbor in neighbors.items():
        if neighbor in indices:
            opposite_flow = list(FLOW_INDEX.keys())[list(FLOW_INDEX.values()).index((FLOW_INDEX[dir][0] * -1, FLOW_INDEX[dir][1] * -1))]
            lp_problem += four_flow[dir][(i, j)] == four_flow[opposite_flow][neighbor], f"Flow_Continuity_{dir}_{i}_{j}"

        else:
            # Border Case
            lp_problem += four_flow[dir][(i, j)] == 0, f"Flow_Continuity_{dir}_{i}_{j}"
        
    if isinstance(grid.grid[i][j], Wire):        
        lp_problem += pulp.lpSum([four_flow[dir][(i, j)] for dir in four_flow.keys()]) == 0, f"Flow_Continuity_{i}_{j}"
    elif isinstance(grid.grid[i][j], Server):
        lp_problem += pulp.lpSum([four_flow[dir][(i, j)] for dir in four_flow.keys()]) >= 0, f"Flow_Source_{i}_{j}"
    elif isinstance(grid.grid[i][j], User):
        lp_problem += pulp.lpSum([four_flow[dir][(i, j)] for dir in four_flow.keys()]) <= 0, f"Flow_Sink_{i}_{j}"

# Objective Function
lp_problem += pulp.lpSum([[four_flow[direction][(x, y)] for direction in four_flow.keys()] for x, y in indices if isinstance(grid.grid[x][y], User)])             
            

# Solve the LP Problem
lp_problem.solve(pulp.PULP_CBC_CMD(msg=True, warmStart=True))
print("Status:", pulp.LpStatus[lp_problem.status])

# Get the optimized flow values
flow_values = np.zeros((grid.size, grid.size))
for i, j in indices:
    for direction in four_flow.keys():
        # if isinstance(grid.grid[i][j], Wire):
        flow_values[i, j] += abs(four_flow[direction][(i, j)].value())
        

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="compressed")

ax = axes[0]

img = ax.imshow(flow_values, cmap=cm, vmin=0, vmax=1, zorder=5, origin="lower", aspect="auto",
                extent=[0, grid.size, 0, grid.size])
cbar = fig.colorbar(img, ax=ax, orientation="horizontal", pad=0.1)
cbar.set_label("Bandwidth")
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])


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

ax = axes[1]
wire_bandwidth = grid.get_wires()


plt.show()

            

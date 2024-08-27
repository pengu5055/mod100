"""
Make connections one-way only and try to optimize. Bidirectional connections are too difficult.
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
for ind in indices:
    i, j = ind
    grid[i][j].bandwidth = 1

server = Server(0, (6, 6), 0.7, 0.7)
grid.add_server(server)
server = Server(1, (3, 6), 0.7, 0.7)
grid.add_server(server)
user = User(0, (3, 3), 0.6, 0.6)
grid.add_user(user)

# Setup LP Problem
lp_problem = pulp.LpProblem("Network_Flow", pulp.LpMaximize)
four_flow = {
    "N": pulp.LpVariable.dicts("Flow_N", indices, lowBound=0, upBound=1, cat="Continuous"),
    "E": pulp.LpVariable.dicts("Flow_E", indices, lowBound=0, upBound=1, cat="Continuous"),
    "S": pulp.LpVariable.dicts("Flow_S", indices, lowBound=0, upBound=1, cat="Continuous"),
    "W": pulp.LpVariable.dicts("Flow_W", indices, lowBound=0, upBound=1, cat="Continuous"),
}

# Set Initial Values
for i, j in indices:
    for dir in four_flow.keys():
        four_flow[dir][(i, j)].setInitialValue(0)

# Constraint 1: Flow must obey the bandwidth constraints
for i, j in indices:
    if isinstance(grid[i][j], User):
        lp_problem += pulp.lpSum([four_flow[dir][(i, j)] for dir in four_flow.keys()]) == grid[i][j].bandwidth
    else:
        lp_problem += pulp.lpSum([four_flow[dir][(i, j)] for dir in four_flow.keys()]) <= grid[i][j].bandwidth

# Constraint 2: Continuity of Flow
for i, j in indices:
    neighbors = grid.get_neighbors(i, j)
    for dir, neighbor in neighbors.items():
        if neighbor is not None:
            lp_problem += four_flow[dir][(i, j)] == four_flow[grid.get_reverse_direction(dir)][neighbor], f"Flow_Continuity_{dir}_{i}_{j}"
        else:
            lp_problem += four_flow[dir][(i, j)] == 0, f"Flow_Continuity_{dir}_{i}_{j}"
        
# Define Objective Function
lp_problem += pulp.lpSum([four_flow[dir][(user[1], user[2])] for dir in four_flow.keys()] for user in grid.get_users())

# Solve LP Problem
lp_problem.solve(pulp.GUROBI(msg=True, warmStart=True))
print("Status:", pulp.LpStatus[lp_problem.status])

# Extract Results
flow_results = {dir: np.zeros((grid.size, grid.size)) for dir in four_flow.keys()}
flow_sum = np.zeros((grid.size, grid.size))
for i, j in indices:
    for dir in four_flow.keys():
        flow_results[dir][i][j] = four_flow[dir][(i, j)].value()
        flow_sum[i][j] += flow_results[dir][i][j]

max_flow = np.max([flow_results[dir].max() for dir in four_flow.keys()])
if max_flow == 0:
    print("No flow found.")
    exit()
else:
    print(f"Max Flow: {max_flow}")

fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="compressed")
# for i, j in indices:
#     for dir in four_flow.keys():
#         if flow_results[dir][i][j] > 0:
#             ax.arrow(j, i, grid.get_dir_offset(dir)[1], grid.get_dir_offset(dir)[0], head_width=0.1, head_length=0.1, fc=colors[0], ec=colors[0])
cm2 = cmr.get_sub_cmap("cmr.redshift", 0.15, 0.85)
norm = mpl.colors.Normalize(vmin=0, vmax=max_flow)
sm = plt.cm.ScalarMappable(cmap=cm2, norm=norm)

img = ax.imshow(flow_sum.T, cmap=cm2, norm=norm, zorder=5, origin="lower", aspect="auto",
                extent=[0, grid.size, 0, grid.size], alpha=1)
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Flow")

# Show Server and User Icons
servers = grid.get_servers()
s_ico = plt.imread("server_ico.png")
for server in servers:
    ax.imshow(s_ico, extent=[server[1], server[1] + 1, server[2], server[2] + 1], zorder=10)
users = grid.get_users()
u_ico = plt.imread("user_ico.png")
for user in users:
    ax.imshow(u_ico, extent=[user[1], user[1] + 1, user[2], user[2] + 1], zorder=10)

# Show Grid
for i in range(grid.size):
    ax.axhline(i, color="#767676", lw=2, zorder=7, alpha=0.5)
    ax.axvline(i, color="#767676", lw=2, zorder=7, alpha=0.5)
    
# Find nonzero flow and order points with values of flow and direction
path = []
path_points =  []
for i, j in indices:
    for direction in four_flow.keys():
        if flow_results[direction][i, j] > 0:
            path.append((i, j, flow_results[direction][i, j], direction))
path = sorted(path, key=lambda x: x[2], reverse=True)
path = np.array(path)
print(path)

norm = mpl.colors.Normalize(vmin=0, vmax=max_flow)
sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
cbar2 = fig.colorbar(sm, ax=ax, orientation="horizontal")
cbar2.set_label("Flow")

for i, j, flow, _ in path:
    i, j = int(i) + 0.5, int(j) + 0.5
    flow = float(flow)
    for dir in four_flow.keys():
        if flow_results[dir][int(i - 0.5), int(j - 0.5)] != 0:
            i_next, j_next = i + FLOW_INDEX[dir][0], j + FLOW_INDEX[dir][1]
            ax.plot([i, i_next], [j, j_next], color=cm(norm(flow)), lw=2, zorder=8)

# Compass
if False:
    c_center = np.array([2, 7]) + 0.5
    ax.scatter(*c_center, color="white", s=50, zorder=10)
    c_colors = ["blue", "yellow", "red", "green"]
    for i, dir in enumerate(four_flow.keys()):
        second = c_center + np.array(FLOW_INDEX[dir])
        ax.plot([c_center[0], second[0]], [c_center[1], second[1]], color=c_colors[i], lw=2, zorder=9)
        ax.text(second[0]+FLOW_INDEX[dir][0]*0.33, second[1]+FLOW_INDEX[dir][1]*0.33, dir, color=c_colors[i], fontsize=10, ha="center", va="center", zorder=10)

ax.set_xticks(np.arange(0.5, grid.size, 1))
ax.set_xticklabels(np.arange(0, grid.size, 1))
ax.set_yticks(np.arange(0.5, grid.size, 1))
ax.set_yticklabels(np.arange(0, grid.size, 1))
ax.set_xlim(0, grid.size)
ax.set_ylim(0, grid.size)

plt.show()

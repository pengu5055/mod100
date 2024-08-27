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
server = Server(0, (6, 4), 1, 1)
grid.add_server(server)
# server = Server(1, (3, 6), 0.4, 0.4)
# grid.add_server(server)
user = User(0, (6, 0), 0.5, 0.5)
grid.add_user(user)

for ind in indices:
    i, j = ind
    grid[i][j].bandwidth = 1 

# Setup Flow Dictionary with all directions
lp_problem = pulp.LpProblem("Network_Flow", pulp.LpMaximize)
four_flow_positive = {
    "N": pulp.LpVariable.dicts("Flow_N_pos", indices, lowBound=0, upBound=1, cat="Continuous"),
    "E": pulp.LpVariable.dicts("Flow_E_pos", indices, lowBound=0, upBound=1, cat="Continuous"),
    "S": pulp.LpVariable.dicts("Flow_S_pos", indices, lowBound=0, upBound=1, cat="Continuous"),
    "W": pulp.LpVariable.dicts("Flow_W_pos", indices, lowBound=0, upBound=1, cat="Continuous"),
}
four_flow_negative = {
    "N": pulp.LpVariable.dicts("Flow_N_neg", indices, lowBound=0, upBound=1, cat="Continuous"),
    "E": pulp.LpVariable.dicts("Flow_E_neg", indices, lowBound=0, upBound=1, cat="Continuous"),
    "S": pulp.LpVariable.dicts("Flow_S_neg", indices, lowBound=0, upBound=1, cat="Continuous"),
    "W": pulp.LpVariable.dicts("Flow_W_neg", indices, lowBound=0, upBound=1, cat="Continuous"),
}

four_flow = {
    "N": pulp.LpVariable.dicts("Flow_N", indices, lowBound=-1, upBound=1, cat="Continuous"),
    "E": pulp.LpVariable.dicts("Flow_E", indices, lowBound=-1, upBound=1, cat="Continuous"),
    "S": pulp.LpVariable.dicts("Flow_S", indices, lowBound=-1, upBound=1, cat="Continuous"),
    "W": pulp.LpVariable.dicts("Flow_W", indices, lowBound=-1, upBound=1, cat="Continuous"),
}

# Set Initial Values
for ind in indices:
    i, j = ind
    for direction in four_flow.keys():
        four_flow_positive[direction][(i, j)].setInitialValue(0)
        four_flow_negative[direction][(i, j)].setInitialValue(0)

# Constraint 0: Build flow from positive and negative flow
for i, j in indices:
    for dir in four_flow.keys():
        lp_problem += four_flow[dir][(i, j)] == four_flow_positive[dir][(i, j)] - four_flow_negative[dir][(i, j)], f"Flow_Build_{dir}_{i}_{j}"


# Constraint 1: Flow must obey the bandwidth constraints
for i, j in indices:
        if isinstance(grid.grid[i][j], Wire):
            lp_problem += pulp.lpSum([four_flow_positive[direction][(i, j)] + four_flow_negative[direction][(i, j)] for direction in four_flow.keys()]) <= grid[i][j].bandwidth, f"Bandwidth_{i}_{j}"
            lp_problem += pulp.lpSum([four_flow[dir][(i, j)] for dir in four_flow.keys()]) == 0, f"Prevent_Leak_{i}_{j}"
        
        if isinstance(grid.grid[i][j], Server):
            for dir in four_flow.keys():
                lp_problem += four_flow_negative[dir][(i, j)] == 0, f"Prevent_Server_In_{dir}_{i}_{j}"
            lp_problem += pulp.lpSum([four_flow_positive[direction][(i, j)] for direction in four_flow.keys()]) <= grid[i][j].bandwidth, f"Server_Out_Balance_{i}_{j}"
        
        if isinstance(grid.grid[i][j], User):
            for dir in four_flow.keys():
                lp_problem += four_flow_positive[dir][(i, j)] == 0, f"Prevent_User_Out_{dir}_{i}_{j}"
            lp_problem += pulp.lpSum([four_flow_negative[direction][(i, j)] for direction in four_flow.keys()]) <= grid[i][j].bandwidth, f"User_In_Balance_{i}_{j}"

# Constraint 2: Continuity of flow between wires
for i, j in indices:
    neighbors = {dir: (i + fi[0], j + fi[1]) for dir, fi in FLOW_INDEX.items()}
    for dir, neighbor in neighbors.items():
        if neighbor in indices:
            opposite_flow = list(FLOW_INDEX.keys())[list(FLOW_INDEX.values()).index((FLOW_INDEX[dir][0] * -1, FLOW_INDEX[dir][1] * -1))]
            lp_problem += four_flow[dir][(i, j)] == four_flow[opposite_flow][neighbor], f"Flow_Continuity_{dir}_{i}_{j}"

        else:
            # Border Case
            lp_problem += four_flow[dir][(i, j)] == 0, f"Flow_Continuity_{dir}_{i}_{j}"

# Objective Function
lp_problem += pulp.lpSum([[four_flow_negative[dir][(x, y)] for dir in four_flow.keys()] for x, y in indices if isinstance(grid[x][y], User)])

# Solve the LP Problem
lp_problem.solve(pulp.GUROBI(msg=True, warmStart=True))
print("Status:", pulp.LpStatus[lp_problem.status])

# Get the optimized flow values
flow_pos = {
    'N': [],
    'E': [],
    'S': [],
    'W': []
}
flow_neg = {
    'N': [],
    'E': [],
    'S': [],
    'W': []
}

max_flow = 0
for i, j in indices:
    flow_pos['N'].append((i, j, four_flow_positive['N'][(i, j)].value(), "N"))
    flow_pos['E'].append((i, j, four_flow_positive['E'][(i, j)].value(), "E"))
    flow_pos['S'].append((i, j, four_flow_positive['S'][(i, j)].value(), "S"))
    flow_pos['W'].append((i, j, four_flow_positive['W'][(i, j)].value(), "W"))

    flow_neg['N'].append((i, j, four_flow_negative['N'][(i, j)].value(), "N"))
    flow_neg['E'].append((i, j, four_flow_negative['E'][(i, j)].value(), "E"))
    flow_neg['S'].append((i, j, four_flow_negative['S'][(i, j)].value(), "S"))
    flow_neg['W'].append((i, j, four_flow_negative['W'][(i, j)].value(), "W"))

    max_flow_attempt = max(four_flow_positive[direction][(i, j)].value(), four_flow_negative[direction][(i, j)].value())
    max_flow = max(max_flow, max_flow_attempt)

flow_sum = {
    'N': np.zeros((grid.size, grid.size)),
    'E': np.zeros((grid.size, grid.size)),
    'S': np.zeros((grid.size, grid.size)),
    'W': np.zeros((grid.size, grid.size)),
}
for i, j in indices:
    for direction in four_flow.keys():
        flow_sum[direction][i, j] = four_flow[direction][(i, j)].value()

flow_sum_sum = np.zeros((grid.size, grid.size))
flow_sum_abs = np.zeros((grid.size, grid.size))
for i, j in indices:
    flow_sum_sum[i, j] = sum([four_flow[direction][(i, j)].value() for direction in four_flow.keys()])
    flow_sum_abs[i, j] = np.sum(np.abs([four_flow[direction][(i, j)].value() for direction in four_flow.keys()]))

if max_flow == 0:
    print("No flow found.")
    exit()

# Plot the optimized network
fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="compressed")

ax = axes[0, 0]
z = np.zeros((grid.size, grid.size))
cm2 = cmr.get_sub_cmap("cmr.redshift", 0.15, 0.85)
norm = mpl.colors.Normalize(vmin=-max_flow, vmax=max_flow)
sm = plt.cm.ScalarMappable(cmap=cm2, norm=norm)

img = ax.imshow(flow_sum_sum.T, cmap=cm2, norm=norm, zorder=5, origin="lower", aspect="auto",
                extent=[0, grid.size, 0, grid.size], alpha=1)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Flow Divergence")

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
    for i in range(grid.size):
        ax.axhline(i, color="#767676", lw=2, zorder=7, alpha=0.5)
        ax.axvline(i, color="#767676", lw=2, zorder=7, alpha=0.5)
    
    # Find nonzero flow and order points with values of flow and direction
    path = []
    path_points =  []
    for i, j in indices:
        for direction in four_flow.keys():
            if flow_sum[direction][i, j] > 0:
                path.append((i, j, flow_sum[direction][i, j], direction))
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
            if flow_sum[dir][int(i - 0.5), int(j - 0.5)] != 0:
                i_next, j_next = i + FLOW_INDEX[dir][0], j + FLOW_INDEX[dir][1]
                ax.plot([i, i_next], [j, j_next], color=cm(norm(flow)), lw=2, zorder=8)

    # Compass
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


ax = axes[0, 1]
norm = mpl.colors.Normalize(vmin=flow_sum_abs.min(), vmax=flow_sum_abs.max())
sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
ax.imshow(flow_sum_abs.T, cmap=cm, origin="lower", norm=norm,
          aspect="auto", zorder=5, extent=[0, grid.size, 0, grid.size])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Flow Magnitude")

plt.show()

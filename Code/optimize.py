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
server = Server(0, (6, 6), 1, 1)
user = User(0, (3, 3), 0.5, 0.5)
grid.add_server(server)
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
        

# Constraint 2: Sum of flow out of servers must be equal to sum of flow into users
# users = grid.get_users()
# servers = grid.get_servers()
# user_in = 0
# server_out = 0
# 
# for user_data in users:
#     _, x, y, _, _ = user_data
#     x = int(x)
#     y = int(y)
#     user = grid.grid[x][y]
#     user_in += pulp.lpSum([four_flow_negative[direction][(x, y)] for direction in four_flow.keys()])
# 
# for server_data in servers:
#     _, x, y, _, _ = server_data
#     x = int(x)
#     y = int(y)
#     server = grid.grid[x][y]
#     server_out += pulp.lpSum([four_flow_positive[direction][(x, y)] for direction in four_flow.keys()])
# 
# lp_problem += user_in == server_out, "Flow_Balance"

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

# Objective Function
lp_problem += pulp.lpSum([[four_flow_negative["N"][(x, y)] for x, y in indices if isinstance(grid[x][y], User)] for x, y in indices if isinstance(grid[x][y], User)])
# lp_problem += pulp.lpSum([[four_flow_negative[direction][(x, y)] for direction in four_flow.keys()] for x, y in indices if isinstance(grid[x][y], User)])             
            

# Solve the LP Problem
# lp_problem.solve(pulp.PULP_CBC_CMD(msg=True, warmStart=True))
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

flow_sum_neg = np.zeros((grid.size, grid.size))
for i, j in indices:
    flow_sum_neg[i, j] = sum([four_flow_negative[direction][(i, j)].value() for direction in four_flow.keys()])


# Plot the optimized network
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="compressed")

ax = axes[0, 0]
z = np.zeros((grid.size, grid.size))
norm = mpl.colors.Normalize(vmin=0, vmax=max_flow)
sm = plt.cm.ScalarMappable(cmap=custom_cmap2, norm=norm)
img = ax.imshow(flow_sum_neg.T, cmap=custom_cmap2, norm=norm, zorder=5, origin="lower", aspect="auto",
                extent=[0, grid.size, 0, grid.size], alpha=1)
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.1)
cbar.set_label("Flow")

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
    N_filtered = np.array([x for x in flow_pos['N'] if x[2] > 0])
    E_filtered = np.array([x for x in flow_pos['E'] if x[2] > 0])
    S_filtered = np.array([x for x in flow_pos['S'] if x[2] > 0])
    W_filtered = np.array([x for x in flow_pos['W'] if x[2] > 0])

    path = np.concatenate([N_filtered, E_filtered, S_filtered, W_filtered], axis=0)
    colors = cmr.take_cmap_colors("cmr.tropical", path.shape[0], cmap_range=(0, 0.85))
    for i, j, flow, direction in path:
        i, j = int(i), int(j)
        i_o, j_o = i + FLOW_INDEX[direction][0], j + FLOW_INDEX[direction][1]

        ax.scatter(i + 0.5, j + 0.5, color=colors[int(i-0.5)], zorder=10)
        ax.scatter(i_o + 0.5, j_o + 0.5, color=colors[int(i-0.5)], zorder=10)

        # ax.plot([i, i_o], [j, j_o], color=colors[int(i-0.5)], alpha=1, lw=2, zorder=8)


# ax.set_xticks(np.arange(0.5, grid.size, 1))
# ax.set_xticklabels(np.arange(0, grid.size, 1))
# ax.set_yticks(np.arange(0.5, grid.size, 1))
# ax.set_yticklabels(np.arange(0, grid.size, 1))
ax.set_xlim(0, grid.size)
ax.set_ylim(0, grid.size)

plt.show()

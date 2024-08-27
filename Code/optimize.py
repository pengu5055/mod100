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

flow_sum = {
    'N': np.zeros((grid.size, grid.size)),
    'E': np.zeros((grid.size, grid.size)),
    'S': np.zeros((grid.size, grid.size)),
    'W': np.zeros((grid.size, grid.size)),
}
for i, j in indices:
    for direction in four_flow.keys():
        flow_sum[direction][i, j] = four_flow[direction][(i, j)].value()

# Plot the optimized network
fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="compressed")

ax = axes[0, 0]
z = np.zeros((grid.size, grid.size))
norm = mpl.colors.Normalize(vmin=0, vmax=max_flow)
sm = plt.cm.ScalarMappable(cmap=custom_cmap2, norm=norm)

img = ax.imshow(flow_sum_neg.T, cmap=custom_cmap2, norm=norm, zorder=5, origin="lower", aspect="auto",
                extent=[0, grid.size, 0, grid.size], alpha=1)
cbar = fig.colorbar(sm, ax=ax, pad=0.1)
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
    for i in range(grid.size):
        ax.axhline(i, color="black", lw=1)
        ax.axvline(i, color="black", lw=1)
    
    # Find nonzero flow and order points with values of flow and direction
    path = []
    path_points =  []
    for i, j in indices:
        for direction in four_flow.keys():
            if flow_sum[direction][i, j] > 0:
                path.append((i, j, flow_sum[direction][i, j], direction))
    path = sorted(path, key=lambda x: x[2], reverse=True)
    path = np.array(path)

    for i, j, flow, direction in path:
        i, j = int(i) + 0.5, int(j) + 0.5
        i_o, j_o = i + FLOW_INDEX[direction][0], j + FLOW_INDEX[direction][1]
        if (i, j) not in path_points:
            path_points.append((i, j))
        if (i_o, j_o) not in path_points:
            path_points.append((i_o, j_o))

    users = grid.get_users()
    user_points = []
    for user in users:
        point = (user[1] + 0.5, user[2] + 0.5)
        if point not in path_points:
            user_points.append(point)

    servers = grid.get_servers()
    server_points = []
    for server in servers:
        point = (server[1] + 0.5, server[2] + 0.5)
        if point not in path_points:
            server_points.append(point)


    colors = cmr.take_cmap_colors("cmr.tropical", path.shape[0], cmap_range=(0, 0.85))
    
    # Sort the path points
    # path_points = sorted(path_points, key=lambda x: x[0]**2+x[1]**2)
    path_points.insert(0, user_points[0])

    for pair in zip(path_points[:-1], path_points[1:]):
        p0, p1 = np.array(pair) - 0.5
        delta = p1 - p0
        dir = list(FLOW_INDEX.keys())[list(FLOW_INDEX.values()).index(tuple(delta))]
        dir2 = list(FLOW_INDEX.keys())[list(FLOW_INDEX.values()).index(tuple(-delta))]
        print(dir)
        print(flow_sum[dir][int(p0[0]), int(p0[1])])
        print(flow_sum[dir2][int(p1[0]), int(p1[1])])
        ax.plot(*zip(*pair), color=colors[0], zorder=8)


# ax.set_xticks(np.arange(0.5, grid.size, 1))
# ax.set_xticklabels(np.arange(0, grid.size, 1))
# ax.set_yticks(np.arange(0.5, grid.size, 1))
# ax.set_yticklabels(np.arange(0, grid.size, 1))
ax.set_xlim(0, grid.size)
ax.set_ylim(0, grid.size)

plt.show()

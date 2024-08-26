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
grid = Grid(5)
indices = [(i, j) for i in range(grid.size) for j in range(grid.size)]
server = Server(0, (4, 4), 0.5, 0.5)
user = User(0, (0, 0), 0.3, 0.3)
grid.add_server(server)
grid.add_user(user)

# Setup Flow Dictionary with all directions
lp_problem = pulp.LpProblem("Network_Flow", pulp.LpMaximize)
four_flow = {
    "N": pulp.LpVariable.dicts("Flow_N", indices, lowBound=0, upBound=1, cat="Continuous"),
    "E": pulp.LpVariable.dicts("Flow_E", indices, lowBound=0, upBound=1, cat="Continuous"),
    "S": pulp.LpVariable.dicts("Flow_S", indices, lowBound=0, upBound=1, cat="Continuous"),
    "W": pulp.LpVariable.dicts("Flow_W", indices, lowBound=0, upBound=1, cat="Continuous"),
}

# Objective Function
for ind in indices:
    for direction in four_flow.keys():
        lp_problem += four_flow[direction][ind].setInitialValue(grid.grid[ind[0]][ind[1]].bandwidth)

# Constraint 1: Flow must obey the bandwidth constraints
for i, j in indices:
    for direction in four_flow.keys():
        if isinstance(grid.grid[i][j], Wire):
            lp_problem += four_flow[direction][(i, j)] <= grid.grid[i][j].bandwidth, f"Bandwidth_{direction}_{i}_{j}"

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
    if isinstance(grid.grid[i][j], Wire):
        neighbors = {dir: (i + fi[0], j + fi[1]) for dir, fi in FLOW_INDEX.items()}
        for dir, neighbor in neighbors.items():
            if neighbor in indices:
                opposite_flow = list(FLOW_INDEX.keys())[list(FLOW_INDEX.values()).index((FLOW_INDEX[dir][0] * -1, FLOW_INDEX[dir][1] * -1))]
                lp_problem += four_flow[dir][(i, j)] == four_flow[opposite_flow][neighbor], f"Flow_Continuity_{dir}_{i}_{j}"
            else:
                lp_problem += four_flow[dir][(i, j)] == 0, f"Flow_Continuity_{dir}_{i}_{j}"
                
            

# Solve the LP Problem
lp_problem.solve(pulp.PULP_CBC_CMD(msg=True, warmStart=True))
print("Status:", pulp.LpStatus[lp_problem.status])

# Get the optimized flow values
flow_values = np.zeros((grid.size, grid.size))
for i, j in indices:
    for direction in four_flow.keys():
        if isinstance(grid.grid[i][j], Wire):
            flow_values[i, j] += four_flow[direction][(i, j)].value()

plt.imshow(flow_values, cmap=cm, origin="lower")
plt.colorbar()
plt.show()

            

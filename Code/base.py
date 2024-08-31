"""
Base python file to house all the base classes and functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import pulp

# hex_colors = ["#2F2E2E", "#787878", "#900EA5", "#B60683"]  # , "#E6E6E6"] #, "#FFFFFF"]  "#3CE89E",
hex_colors = ["#89CA53", "#B8A51B", "#FFFFFF", "#B60683", "#900EA5"]
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("ma-pink", hex_colors)
# hex_colors2 = ["#2F2E2E", "#787878", "#0E90A5", "#0683B6", "#E6E6E6"] #, "#FFFFFF"]
hex_colors2 = ["#D1233C", "#000000", "#00E8C8"]
custom_cmap2 = mpl.colors.LinearSegmentedColormap.from_list("ma-blue", hex_colors2)

# Flow Directions to Index Mapping
FLOW_INDEX = {
    "S": (0, -1),
    "E": (1, 0),
    "N": (0, 1),
    "W": (-1, 0),
}

PLOT_FLOW_INDEX = {
    "N": (0, 1),
    "E": (-1, 0),
    "S": (0, -1),
    "W": (1, 0),
}

class Node:
    def __init__(self, id, x, y, bandwidth):
        self.id = id
        self.x = x
        self.y = y
        self.position = (x, y)
        self.bandwidth = bandwidth
        self.g_cost = bandwidth
        self.h_cost = 0
        self.f_cost = self.g_cost + self.h_cost
        self.parent = None
        self.flow = Flow(0, 0, 0, 0)


class Server(Node):
    """
    Server class to represent a server in the system.
    """
    def __init__(self,
                 server_id,
                 server_position, 
                 server_bandwidth,
                 ):
        super(Server, self).__init__(server_id, server_position[0], server_position[1], server_bandwidth)

    def __str__(self):
        return f"[Server: {self.id} // Bandwidth In: {self.bandwidth} // Position: {self.position}]"
    
    def __repr__(self):
        return f"[Server: {self.id} // Bandwidth In: {self.bandwidth} // Position: {self.position}]"
    
    def __eq__(self, other):
        if other is None:
            return False
        return self.position == other.position and self.bandwidth == other.bandwidth
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __gt__(self, other):
        return self.bandwidth > other.bandwidth
    
    
class User(Node):
    """
    User class to represent a user in the system.
    """
    def __init__(self,
                 user_id,
                 user_position,
                 user_bandwidth,
                 ):
        super(User, self).__init__(user_id, user_position[0], user_position[1], user_bandwidth)

    def __str__(self):
        return f"[User: {self.id} // Bandwidth In: {self.bandwidth} // Position: {self.position}]"
    
    def __repr__(self):
        return f"[User: {self.id} // Bandwidth In: {self.bandwidth} // Position: {self.position}]"
    
    def __eq__(self, other):
        if other is None:
            return False
        return self.position == other.position and self.bandwidth == other.bandwidth
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __gt__(self, other):
        return self.bandwidth > other.bandwidth
    
class Wire(Node):
    """
    Wire class to represent a wire in the system.
    """
    def __init__(self,
                 wire_id,
                 wire_position,
                 wire_bandwidth,
                 ):
        super(Wire, self).__init__(wire_id, wire_position[0], wire_position[1], wire_bandwidth)


    def __str__(self):
        return f"[Wire: {self.id} // Bandwidth: {self.bandwidth}]"
    
    def __repr__(self):
        return f"[Wire: {self.id} // Bandwidth: {self.bandwidth}]"
    
    def __eq__(self, other):
        if other is None:
            return False
        return self.position == other.position and self.bandwidth == other.bandwidth
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __gt__(self, other):
        return self.bandwidth > other.bandwidth
    

class Flow:
    def __init__(self,
                 flow_N,
                 flow_E,
                 flow_S,
                 flow_W,
                 ) -> None:
        self.EPS = 1e-6
        self.N = flow_N
        self.E = flow_E
        self.S = flow_S
        self.W = flow_W

    def __str__(self) -> str:
        return f"[Flow: N: {self.N} // E: {self.E} // S: {self.S} // W: {self.W}]"
    
    def __eq__(self, other):
        return self.N == other.N and self.E == other.E and self.S == other.S and self.W == other.W
        
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def sum(self):
        return self.N + self.E + self.S + self.W
    
    def is_valid(self):
        return self.N + self.E + self.S + self.W <= self.EPS
    

class Grid:
    """
    Grid class to represent the grid of network components.
    """
    def __init__(self,
                 size,
                 ):
        self.size = size
        self.grid = np.zeros((size, size), dtype=object)
        self.indices = [(i, j) for i in range(self.size) for j in range(self.size)]
        for x in range(size):
            for y in range(size):
                self.grid[x][y] = Wire(x * size + y, (x, y), np.random.random())
        self.randomizer = np.random.random()
    
    def rerandomize(self, randomizer):
        self.randomizer = randomizer
        for x in range(self.size):
            for y in range(self.size):
                self.grid[x][y].bandwidth = self.randomizer()

    def add_server(self, server):
        self.grid[server.position] = server

    def add_user(self, user):
        self.grid[user.position] = user

    def get_indices(self):
        return self.indices
    
    def get_wires(self):
        output = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                if isinstance(self.grid[x][y], Wire):
                    output[x][y] = self.grid[x][y].bandwidth
                elif isinstance(self.grid[x][y], Server):
                    output[x][y] = np.nan
                elif isinstance(self.grid[x][y], User):
                    output[x][y] = np.nan
                else:
                    raise ValueError(f"Invalid type {type(self.grid[x][y])} in grid.")
                
        return output
    
    def get_max_bandwidth(self):
        output = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                output[x][y] = self.grid[x][y].bandwidth
                
        return output
    
    def get_servers(self):
        list_servers = [x for x in self.grid.flatten() if isinstance(x, Server)]
        output = [[s.id, s.x, s.y, s.bandwidth] for s in list_servers]

        return np.array(output)
    
    def get_users(self):
        list_users = [x for x in self.grid.flatten() if isinstance(x, User)]
        output = [[u.id, u.x, u.y, u.bandwidth] for u in list_users]

        return np.array(output)
    
    def get_neighbors(self, x, y):
        neighbors = {}
        for dir in FLOW_INDEX.keys():
            dx, dy = FLOW_INDEX[dir]
            check = (int(x + dx), int(y + dy))
            if check in self.indices:
                neighbors[dir] = check
            else:
                neighbors[dir] = None
        
        return neighbors
    
    def get_direction(self, x0, y0, x1, y1):
        if x0 == x1 and y0 == y1:
            return None
        
        dx = x1 - x0
        dy = y1 - y0
        for dir, (ddx, ddy) in FLOW_INDEX.items():
            if ddx == dx and ddy == dy:
                return dir
        
        return None
    
    def get_reverse_direction(self, dir):
        return list(FLOW_INDEX.keys())[list(FLOW_INDEX.values()).index((FLOW_INDEX[dir][0] * -1, FLOW_INDEX[dir][1] * -1))]
    
    def __str__(self):
        output = np.zeros((self.size, self.size), dtype=object)
        for x in range(self.size):
            for y in range(self.size):
                if isinstance(self.grid[x][y], Wire):
                    # If wire is on edge print - or | else +
                    if x % (self.size - 1) == 0 or y % (self.size - 1) == 0:
                        if x % (self.size - 1) == 0:
                            output[x][y] = "-"
                        else:
                            output[x][y] = "|"
                    else:
                        output[x][y] = "+"

                elif isinstance(self.grid[x][y], Server):
                    output[x][y] = f"S:{self.grid[x][y].id}"
                elif isinstance(self.grid[x][y], User):
                    output[x][y] = f"U:{self.grid[x][y].id}"
                else:
                    raise ValueError(f"Invalid type {type(self.grid[x][y])} in grid.")
        
        # Fix corners 
        output[0][0] = "┌" if output[0][0] == "-" else output[0][0]
        output[0][self.size - 1] = "┐" if output[0][self.size - 1] == "-" else output[0][self.size - 1]
        output[self.size - 1][0] = "└" if output[self.size - 1][0] == "-" else output[self.size - 1][0]
        output[self.size - 1][self.size - 1] = "┘" if output[self.size - 1][self.size - 1] == "-" else output[self.size - 1][self.size - 1]

        return "\n".join(["".join([str(x + "\t") for x in row]) for row in output])
    
    def __getitem__(self, key):
        return self.grid[key]
    
    def derandomize(self, value):
        for ind in self.indices:
            i, j = ind
            self.grid[i][j].bandwidth = value
    
    def setup_LP(self):
        self.lp_problem = pulp.LpProblem("Network_Flow", pulp.LpMaximize)
        self.four_flow_positive = {
            "N": pulp.LpVariable.dicts("Flow_N_pos", self.indices, lowBound=0, upBound=1, cat="Continuous"),
            "E": pulp.LpVariable.dicts("Flow_E_pos", self.indices, lowBound=0, upBound=1, cat="Continuous"),
            "S": pulp.LpVariable.dicts("Flow_S_pos", self.indices, lowBound=0, upBound=1, cat="Continuous"),
            "W": pulp.LpVariable.dicts("Flow_W_pos", self.indices, lowBound=0, upBound=1, cat="Continuous"),
        }
        self.four_flow_negative = {
            "N": pulp.LpVariable.dicts("Flow_N_neg", self.indices, lowBound=0, upBound=1, cat="Continuous"),
            "E": pulp.LpVariable.dicts("Flow_E_neg", self.indices, lowBound=0, upBound=1, cat="Continuous"),
            "S": pulp.LpVariable.dicts("Flow_S_neg", self.indices, lowBound=0, upBound=1, cat="Continuous"),
            "W": pulp.LpVariable.dicts("Flow_W_neg", self.indices, lowBound=0, upBound=1, cat="Continuous"),
        }
        self.four_flow = {
            "N": pulp.LpVariable.dicts("Flow_N", self.indices, lowBound=-1, upBound=1, cat="Continuous"),
            "E": pulp.LpVariable.dicts("Flow_E", self.indices, lowBound=-1, upBound=1, cat="Continuous"),
            "S": pulp.LpVariable.dicts("Flow_S", self.indices, lowBound=-1, upBound=1, cat="Continuous"),
            "W": pulp.LpVariable.dicts("Flow_W", self.indices, lowBound=-1, upBound=1, cat="Continuous"),
        }

        # Set Initial Values
        for i, j in self.indices:
            for dir in self.four_flow.keys():
                self.four_flow_positive[dir][(i, j)].setInitialValue(0)
                self.four_flow_negative[dir][(i, j)].setInitialValue(0)

        # Constraint 0: Build flow from positive and negative flow
        for i, j in self.indices:
            for dir in self.four_flow.keys():
                self.lp_problem += self.four_flow[dir][(i, j)] == self.four_flow_positive[dir][(i, j)] - self.four_flow_negative[dir][(i, j)], f"Flow_Build_{dir}_{i}_{j}"

        # Constraint 1: Flow must obey the bandwidth constraints
        for i, j in self.indices:
            if isinstance(self.grid[i][j], User):
                self.lp_problem += pulp.lpSum([self.four_flow_positive[dir][(i, j)] for dir in self.four_flow.keys()]) == 0, f"User_Out_{i}_{j}"
                self.lp_problem += pulp.lpSum([self.four_flow_negative[dir][(i, j)] for dir in self.four_flow.keys()]) == self.grid[i][j].bandwidth, f"User_{i}_{j}"
            elif isinstance(self.grid[i][j], Server):
                self.lp_problem += pulp.lpSum([self.four_flow_negative[dir][(i, j)] for dir in self.four_flow.keys()]) == 0, f"Server_In_{i}_{j}"
                self.lp_problem += pulp.lpSum([self.four_flow_positive[dir][(i, j)] for dir in self.four_flow.keys()]) <= self.grid[i][j].bandwidth, f"Server_{i}_{j}"
            elif isinstance(self.grid[i][j], Wire):
                self.lp_problem += pulp.lpSum([self.four_flow_positive[dir][(i, j)] + self.four_flow_negative[dir][(i, j)] for dir in self.four_flow.keys()]) <= 2*self.grid[i][j].bandwidth, f"Wire_{i}_{j}"
                self.lp_problem += pulp.lpSum([self.four_flow_positive[dir][(i, j)] for dir in self.four_flow.keys()]) == pulp.lpSum([self.four_flow_negative[dir][(i, j)] for dir in self.four_flow.keys()]), f"Wire_Leak_Prevention_{i}_{j}"

        # Constraint 2: Continuity of Flow
        for i, j in self.indices:
            neighbors = self.get_neighbors(i, j)
            for dir, neighbor in neighbors.items():
                if neighbor is not None:
                    self.lp_problem += self.four_flow_positive[dir][(i, j)] == self.four_flow_negative[self.get_reverse_direction(dir)][neighbor], f"Flow_Continuity_P_{dir}_{i}_{j}"
                    self.lp_problem += self.four_flow_negative[dir][(i, j)] == self.four_flow_positive[self.get_reverse_direction(dir)][neighbor], f"Flow_Continuity_N_{dir}_{i}_{j}"
                else:
                    self.lp_problem += self.four_flow[dir][(i, j)] == 0, f"Flow_Continuity_{dir}_{i}_{j}"
        
        # Define Objective Function
        self.lp_problem += - pulp.lpSum([self.four_flow[dir][(x, y)] for dir in self.four_flow.keys() for x, y in self.indices if isinstance(self.grid[x][y], Server)]) \
                           - pulp.lpSum([self.four_flow[dir][(x, y)] for dir in self.four_flow.keys() for x, y in self.indices if isinstance(self.grid[x][y], Wire)])         


    def solve_LP(self):
        self.lp_problem.solve()
        print("Status:", pulp.LpStatus[self.lp_problem.status])

        # Extract Results
        self.flow_results = {dir: np.zeros((self.size, self.size)) for dir in self.four_flow.keys()}
        self.flow_sum = np.zeros((self.size, self.size))
        self.flow_abs_sum = np.zeros((self.size, self.size))
        for i, j in self.indices:
            for dir in self.four_flow.keys():
                self.flow_results[dir][i][j] = self.four_flow[dir][(i, j)].value()
                self.flow_sum[i][j] += self.flow_results[dir][i][j]
                self.flow_abs_sum[i][j] += self.four_flow_positive[dir][(i, j)].value() + self.four_flow_negative[dir][(i, j)].value()

        self.max_flow = np.max([self.flow_results[dir].max() for dir in self.four_flow.keys()])
        if self.max_flow == 0:
            print("No flow found.")
        else:
            print("Max Flow:", self.max_flow)

        return self.flow_results, self.flow_sum, self.flow_abs_sum
    
    def plot_icons(self, ax):
        servers = self.get_servers()
        s_ico = plt.imread("server_ico.png")
        for server in servers:
            ax.imshow(s_ico, extent=[server[1], server[1] + 1, server[2], server[2] + 1], zorder=10)
        users = self.get_users()
        u_ico = plt.imread("user_ico.png")
        for user in users:
            ax.imshow(u_ico, extent=[user[1], user[1] + 1, user[2], user[2] + 1], zorder=10)

    def plot_grid(self, ax):
        for i in range(self.size):
            ax.axhline(i, color="#767676", lw=2, zorder=7, alpha=0.5)
            ax.axvline(i, color="#767676", lw=2, zorder=7, alpha=0.5)

        ax.set_xticks(np.arange(0.5, self.size, 1))
        ax.set_xticklabels(np.arange(0, self.size, 1))
        ax.set_yticks(np.arange(0.5, self.size, 1))
        ax.set_yticklabels(np.arange(0, self.size, 1))
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)

    def plot_flow_divergence(self, fig, ax):
        cm2 = cmr.get_sub_cmap(custom_cmap2, 0.1, 0.9)
        norm = mpl.colors.Normalize(vmin=-self.max_flow, vmax=self.max_flow)
        sm = plt.cm.ScalarMappable(cmap=cm2, norm=norm)

        img = ax.imshow(self.flow_sum.T, cmap=cm2, norm=norm, zorder=5, origin="lower", aspect="auto",
                        extent=[0, self.size, 0, self.size], alpha=1)
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label("Flow Divergence")

    def plot_flow(self, fig, ax, data):
        # Find nonzero flow and order points with values of flow and direction
        path = []
        for i, j in self.indices:
            for direction in self.four_flow.keys():
                if data[direction][i, j] != 0:
                    path.append((i, j, data[direction][i, j], direction))
        path = sorted(path, key=lambda x: x[2], reverse=True)
        path = np.array(path)
        # print(path)

        cm = custom_cmap
        norm = mpl.colors.Normalize(vmin=-self.max_flow, vmax=self.max_flow)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        cbar2 = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
        cbar2.set_label("Flow")

        for i, j, flow, _ in path:
            i, j = int(i), int(j)
            flow = float(flow)
            for dir in self.four_flow.keys():
                if data[dir][i, j] == 0:
                    continue
                elif data[dir][i, j] < -self.grid[i, j].flow.EPS:
                    i0, j0 = i + 0.33, j + 0.66
                    i_next, j_next = i0 + FLOW_INDEX[dir][0], j0 + FLOW_INDEX[dir][1]
                    f = data[dir][i, j]
                    ax.plot([i0, i_next], [j0, j_next], color=cm(norm(f)), lw=3, zorder=8)
                elif data[dir][i, j] > self.grid[i, j].flow.EPS:
                    i0, j0 = i + 0.66, j + 0.33
                    i_next, j_next = i0 + FLOW_INDEX[dir][0], j0 + FLOW_INDEX[dir][1]
                    f = data[dir][i, j]
                    ax.plot([i0, i_next], [j0, j_next], color=cm(norm(f)), lw=3, zorder=8)
    
    def plot_compass(self, ax):
        c_center = np.array([2, 7]) + 0.5
        ax.scatter(*c_center, color="white", s=50, zorder=10)
        c_colors = ["blue", "yellow", "red", "green"]
        for i, dir in enumerate(self.four_flow.keys()):
            second = c_center + np.array(FLOW_INDEX[dir])
            ax.plot([c_center[0], second[0]], [c_center[1], second[1]], color=c_colors[i], lw=2, zorder=9)
            ax.text(second[0]+FLOW_INDEX[dir][0]*0.33, second[1]+FLOW_INDEX[dir][1]*0.33, dir, color=c_colors[i], fontsize=10, ha="center", va="center", zorder=10)

    def plot_wire_bandwidths(self, fig, ax):
        wires = self.get_wires()
        cmap = plt.get_cmap("gray_r")
        server_locations = [(d[1], d[2]) for d in self.get_servers()]
        user_locations = [(d[1], d[2]) for d in self.get_users()]
        # Remove data from server and user locations
        for loc in server_locations + user_locations:
            i, j = loc
            i, j = int(i), int(j)
            wires[i][j] = np.nan
        img = ax.imshow(wires.T, cmap=cmap, vmin=0, vmax=1, zorder=6, origin="lower",
                        extent=[0, self.size, 0, self.size])
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Bandwidth")
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])


class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.indices = grid.get_indices()
        self.size = grid.size
        self.open_set = []
        self.closed_set = []
        
    def search(self, start_node, end_node):
        # This is an object array
        self.open_set.append(start_node)

        while self.open_set:
            # Sort the Open Set to get Node with Lowest Cost First
            self.open_set.sort()
            current_node = self.open_set.pop(0)

            # Add Current Node to Closed Set
            self.closed_set.append(current_node)

            if current_node == end_node:
                # Have Reached the End Node
                print("Path Found")
                return self.reconstruct_path(start_node, current_node)
            
            # Get Neighbors
            neighbors = self.get_node_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor in self.closed_set:
                    continue

                neighbor.parent = current_node
                neighbor.g_cost = current_node.g_cost + 1  # Cost to Move to Neighbor
                neighbor.h_cost = self.heuristic(neighbor, end_node)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                # Check if this is a better path
                if neighbor in self.open_set:
                    if neighbor.g_cost > current_node.g_cost:
                        continue
                    
                self.open_set.append(neighbor)

        # No Path Found
        print("No Path Found")
        return None

    def get_node_neighbors(self, node):
        dirs = FLOW_INDEX.values()
        neighbors = []

        for dir in dirs:
            neighbor_pos = (node.x + dir[0], node.y + dir[1])

            # Check if in grid
            if neighbor_pos in self.indices:
                # Check if traversable (so Wire class)
                # if isinstance(self.grid[neighbor_pos], Wire):
                neighbors.append(self.grid[neighbor_pos[0]][neighbor_pos[1]])

        return neighbors
    
    def heuristic(self, node, end_node):
        # Use Manhattan Distance
        d = np.abs(node.x - end_node.x) + np.abs(node.y - end_node.y)
        return d
    
    def reconstruct_path(self, start_node, end_node):
        path = [end_node]
        current = end_node

        print(current)
        print(current.parent)

        while current.parent != start_node:
            path.append(current.parent)
            current = current.parent

        # Add Start Node
        path.append(start_node)

        # Reverse Path
        return path[::-1]
    
    def update_node(self, neighbor, current_node, g_cost, h_cost):
        print(neighbor)
        print(neighbor.parent)
        neighbor.g_cost = g_cost
        neighbor.h_cost = h_cost
        neighbor.f_cost = g_cost + h_cost
        neighbor.parent = current_node
        print(neighbor.parent)
        
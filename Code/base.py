"""
Base python file to house all the base classes and functions.
"""
import numpy as np
import matplotlib as mpl

hex_colors = ["#2F2E2E", "#787878", "#900EA5", "#B60683", "#E6E6E6"] #, "#FFFFFF"]
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("ma-pink", hex_colors)

# Flow Directions to Index Mapping
FLOW_INDEX = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}


class Server:
    """
    Server class to represent a server in the system.
    """
    def __init__(self,
                 server_id,
                 server_position, 
                 server_bandwidth_in,
                 server_bandwidth_out,
                 ):
        self.id = server_id
        self.position = server_position
        self.x = server_position[0]
        self.y = server_position[1]
        self.bandwidth_in = server_bandwidth_in
        self.bandwidth_out = server_bandwidth_out
        self.bandwidth = server_bandwidth_in
        self.flow = Flow(0, 0, 0, 0)

    def __str__(self):
        return f"[Server: {self.id} // Bandwidth In: {self.bandwidth_in} // Bandwidth Out: {self.bandwidth_out}]"
    
    def __repr__(self):
        return f"[Server: {self.id} // Bandwidth In: {self.bandwidth_in} // Bandwidth Out: {self.bandwidth_out}]"
    
    def __eq__(self, other):
        return self.id == other.id and self.bandwidth_in == other.bandwidth_in and self.bandwidth_out == other.bandwidth_out
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    
class User:
    """
    User class to represent a user in the system.
    """
    def __init__(self,
                 user_id,
                 user_position,
                 user_bandwidth_in,
                 user_bandwidth_out,
                 ):
        self.id = user_id
        self.position = user_position
        self.x = user_position[0]
        self.y = user_position[1]
        self.bandwidth_in = user_bandwidth_in
        self.bandwidth_out = user_bandwidth_out
        self.bandwidth = user_bandwidth_in
        self.flow = Flow(0, 0, 0, 0)

    def __str__(self):
        return f"[User: {self.id} // Bandwidth In: {self.bandwidth_in} // Bandwidth Out: {self.bandwidth_out}]"
    
    def __repr__(self):
        return f"[User: {self.id} // Bandwidth In: {self.bandwidth_in} // Bandwidth Out: {self.bandwidth_out}]"
    
    def __eq__(self, other):
        return self.id == other.id and self.bandwidth_in == other.bandwidth_in and self.bandwidth_out == other.bandwidth_out
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
class Wire:
    """
    Wire class to represent a wire in the system.
    """
    def __init__(self,
                 wire_id,
                 wire_position,
                 wire_bandwidth,
                 ):
        self.id = wire_id
        self.x = wire_position[0]
        self.y = wire_position[1]
        self.position = wire_position
        self.bandwidth = wire_bandwidth
        self.flow = Flow(0, 0, 0, 0)

    def __str__(self):
        return f"[Wire: {self.id} // Bandwidth: {self.bandwidth}]"
    
    def __repr__(self):
        return f"[Wire: {self.id} // Bandwidth: {self.bandwidth}]"
    
    def __eq__(self, other):
        return self.id == other.id and self.bandwidth == other.bandwidth
    
    def __ne__(self, other):
        return not self.__eq__(other)
    

class Flow:
    def __init__(self,
                 flow_N,
                 flow_E,
                 flow_S,
                 flow_W,
                 ) -> None:
        self.EPS = 1e-8
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
        self.indices = [[x, y] for x in range(size) for y in range(size)]
        for x in range(size):
            for y in range(size):
                self.grid[x][y] = Wire(x * size + y, (x, y), np.random.random())

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
        output = [[s.id, s.x, s.y, s.bandwidth_in, s.bandwidth_out] for s in list_servers]

        return np.array(output)
    
    def get_users(self):
        list_users = [x for x in self.grid.flatten() if isinstance(x, User)]
        output = [[u.id, u.x, u.y, u.bandwidth_in, u.bandwidth_out] for u in list_users]

        return np.array(output)
    
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
    

    
        

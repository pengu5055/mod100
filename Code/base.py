"""
Base python file to house all the base classes and functions.
"""
import numpy as np
import matplotlib as mpl

hex_colors = ["#2F2E2E", "#787878", "#900EA5", "#B60683", "#E6E6E6"] #, "#FFFFFF"]
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("ma-pink", hex_colors)


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
        self.server_bandwidth_in = server_bandwidth_in
        self.server_bandwidth_out = server_bandwidth_out

    def __str__(self):
        return f"[Server: {self.id} // Bandwidth In: {self.server_bandwidth_in} // Bandwidth Out: {self.server_bandwidth_out}]"
    
    def __repr__(self):
        return f"[Server: {self.id} // Bandwidth In: {self.server_bandwidth_in} // Bandwidth Out: {self.server_bandwidth_out}]"
    
    def __eq__(self, other):
        return self.id == other.server_id and self.server_bandwidth_in == other.server_bandwidth_in and self.server_bandwidth_out == other.server_bandwidth_out
    
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
        self.user_bandwidth_in = user_bandwidth_in
        self.user_bandwidth_out = user_bandwidth_out

    def __str__(self):
        return f"[User: {self.id} // Bandwidth In: {self.user_bandwidth_in} // Bandwidth Out: {self.user_bandwidth_out}]"
    
    def __repr__(self):
        return f"[User: {self.id} // Bandwidth In: {self.user_bandwidth_in} // Bandwidth Out: {self.user_bandwidth_out}]"
    
    def __eq__(self, other):
        return self.id == other.user_id and self.user_bandwidth_in == other.user_bandwidth_in and self.user_bandwidth_out == other.user_bandwidth_out
    
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
        self.wire_id = wire_id
        self.wire_position = wire_position
        self.wire_bandwidth = wire_bandwidth

    def __str__(self):
        return f"[Wire: {self.wire_id} // Bandwidth: {self.wire_bandwidth}]"
    
    def __repr__(self):
        return f"[Wire: {self.wire_id} // Bandwidth: {self.wire_bandwidth}]"
    
    def __eq__(self, other):
        return self.wire_id == other.wire_id and self.wire_bandwidth == other.wire_bandwidth
    
    def __ne__(self, other):
        return not self.__eq__(other)
    

class Grid:
    """
    Grid class to represent the grid of network components.
    """
    def __init__(self,
                 size,
                 ):
        self.size = size
        self.grid = np.zeros((size, size), dtype=object)
        for x in range(size):
            for y in range(size):
                self.grid[x][y] = Wire(x * size + y, (x, y), np.random.random())

    def add_server(self, server):
        self.grid[server.position] = server

    def add_user(self, user):
        self.grid[user.position] = user
    
    def plot_wires(self):
        output = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                if isinstance(self.grid[x][y], Wire):
                    output[x][y] = self.grid[x][y].wire_bandwidth
                elif isinstance(self.grid[x][y], Server):
                    output[x][y] = np.nan
                elif isinstance(self.grid[x][y], User):
                    output[x][y] = np.nan
                else:
                    raise ValueError(f"Invalid type {type(self.grid[x][y])} in grid.")
                
        return output
    
    def plot_servers(self):
        list_servers = [x for x in self.grid.flatten() if isinstance(x, Server)]
        output = [[s.id, s.x, s.y, s.server_bandwidth_in, s.server_bandwidth_out] for s in list_servers]

        return np.array(output)
    
    def plot_users(self):
        list_users = [x for x in self.grid.flatten() if isinstance(x, User)]
        output = [[u.id, u.x, u.y, u.user_bandwidth_in, u.user_bandwidth_out] for u in list_users]

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
                    print(output[x][y])
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
    

        
        

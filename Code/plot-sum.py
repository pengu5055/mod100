"""
Plot results of flow-sum.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from base import *

# Use Custom Style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))
cm = custom_cmap

# Load Data
flow_sum = np.load("./Data/flow_sum.npy")
flow_abs_sum = np.load("./Data/flow_abs_sum.npy")
sizes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Fits
def cubic_fit(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def square_fit(x, a, b, c):
    return a*x**2 + b*x + c

# Calculate mean and standard deviation
mean_flow = []
flow_std = []
mean_abs_flow = []
std_abs_flow = []
for i in range(len(sizes)):
    flow_row = flow_sum[i]
    flow_abs_row = flow_abs_sum[i]
    flow_row = [value for value in flow_row if value > 0]
    flow_abs_row = [value for value in flow_abs_row if value > 0]
    mean_flow.append(np.mean(flow_row))
    flow_std.append(np.std(flow_row))
    mean_abs_flow.append(np.mean(flow_abs_row))
    std_abs_flow.append(np.std(flow_abs_row))
    
# Plot
mean_flow = mean_flow / sizes**2
mean_abs_flow = mean_abs_flow / sizes**2

fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="compressed")

ax.errorbar(sizes, mean_flow, yerr=flow_std, color=colors[0], lw=2.5, label="Mean Flow", marker="o", capsize=5)
ax.errorbar(sizes, mean_abs_flow, yerr=std_abs_flow, color=colors[1], lw=2.5, label="Mean Abs. Flow", marker="o", capsize=5)
ax.set_xlabel("Grid Size")
ax.set_ylabel("Average Flow per Node")
ax.legend(loc="upper left", fontsize=12, ncols=2)
plt.suptitle("Average Flow per Node vs Grid Size")
ax.set_ylim(-0.05, 1.5)
plt.savefig("./Images/flow-sum.pdf", dpi=500)
plt.show()

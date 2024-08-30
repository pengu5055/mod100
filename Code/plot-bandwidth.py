"""
Plot data generated bt bandwidth-dist.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from scipy.optimize import curve_fit

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# Use Custom Style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))

# Load Data
s_bw = np.load("./Data/server_bandwidths-5-10-lowuser.npy")
s_bw2 = np.load("./Data/server_bandwidths-10-0.1.npy")
s_bw3 = np.load("./Data/server_bandwidths-10-2.npy")
s_bw4 = np.load("./Data/server_bandwidths-10-5.npy")

bw_1 = len(s_bw)/2400
bw_2 = len(s_bw2)/2400
bw_3 = len(s_bw3)/2400
bw_4 = len(s_bw4)/2400

fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="compressed")
ax = ax.flatten()

n, bins = np.histogram(s_bw, bins=15, density=False)
colors = cmr.take_cmap_colors("cmr.tropical", len(bins), cmap_range=(0, 0.85))
ax[0].bar(bins[:-1], n, width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=4, label=f"Server Bandwidth\nSamples: {len(s_bw)}")
ax[0].bar(bins[:-1], n/(bw_1), width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=3, alpha=0.2, label=f"Adjusted for 2400\nServer Samples")
ax[0].set_title("Beta(5, 10) Low User Demand of 0.1")
ax[0].set_xlabel("Bandwidth")
ax[0].set_ylabel("Frequency")
ax[0].set_xticks(np.arange(0, 1.1, 0.1))

n, bins = np.histogram(s_bw2, bins=15, density=False)
colors = cmr.take_cmap_colors("cmr.tropical", len(bins), cmap_range=(0, 0.85))
ax[1].bar(bins[:-1], n, width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=4, label=f"Server Bandwidth\nSamples: {len(s_bw2)}")
ax[1].bar(bins[:-1], n/(bw_2), width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=3, alpha=0.2, label=f"Adjusted for 2400\nServer Samples")
ax[1].set_title("Beta(10, 0.1)")
ax[1].set_xlabel("Bandwidth")
ax[1].set_ylabel("Frequency")
ax[1].set_xticks(np.arange(0, 1.1, 0.1))

n, bins = np.histogram(s_bw3, bins=15, density=False)
colors = cmr.take_cmap_colors("cmr.tropical", len(bins), cmap_range=(0, 0.85))
ax[2].bar(bins[:-1], n, width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=4, label=f"Server Bandwidth\nSamples: {len(s_bw3)}")
ax[2].bar(bins[:-1], n/(bw_3), width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=3, alpha=0.2, label=f"Adjusted for 2400\nServer Samples")
ax[2].set_title("Beta(10, 2)")
ax[2].set_xlabel("Bandwidth")
ax[2].set_ylabel("Frequency")
ax[2].set_xticks(np.arange(0, 1.1, 0.1))

n, bins = np.histogram(s_bw4, bins=15, density=False)
colors = cmr.take_cmap_colors("cmr.tropical", len(bins), cmap_range=(0, 0.85))
ax[3].bar(bins[:-1], n, width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=4, label=f"Server Bandwidth\nSamples: {len(s_bw4)}")
ax[3].bar(bins[:-1], n/(bw_4), width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=3, alpha=0.2, label=f"Adjusted for 2400\nServer Samples")
ax[3].set_title("Beta(10, 5)")
ax[3].set_xlabel("Bandwidth")
ax[3].set_ylabel("Frequency")
ax[3].set_xticks(np.arange(0, 1.1, 0.1))

for a in ax:
    a.legend()
plt.suptitle("Server-load Distribution for Different Wire Bandwidth\nDistributions at User Demand of 0.5")
plt.savefig("./Images/bandwidth-dist.pdf", dpi=500)
plt.show()

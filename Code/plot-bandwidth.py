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
server_bandwidths = np.load("./Data/server_bandwidths-5-10-lowuser.npy")


fig, ax = plt.subplots(1, 1, figsize=(12, 8), layout="compressed")

n, bins = np.histogram(server_bandwidths, bins=15)
colors = cmr.take_cmap_colors("cmr.tropical", len(bins), cmap_range=(0, 0.85))
ax.bar(bins[:-1], n, width=np.diff(bins), color=colors, edgecolor="black", linewidth=0.5, zorder=4)
# Fit Normal Distribution
print(np.array(n).max())
bins = bins + (bins[1]-bins[0])/2
n_total = np.sum(n, axis=0)/np.sum(n)
n_total = n_total/np.max(n_total)
print(n_total)

popt, pcov = curve_fit(gaussian, bins[:-1], n_total, p0=[0.7, 0.5])
x = np.linspace(0, 0.2, 100)
y = gaussian(x, *popt)
print(np.max(n))
y = y/np.max(y) * np.mean(n, axis=0).max()
ax.plot(x, y, color="black", ls="--", linewidth=2, label="Normal Distribution", zorder=5)
ax.legend()
ax.set_title("Server Bandwidth Distribution")
ax.set_xlabel("Bandwidth")
plt.show()

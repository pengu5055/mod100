"""
Plot the probability density function of the beta distribution for different values of the shape 
parameters alpha and beta. 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from scipy.stats import beta

# Use Custom Style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))

fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="compressed")

x = np.linspace(0, 1, 100)
for a, b in [(0.5, 0.5), (1, 1), (2, 3), (8, 4), (10, 0.1), (10, 2), (10, 5)]:
    y = beta.pdf(x, a, b)
    ax.plot(x, y, label=f"$\\alpha$={a}, $\\beta$={b}", lw=3, color=colors.pop(0))

ax.legend()
ax.set_title("Beta Distribution")
ax.set_xlabel("x")
ax.set_ylabel("PDF")
plt.savefig("./Images/beta_dist.pdf", dpi=500)
plt.show()


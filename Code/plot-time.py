"""
Plot data from time-complexity.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from base import *
from scipy.optimize import curve_fit

# Use Custom Style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))
cm = custom_cmap

# Load Data
total_times = np.load("./Data/total_times.npy")
solve_times = np.load("./Data/solve_times.npy")
sizes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Fits
def cubic_fit(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def square_fit(x, a, b, c):
    return a*x**2 + b*x + c


fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout="compressed")

# Calculate mean and standard deviation
mean_times = []
std_times = []
mean_solve_times = []
std_solve_times = []
for i in range(len(sizes)):
    time_row = total_times[i]
    s_time_row = solve_times[i]
    time_row = [time for time in time_row if time > 0]
    s_time_row = [time for time in s_time_row if time > 0]
    mean_times.append(np.mean(time_row))
    std_times.append(np.std(time_row))
    mean_solve_times.append(np.mean(s_time_row))
    std_solve_times.append(np.std(s_time_row))


# Fit cubic and square to mean times
cubic_params, cubic_cov = curve_fit(cubic_fit, sizes, mean_times)
square_params, square_cov = curve_fit(square_fit, sizes, mean_times)
cubic_curve = cubic_fit(sizes, *cubic_params)
square_curve = square_fit(sizes, *square_params)
cubic_params_solve, cubic_cov_solve = curve_fit(cubic_fit, sizes, mean_solve_times)
square_params_solve, square_cov_solve = curve_fit(square_fit, sizes, mean_solve_times)
cubic_curve_solve = cubic_fit(sizes, *cubic_params_solve)
square_curve_solve = square_fit(sizes, *square_params_solve)


ax[0].plot(sizes, cubic_curve, color=colors[0], ls="--", alpha=0.5, lw=2.5, label=f'Total Times Cubic Fit\na={cubic_params[0]:.2e}, b={cubic_params[1]:.2e}, c={cubic_params[2]:.2e}, d={cubic_params[3]:.2e}')
ax[0].plot(sizes, square_curve, color=colors[0], ls=":", alpha=0.5, lw=2.5, label=f'Total Times Square Fit\na={square_params[0]:.2e}, b={square_params[1]:.2e}, c={square_params[2]:.2e}')
ax[0].plot(sizes, cubic_curve_solve, color=colors[3], ls="--", alpha=0.5, lw=2.5, label=f'Solve Times Cubic Fit\na={cubic_params_solve[0]:.2e}, b={cubic_params_solve[1]:.2e}, c={cubic_params_solve[2]:.2e}, d={cubic_params_solve[3]:.2e}')
ax[0].plot(sizes, square_curve_solve, color=colors[3], ls=":", alpha=0.5, lw=2.5, label=f'Solve Times Square Fit\na={square_params_solve[0]:.2e}, b={square_params_solve[1]:.2e}, c={square_params_solve[2]:.2e}')

ax[0].errorbar(sizes, mean_times, yerr=std_times, fmt='o', color=colors[0], label='Total Times')
ax[0].errorbar(sizes, mean_solve_times, yerr=std_solve_times, fmt='o', color=colors[3], label='Solve Times')

ax[0].set_xlabel('Size')
ax[0].set_ylabel('Time (s)')
ax[0].set_title('Execution Times')
ax[0].legend(loc="upper left", fontsize=8)

ax[1].plot(sizes, np.abs(cubic_curve - mean_times)/mean_times, color=colors[0], ls="--", alpha=0.8, lw=2.5, label='Total Times Cubic Fit')
ax[1].plot(sizes, np.abs(square_curve - mean_times)/mean_times, color=colors[0], ls=":", alpha=0.8, lw=2.5, label='Total Times Square Fit')
ax[1].plot(sizes, np.abs(cubic_curve_solve - mean_solve_times)/mean_solve_times, color=colors[3], ls="--", alpha=0.8, lw=2.5, label='Solve Times Cubic Fit')
ax[1].plot(sizes, np.abs(square_curve_solve - mean_solve_times)/mean_solve_times, color=colors[3], ls=":", alpha=0.8, lw=2.5, label='Solve Times Square Fit')

ax[1].set_xlabel('Size')
ax[1].set_ylabel('Relative Error')
ax[1].set_title('Relative Error of Fits')
ax[1].legend(loc="upper right")
ax[1].set_yscale('log')

plt.suptitle("Time Complexity of LP Problem")
plt.savefig("./Images/time-complexity.pdf", dpi=500)
plt.show()

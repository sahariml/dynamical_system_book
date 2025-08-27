import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Paramètres de Hénon classiques
a = 1.4
b = 0.3
N_transient = 100
N_iter = 100_000

# Initialisation
x, y = 0.0, 0.0
x_vals = []
y_vals = []

# Phase transitoire
for _ in range(N_transient):
    x, y = 1 - a * x**2 + y, b * x

# Génération de l’attracteur
for _ in range(N_iter):
    x, y = 1 - a * x**2 + y, b * x
    x_vals.append(x)
    y_vals.append(y)

# Plot principal
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x_vals, y_vals, s=0.01, color='black')
#ax.set_title("Attracteur de Hénon (a = 1.4, b = 0.3)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

# -----------------------------
# ZOOM 1 - zone dense
# -----------------------------
axins1 = inset_axes(ax, width="35%", height="35%", loc='upper right')
axins1.scatter(x_vals, y_vals, s=0.01, color='black')

x1, x2 = -0.2, 0.2
y1, y2 = 0.2, 0.3
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.set_xticks([])
axins1.set_yticks([])
#axins1.set_title("Zoom sur zone I")

# Encadrement + liaison zoom
mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="red", lw=1.0)

# -----------------------------
# ZOOM 2 - branche fractale
# -----------------------------
axins2 = inset_axes(ax, width="30%", height="30%", loc='lower left')
axins2.scatter(x_vals, y_vals, s=0.01, color='black')

x3, x4 = 0.6, 0.8
y3, y4 = -0.05, 0.05
axins2.set_xlim(x3, x4)
axins2.set_ylim(y3, y4)
axins2.set_xticks([])
axins2.set_yticks([])
#axins2.set_title("Zoom sur Zone II")

# Encadrement + liaison zoom
mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="blue", lw=1.0)

plt.tight_layout()
plt.show()

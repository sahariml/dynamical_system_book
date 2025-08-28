import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = 2.0 * np.arccos(0.0)

# ================================
# Fonctions complexes
# ================================
def argument(x):
    re = np.real(x)
    im = np.imag(x)
    absx = np.abs(x)
    eps = 1e-18
    if re < 0.0 and abs(im) < eps:
        return pi
    else:
        return 2.0 * np.arctan2(im, re + absx)

def cub(x, k):
    aa = argument(x)
    bb = abs(x)
    r = bb**(1.0/3.0)
    theta = (aa + 2.0 * k * pi) / 3.0
    return r * (np.cos(theta) + 1j * np.sin(theta))

def phie(z, k):
    sqrt_term = np.sqrt(1.0 - z**3)
    arg = -2.0 + z**3 + 2.0 * sqrt_term
    c = cub(arg, k)
    return c + (z**2) / c

def rec(x, k):
    return 0.5 * (phie(x, k) + x)

def aron(x):
    if abs(np.imag(x)) < 1e-8:
        return np.real(x) + 0.0j
    else:
        return x

# ================================
# Projection stéréographique
# ================================
def xster(x):
    return (2.0 * np.real(x)) / (np.abs(x)**2 + 1.0)

def yster(x):
    return (2.0 * np.imag(x)) / (np.abs(x)**2 + 1.0)

def zster(x):
    return -((np.abs(x)**2 - 1.0) / (np.abs(x)**2 + 1.0))

# ================================
# Génération des points
# ================================
kmax = 9  # profondeur
def som(m):
    return (3**(m+1) - 1) // 2

total_points = som(kmax)
A = np.empty(total_points, dtype=complex)

# Niveau 0
A[0] = 0.0 + 0.0j
A[1] = -(0.5) * 2**(2.0/3.0) + 0j
A[2] = 0.25 * 2**(2.0/3.0) - 0.25*np.sqrt(3)*2**(2.0/3.0)*1j
A[3] = 0.25 * 2**(2.0/3.0) + 0.25*np.sqrt(3)*2**(2.0/3.0)*1j

cursor = 4
for k in range(1, kmax):
    prev_level_start = som(k-1) - 3**(k-1)
    prev_level_end = som(k-1)
    for i in range(prev_level_start, prev_level_end):
        for j in range(1, 4):
            A[cursor] = aron(rec(A[i], j))
            cursor += 1

# ================================
# Projection 3D
# ================================
X = [xster(z) for z in A]
Y = [yster(z) for z in A]
Z = [zster(z) for z in A]

# ================================
# Création de la sphère et des géodésiques
# ================================
# Création d'une grille pour la sphère
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 25)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

# Création de géodésiques (grands cercles)
num_geodesics = 12
geodesics = []
for i in range(num_geodesics):
    theta = i * np.pi / num_geodesics
    u_geo = np.linspace(0, 2 * np.pi, 100)
    x_geo = np.cos(u_geo) * np.sin(theta)
    y_geo = np.sin(u_geo) * np.sin(theta)
    z_geo = np.cos(theta) * np.ones_like(u_geo)
    geodesics.append((x_geo, y_geo, z_geo))

# ================================
# Tracé 3D
# ================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracer la sphère (en fil de fer)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)

# Tracer les géodésiques
for geo in geodesics:
    ax.plot(geo[0], geo[1], geo[2], color='gray', alpha=0.1, linewidth=1)

# Tracer les points de la projection stéréographique
ax.scatter(X, Y, Z, s=2, c='blue', alpha=0.7)

# Configuration des axes
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.set_zlabel("$Z$")
ax.set_box_aspect([1, 1, 1])

# Ajuster les limites pour une meilleure visualisation
max_range = 1.2
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

plt.show()
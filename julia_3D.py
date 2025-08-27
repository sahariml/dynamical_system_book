import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour activer 3D

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
# Tracé 3D
# ================================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, s=2, c='b', alpha=0.7)
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.set_zlabel("$Z$")
#ax.set_title(f"Projection stéréographique (kmax={kmax})")
ax.set_box_aspect([1, 1, 1])
plt.show()

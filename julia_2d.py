import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def aron(x):
    return complex(np.real(x), 0.0) if abs(np.imag(x)) < 1e-8 else x

def som(m):
    return sum(3**k for k in range(m+1))

def argument(x):
    pi = 2.0 * np.arccos(0.0)
    if np.real(x) < 0.0 and abs(np.imag(x)) < 1e-18:
        return pi
    else:
        return 2.0 * np.arctan(np.imag(x) / (np.real(x) + abs(x)))

def cub(x, k):
    pi = 2.0 * np.arccos(0.0)
    aa = argument(x)
    bb = abs(x)
    return bb**(1.0/3.0) * complex(np.cos((aa + 2*k*pi)/3.0),
                                   np.sin((aa + 2*k*pi)/3.0))

def phie(z, k):
    return cub(-2.0 + z**3 + 2.0*np.sqrt(1.0 - z**3), k) + \
           (z**2) / cub(-2.0 + z**3 + 2.0*np.sqrt(1.0 - z**3), k)

def psi(z, k):
    return cub(-2.0 + z**3 + 2.0*np.sqrt(1.0 - z**3), k) - \
           (z**2) / cub(-2.0 + z**3 + 2.0*np.sqrt(1.0 - z**3), k)

def xster(x):
    return (2.0*np.real(x)) / (abs(x)**2 + 1.0)

def yster(x):
    return (2.0*np.imag(x)) / (abs(x)**2 + 1.0)

def zster(x):
    return -((abs(x)**2 - 1.0) / (abs(x)**2 + 1.0))

def rec(x, k):
    return 0.5 * (phie(x, k) + x)

# -------------------------------
# Paramètres
# -------------------------------
kmax = 13
z1 = complex(-(0.5) * 2**(2.0/3.0), 0.0)
z2 = complex((0.25) * 2**(2.0/3.0),
             -(0.25) * np.sqrt(3) * 2**(2.0/3.0))
z3 = complex((0.25) * 2**(2.0/3.0),
              (0.25) * np.sqrt(3) * 2**(2.0/3.0))

# -------------------------------
# Construction des points
# -------------------------------
A = np.zeros(som(kmax), dtype=complex)
A[0] = 0.0 + 0.0j
A[1] = z1
A[2] = z2
A[3] = z3

for k in range(1, kmax):
    c = 0
    deb = som(k-1)
    fin = som(k)
    for i in range(deb, fin):
        for j in range(1, 4):
            c += 1
            A[fin + c - 1] = aron(rec(A[i], j))

# -------------------------------
# Sauvegarde
# -------------------------------
np.savetxt("out.dat", np.column_stack((np.real(A), np.imag(A))))

# -------------------------------
# Tracé
# -------------------------------
x_min, x_max = -2, 2   # à adapter
y_min, y_max = -2, 2   # à adapter

plt.figure(figsize=(6, 6))
plt.scatter(np.real(A), np.imag(A), s=0.5, color="blue")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.axis("equal")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

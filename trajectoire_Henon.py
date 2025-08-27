import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'application de Hénon (classiques)
a = 1.4
b = 0.3

# Nombre d'itérations
n_iterations = 1000

# Initialisation des variables
X = np.zeros(n_iterations)
Y = np.zeros(n_iterations)

# Condition initiale
X[0], Y[0] = 0, 0

# Itération de l'application
for n in range(n_iterations - 1):
    X[n + 1] = 1 - a * X[n]**2 + Y[n]
    Y[n + 1] = b * X[n]

# Tracé de l'évolution
#plt.figure(figsize=(12, 6))

# Évolution de X_n
plt.figure(figsize=(16, 9))
plt.plot(X, 'b-', linewidth=0.8)
#plt.title('Évolution de $X_n$')
plt.ylabel('$x_n$')
plt.grid(alpha=0.3)
plt.tight_layout()
# Évolution de Y_n
plt.figure(figsize=(16, 9))
plt.plot(Y, 'r-', linewidth=0.8)
#plt.title('Évolution de $Y_n$')
plt.xlabel('Itération $n$')
plt.ylabel('$y_n$')
plt.grid(alpha=0.3)

plt.tight_layout()

plt.figure(figsize=(16, 9))
plt.plot(X, Y, 'k.', markersize=3)
plt.title('Attracteur de Hénon')
plt.xlabel('$x_n$')
plt.ylabel('$y_n$')

plt.show()
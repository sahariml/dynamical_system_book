import numpy as np
import matplotlib.pyplot as plt

# Paramètres du système
a = 0.8  # Coefficient pour x
b = 0.8  # Coefficient pour y
c = -0.8  # Coefficient pour x
d = 0.5  # Coefficient pour y

# Point fixe
fixed_point = np.array([0, 0])

# Conditions initiales
x0 = 0.00010
y0 = 0.00010
initial_conditions = np.array([x0, y0])

# Nombre d'itérations
num_iterations = 30

# Liste pour stocker les points
trajectory = [initial_conditions]

# Itérations du système récurrent
for _ in range(num_iterations):
    x_n, y_n = trajectory[-1]
    x_next = a * x_n + b * y_n
    y_next = c * x_n + d * y_n
    trajectory.append(np.array([x_next, y_next]))

# Convertir la liste en tableau NumPy
trajectory = np.array(trajectory)

# Tracer la trajectoire
plt.figure(figsize=(8, 8))
#plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=5, label='Trajectoire')
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=5, label='Trajectoire')

# Longueur fixe pour les flèches
arrow_length = 1

# Ajouter des flèches pour chaque point de la trajectoire avec longueur constante
for i in range(len(trajectory) - 1):
    dx = trajectory[i + 1, 0] - trajectory[i, 0]
    dy = trajectory[i + 1, 1] - trajectory[i, 1]
    norm = np.sqrt(dx**2 + dy**2)
    if norm != 0:  # Vérifier pour éviter la division par zéro
        dx_normalized = (dx / norm) * arrow_length
        dy_normalized = (dy / norm) * arrow_length
        plt.quiver(trajectory[i, 0], trajectory[i, 1], 
                   dx_normalized, dy_normalized,
                   angles='xy', scale_units='xy', color='blue')

# Options pour le champ de vecteurs
X, Y = np.meshgrid(np.linspace(-0.0004, 0.0004, 20), np.linspace(-0.0002, 0.0002, 20))
U = a * X + b * Y
V = c * X + d * Y

# Tracer le champ de vecteurs
#plt.quiver(X, Y, U, V, color='lightgray', alpha=0.5)

plt.plot(fixed_point[0], fixed_point[1], 'ro', label='Point fixe')
#plt.legend(fontsize=10)  # Taille de police pour la légende
#plt.plot(fixed_point[0], fixed_point[1], 'ro')
#plt.title("Trajectoire d'un orbite de point fixe stable avec flèches de longueur constante")
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$y$', fontsize=16)
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid()
#plt.axis('equal')
plt.legend()

plt.show()

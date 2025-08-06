import numpy as np
import matplotlib.pyplot as plt

# Paramètres du système
a = 1.0  # Coefficient pour x
b = 0.5  # Coefficient pour y
c = 1.0  # Coefficient pour x
d = 1.5  # Coefficient pour y

# Point fixe
fixed_point = np.array([0, 0])

# Conditions initiales
initial_conditions = [
    np.array([-0.01950, 0.01851]),
    np.array([-0.01856, 0.01921]),
    np.array([0.01841, -0.01928]),
    np.array([0.01981, -0.01920]),
    np.array([-0.1, 0.1]),
    np.array([0.1, -0.1]),
    np.array([0.0001, 0.0001]),
    np.array([-0.0001, -0.0001])
    
]

x_min=-0.02
x_max=0.02
y_min=-0.02
y_max=0.02
# Nombre d'itérations
num_iterations = 8

# Couleurs pour différencier les trajectoires
colors = ['blue', 'green', 'orange', 'purple','red',"red","m","m"]

plt.figure(figsize=(8, 8))

# Itérer sur chaque condition initiale
for idx, initial_condition in enumerate(initial_conditions):
    trajectory = [initial_condition]
    print("x0=",trajectory)

    # Itérations du système récurrent
    for _ in range(num_iterations):
        x_n, y_n = trajectory[-1]
        x_next = a * x_n + b * y_n
        y_next = c * x_n + d * y_n
        trajectory.append(np.array([x_next, y_next]))

    # Convertir la liste en tableau NumPy
    trajectory = np.array(trajectory)

    # Tracer la trajectoire
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='.', markersize=5, 
             label=f'Trajectoire {idx + 1}', color=colors[idx])

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
                       angles='xy', scale_units='xy', color=colors[idx])

# Tracer les droites avec vecteurs directeurs v1 et v2
v1 = np.array([1, 2])
v2 = np.array([1, -1])
x_range = np.linspace(-0.1, 0.1, 100)

# Droite 1 (v1)
plt.plot(x_range, (v1[1] / v1[0]) * x_range, label='Droite v1 (1, 2)', color='cyan', linestyle='--')

# Droite 2 (v2)
plt.plot(x_range, (v2[1] / v2[0]) * x_range, label='Droite v2 (1, -1)', color='magenta', linestyle='--')


# Tracer le point fixe
plt.plot(fixed_point[0], fixed_point[1], 'ro', label='Point fixe')

# Options de graphique
plt.xlabel(r'$x_k$', fontsize=16)
plt.ylabel(r'$y_k$', fontsize=16)
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid()
#plt.axis('equal')
plt.xlim(x_min, x_max)  # Limites des axes x
plt.ylim(y_min, y_max)  # Limites des axes y
#plt.legend("Point fixe")

plt.show()

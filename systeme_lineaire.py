import numpy as np
import matplotlib.pyplot as plt

def plot_dynamics(a, num_iterations=50, x0=1.0, y0=1.0, save=False):
    """Trace la trajectoire et champ de vecteurs pour X[n+1]=AX[n] avec fenêtrage automatique."""
    
    # Matrice A
    A = np.array([[-1, a],
                  [-2, 1]])
    
    # Point fixe
    fixed_point = np.array([0, 0])
    
    # Trajectoire
    trajectory = [np.array([x0, y0])]
    for _ in range(num_iterations):
        x_n, y_n = trajectory[-1]
        x_next, y_next = A @ np.array([x_n, y_n])
        trajectory.append(np.array([x_next, y_next]))
    trajectory = np.array(trajectory)
    
    # Détermination automatique des bornes
    x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
    
    # Marge pour la lisibilité
    margin_x = 0.2 * (x_max - x_min if x_max != x_min else 1)
    margin_y = 0.2 * (y_max - y_min if y_max != y_min else 1)
    
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y
    
    # === Tracé ===
    plt.figure(figsize=(8, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=5, label='Trajectoire')
    
    # Flèches de direction
    for i in range(len(trajectory) - 1):
        plt.quiver(trajectory[i, 0], trajectory[i, 1],
                   trajectory[i + 1, 0] - trajectory[i, 0],
                   trajectory[i + 1, 1] - trajectory[i, 1],
                   angles='xy', scale_units='xy', scale=2.7, color='blue')
    
    # Champ de vecteurs
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 25), np.linspace(y_min, y_max, 25))
    U = -X + a * Y
    V = -2 * X + Y
    plt.quiver(X, Y, U, V, color='lightgray', alpha=0.6)
    
    # Décorations
    plt.plot(fixed_point[0], fixed_point[1], 'ro', label='Point fixe')
    #plt.title(f"Dynamique pour a={a}")
    plt.xlabel('$x_n$')
    plt.ylabel('$y_n$')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    
    # Sauvegarde si demandé
    if save:
        filename = f"dynamique_a_{a}.eps"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✅ Figure sauvegardée : {filename}")
    
    plt.show()

# === Génération pour les valeurs demandées ===
for a in [0.0, 0.5, 1.0, 1.5]:
    plot_dynamics(a, num_iterations=60, save=True)

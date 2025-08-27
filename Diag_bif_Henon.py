import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def henon_map(a, b, x0, y0, n_iter=1000):
    """Génère les trajectoires (x_n, y_n) pour l'application de Hénon."""
    x, y = x0, y0
    xs = []
    ys = []
    for _ in range(n_iter):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        xs.append(x)
        ys.append(y)

    return xs,ys

def plot_bifurcation_diagram(a_range=(0.5, 1.4), b=0.3,
                              x0=0.1, y0=0.1,
                              n_transient=100, n_attract=200,
                              n_points=1000):
    a_vals = np.linspace(a_range[0], a_range[1], n_points)
    x_values, y_values, a_plot = [], [], []

    print("⏳ Calcul des orbites pour chaque valeur de a...")
    for a in tqdm(a_vals):
        xs,ys = henon_map(a, b, x0, y0, n_transient + n_attract)
        # On garde seulement les dernières valeurs (état final)
        x_values.extend(xs[-n_attract:])
        y_values.extend(ys[-n_attract:])
        a_plot.extend([a] * n_attract)

    plt.figure(figsize=(10, 6))
    plt.plot(a_plot, x_values, '.k', alpha=0.5, markersize=0.5)
    #plt.title("Diagramme de bifurcation de l'application de Hénon")
    plt.xlabel("Paramètre a")
    plt.ylabel("$x_n$")
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    plt.figure(figsize=(10, 6))
    plt.plot(a_plot, y_values, '.k', alpha=0.5, markersize=0.5)
    #plt.title("Diagramme de bifurcation de l'application de Hénon")
    plt.xlabel("Paramètre a")
    plt.ylabel("$y_n$")
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    plt.show()

# --- Appel principal
if __name__ == "__main__":
    plot_bifurcation_diagram(a_range=(0.5, 1.4), b=0.3)

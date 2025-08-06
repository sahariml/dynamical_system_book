import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Paramètre angulaire fixe (rotation)
beta = np.pi / 4  # 45 degrés

def plot_radial_dynamics(mu, ax):
    """Trace la dynamique radiale r_{n+1} = f(r_n)"""
    # Fonction de récurrence radiale
    def r_next(r, mu):
        return (1 + mu) * r - r**3
    
    # Créer la fonction et la diagonale
    r = np.linspace(0, 1.2, 200)
    ax.plot(r, r_next(r, mu), 'b-', label='$r_{n+1} = f(r_n)$')
    ax.plot(r, r, 'k--', label='$r_{n+1} = r_n$')
    
    # Points fixes théoriques
    if mu > 0:
        stable_r = np.sqrt(mu)
        ax.plot(stable_r, stable_r, 'go', markersize=8, label='Cycle limite')
    ax.plot(0, 0, 'ro' if mu < 0 else 'yo' if mu == 0 else 'bo', 
            markersize=8, label='Point fixe')
    
    # Itérations cobweb pour une CI
    r0 = 0.2
    r_vals = [r0]
    s_vals = [0]
    
    for i in range(100):
        s = r_next(r0, mu)
        r_vals.extend([r0, r0])
        s_vals.extend([r0, s])
        r0 = s
    
    ax.plot(r_vals, s_vals, 'r-', alpha=0.7, label='Trajectoire')
    ax.set_title("Dynamique Radiale")
    ax.set_xlabel("$r_n$")
    ax.set_ylabel("$r_{n+1}$")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=7)

def systeme(x, y, mu, seuil=1e6):
    """Calcule l'itération suivante avec contrôle de divergence."""
    if abs(x) > seuil or abs(y) > seuil:
        return float('inf'), float('inf')
    
    # Terme linéaire (rotation)
    term_lin_x = x * np.cos(beta) - y * np.sin(beta)
    term_lin_y = x * np.sin(beta) + y * np.cos(beta)
    
    # Terme non-linéaire (amortissement radial)
    r_sq = x**2 + y**2
    term_nonlin_x = r_sq * (x * np.cos(beta) - y * np.sin(beta))
    term_nonlin_y = r_sq * (x * np.sin(beta) + y * np.cos(beta))
    
    # Itération suivante
    x_next = (1 + mu) * term_lin_x - term_nonlin_x
    y_next = (1 + mu) * term_lin_y - term_nonlin_y
    
    return x_next, y_next

def simuler(mu, x0, y0, n_iter=100, seuil=1e6):
    """Simule la trajectoire et stoppe si divergence."""
    x_vals, y_vals = [x0], [y0]
    x, y = x0, y0
    for _ in range(n_iter):
        x, y = systeme(x, y, mu, seuil)
        if abs(x) > seuil or abs(y) > seuil:
            break
        x_vals.append(x)
        y_vals.append(y)
    return np.array(x_vals), np.array(y_vals)

def ajouter_fleches(ax, x, y, n_fleches=3, color='blue'):
    """Ajoute des flèches directionnelles sur la trajectoire"""
    if len(x) < 2:
        return
    
    # Sélectionne des points régulièrement espacés pour les flèches
    indices = np.linspace(0, len(x)-2, num=n_fleches, dtype=int)
    
    for i in indices:
        # Calcule la direction du déplacement
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        
        # Ajoute la flèche
        ax.quiver(x[i], y[i], dx, dy, 
                  angles='xy', scale_units='xy', scale=1,
                  color=color, width=0.005, headwidth=5, headlength=7, alpha=0.8)

def champ_vecteurs(mu, ax, x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, density=15):
    """Ajoute un champ de vecteurs à l'axe donné"""
    # Créer une grille de points
    x = np.linspace(x_min, x_max, density)
    y = np.linspace(y_min, y_max, density)
    X, Y = np.meshgrid(x, y)
    
    # Calculer les composantes du champ vectoriel
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val, y_val = X[i, j], Y[i, j]
            x_next, y_next = systeme(x_val, y_val, mu)
            U[i, j] = x_next - x_val
            V[i, j] = y_next - y_val
    
    # Normaliser pour une meilleure visualisation
    norm = np.sqrt(U**2 + V**2)
    norm[norm == 0] = 1  # Éviter division par zéro
    U_norm = U / norm
    V_norm = V / norm
    
    # Tracer le champ de vecteurs
    ax.quiver(X, Y, U_norm, V_norm, 
              color='gray', alpha=0.6, scale=25, width=0.004,
              pivot='mid', headwidth=3, headlength=4)

# Paramètres de bifurcation
mu_critique = 0.0
mu_values = [-0.1, mu_critique, 0.1]  # Avant, à, et après la bifurcation
titles = [
    "Avant bifurcation ($\mu = -0.1$)",
    "Bifurcation de Neimark-Hopf ($\mu = 0.0$)",
    "Après bifurcation ($\mu = 0.1$)"
]

# Conditions initiales adaptées à chaque μ
conditions_initiales = [
    [(0.2, 0.3), (0.5, -0.4), (-0.6, 0.1)],  # μ = -0.1
    [(0.2, 0.3), (0.5, -0.4), (-0.6, 0.1)],  # μ = 0.0
    [(0.2, 0.3), (0.5, -0.4), (-0.6, 0.1), (0.1, 0.1)]  # μ = 0.1
]

# Couleurs distinctes pour chaque trajectoire
couleurs = ['b', 'm', 'c', 'r', 'g', 'y']

# Générer chaque figure séparément
for i, mu in enumerate(mu_values):
    # Créer une nouvelle figure
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    
    # Ajouter le champ de vecteurs en premier (en arrière-plan)
    champ_vecteurs(mu, ax, x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, density=15)
    
    # Simuler et tracer les trajectoires
    for j, (x0, y0) in enumerate(conditions_initiales[i]):
        x_vals, y_vals = simuler(mu, x0, y0, n_iter=200)
        if len(x_vals) > 1:
            # Tracer la trajectoire
            couleur = couleurs[j % len(couleurs)]
            ax.plot(x_vals, y_vals, 'o-', markersize=3, lw=1.5, 
                    color=couleur, alpha=0.9, label=f'CI: ({x0:.2f}, {y0:.2f})')
            ax.plot(x_vals[0], y_vals[0], 'go', markersize=8)  # Point initial
            
            # Ajouter des flèches directionnelles
            ajouter_fleches(ax, x_vals, y_vals, n_fleches=8, color=couleur)
    
    # Tracer le point fixe (0,0)
    ax.plot(0, 0, 'ro' if mu < 0 else 'yo' if mu == 0 else 'bo', 
            markersize=10, label='Point fixe', zorder=5)
    
    # Tracer le cycle limite théorique pour μ > 0
    if mu > 0:
        radius = np.sqrt(mu)
        circle = plt.Circle((0, 0), radius, color='b', fill=False, 
                            linestyle='--', linewidth=2, alpha=0.8,
                            label='Cycle limite ($r=\sqrt{{\mu}}$)')
        ax.add_artist(circle)
    
    # Ajouter un cercle unité pour référence
    #circle_unit = plt.Circle((0, 0), 1, color='g', fill=False, 
    #                         linestyle=':', linewidth=1, alpha=0.3,
    #                         label='Cercle unité')
    #ax.add_artist(circle_unit)
    
    # Mise en forme
    ax.set_title(titles[i] + f", $\\beta = {beta:.2f}$", fontsize=14)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    #ax.set_aspect('equal')
    
    # Ajouter des informations sur la stabilité
    stability_text = {
        -0.1: "Point fixe stable\nToutes les trajectoires convergent vers (0,0)",
        0.0: "Point fixe perd sa stabilité\nApparition d'un cycle limite",
        0.1: "Point fixe instable\nCycle limite stable ($r = \sqrt{\mu} \\approx 0.32$)"
    }
    ax.text(0.05, 0.95, stability_text[mu], 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
    
    # Ajouter un sous-graphique pour la dynamique radiale
    #ax_radial = fig.add_axes([0.65, 0.65, 0.25, 0.25])
    #plot_radial_dynamics(mu, ax_radial)
    
    # Légende
    handles, labels = ax.get_legend_handles_labels()
    # Supprimer les doublons de légende
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, fontsize=9, loc='upper right')
    
    # Sauvegarder et afficher
    plt.tight_layout()
    #plt.savefig(f"bifurcation_neimark_hopf_complexe_mu_{mu:.1f}.png", dpi=300)
    plt.show()
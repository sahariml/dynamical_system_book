import numpy as np
import matplotlib.pyplot as plt

def systeme(x, y, mu, seuil=1e6):
    """Calcule l'itération suivante avec contrôle de divergence."""
    if abs(x) > seuil or abs(y) > seuil:
        return float('inf'), float('inf')
    x_next = x - x**2 - y + mu
    y_next = x / 2
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

def champ_vecteurs(mu, ax, x_min=-0.7, x_max=0.2, y_min=-0.35, y_max=0.1, density=15):
    """Ajoute un champ de vecteurs à l'axe donné"""
    # Créer une grille de points
    x = np.linspace(x_min, x_max, density)
    y = np.linspace(y_min, y_max, density)
    X, Y = np.meshgrid(x, y)
    
    # Calculer les composantes du champ vectoriel
    U = -X**2 - Y + mu  # Δx = f(x,y,μ) - x = -x² - y + μ
    V = X/2 - Y         # Δy = g(x,y,μ) - y = x/2 - y
    
    # Normaliser pour une meilleure visualisation
    norm = np.sqrt(U**2 + V**2)
    norm[norm == 0] = 1  # Éviter division par zéro
    U_norm = U / norm
    V_norm = V / norm
    
    # Tracer le champ de vecteurs
    ax.quiver(X, Y, U_norm, V_norm, 
              color='gray', alpha=0.6, scale=20, width=0.004,
              pivot='mid', headwidth=3, headlength=4)
    
    # Ajouter les nullclines
    # Nullcline x: Δx = 0 ⇒ -x² - y + μ = 0 ⇒ y = -x² + μ
    #x_null = np.linspace(x_min, x_max, 100)
    #y_null_x = -x_null**2 + mu
    #ax.plot(x_null, y_null_x, 'r--', alpha=0.5, label='Nullcline x')
    
    # Nullcline y: Δy = 0 ⇒ x/2 - y = 0 ⇒ y = x/2
    #y_null_y = x_null / 2
    #ax.plot(x_null, y_null_y, 'b--', alpha=0.5, label='Nullcline y')

# Paramètres de bifurcation
mu_critique = -1/16  # ≈ -0.0625
mu_values = [-0.1, mu_critique, -0.04]  # Avant, à, et après la bifurcation
titles = [
    "Avant bifurcation ($\mu = -0.1$)",
    "Bifurcation nœud-col ($\mu = -1/16$)",
    "Après bifurcation ($\mu = -0.04$)"
]

# Conditions initiales adaptées à chaque μ
conditions_initiales = [
    [(-0.2, -0.1), (-0.15, -0.075)],  # μ = -0.1 (pas de point fixe)
    [(-0.24, -0.12), (-0.26, -0.13)], # μ = μ_critique (point fixe semi-stable)
    [(-0.3, -0.15), (-0.1, -0.05), (0.0, 0.0)]  # μ = -0.04 (nœud stable et col)
]

# Couleurs distinctes pour chaque trajectoire
couleurs = ['b', 'm', 'c', 'r', 'g']

# Générer chaque figure séparément
for i, mu in enumerate(mu_values):
    # Créer une nouvelle figure pour chaque μ
    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    
    # Ajouter le champ de vecteurs en premier
    champ_vecteurs(mu, ax, x_min=-0.6, x_max=0.1, y_min=-0.3, y_max=0.05, density=20)
    
    for j, (x0, y0) in enumerate(conditions_initiales[i]):
        x_vals, y_vals = simuler(mu, x0, y0)
        if len(x_vals) > 1:
            # Tracer la trajectoire
            couleur = couleurs[j % len(couleurs)]
            ax.plot(x_vals, y_vals, 'o-', markersize=3, lw=1.5, color=couleur, alpha=0.9, label=f'CI: ({x0}, {y0})')
            ax.plot(x_vals[0], y_vals[0], 'go', markersize=8)  # Point initial
            
            # Ajouter des flèches directionnelles
            ajouter_fleches(ax, x_vals, y_vals, n_fleches=4, color=couleur)
    
    # Points fixes théoriques
    if mu == mu_critique:
        ax.plot(-0.25, -0.125, 'ro', markersize=10, label='Point fixe critique', zorder=5)
    elif mu == -0.04:
        # Calculer les points fixes pour μ = -0.04
        # x* = [-1 ± √(1 + 4μ)] / 2, y* = x*/2
        discriminant = 1 + 4*mu
        x1 = (-1 + np.sqrt(discriminant)) / 2
        x2 = (-1 - np.sqrt(discriminant)) / 2
        ax.plot(x1, x1/2, 'bo', markersize=10, label='Nœud stable', zorder=5)
        ax.plot(x2, x2/2, 'ro', markersize=10, label='Col (instable)', zorder=5)
    
    # Mise en forme
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.6, 0.1)
    ax.set_ylim(-0.3, 0.05)
    
    # Ajouter des informations sur la dynamique
    if mu < mu_critique:
        stability_text = "Aucun point fixe\nToutes les trajectoires divergent"
    elif mu == mu_critique:
        stability_text = "Point fixe semi-stable\nà $(-0.25, -0.125)$"
    else:
        stability_text = "Deux points fixes :\n- Nœud stable\n- Col instable"
    
    ax.text(0.05, 0.95, stability_text, 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
    
    # Légende
    handles, labels = ax.get_legend_handles_labels()
    # Supprimer les doublons de légende
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, fontsize=9, loc='lower right')
    
    # Sauvegarder et afficher séparément
    #plt.tight_layout()
    #plt.savefig(f"bifurcation_noeud_col_champ_mu_{mu:.3f}.png", dpi=300)
    plt.show()
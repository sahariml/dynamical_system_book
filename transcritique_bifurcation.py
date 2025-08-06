import numpy as np
import matplotlib.pyplot as plt

def systeme(x, y, mu, seuil=1e6):
    """Calcule l'itération suivante avec contrôle de divergence."""
    if abs(x) > seuil or abs(y) > seuil:
        return float('inf'), float('inf')
    x_next = mu * x - x**2
    y_next = 0.5 * y
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

def champ_vecteurs(mu, ax, x_min=-1.5, x_max=1.5, y_min=-1.0, y_max=1.0, density=15):
    """Ajoute un champ de vecteurs à l'axe donné"""
    # Créer une grille de points
    x = np.linspace(x_min, x_max, density)
    y = np.linspace(y_min, y_max, density)
    X, Y = np.meshgrid(x, y)
    
    # Calculer les composantes du champ vectoriel
    U = (mu - 1)*X - X**2  # Δx = (μ-1)x - x²
    V = -0.5*Y             # Δy = -0.5y
    
    # Normaliser pour une meilleure visualisation
    norm = np.sqrt(U**2 + V**2)
    norm[norm == 0] = 1  # Éviter division par zéro
    U_norm = U / norm
    V_norm = V / norm
    
    # Tracer le champ de vecteurs
    ax.quiver(X, Y, U_norm, V_norm, 
              color='gray', alpha=0.6, scale=25, width=0.004,
              pivot='mid', headwidth=3, headlength=4)
    
    # Ajouter les lignes de flux principales (nullclines)
    # Nullcline x: Δx = 0 ⇒ (μ-1)x - x² = 0 ⇒ x[(μ-1) - x] = 0
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.4)
    if mu != 1:
        ax.axvline(x=mu-1, color='r', linestyle='--', alpha=0.4)
    
    # Nullcline y: Δy = 0 ⇒ -0.5y = 0 ⇒ y = 0
    ax.axhline(y=0, color='b', linestyle='--', alpha=0.4)

# Paramètres de bifurcation
mu_values = [0.5, 1.0, 1.5]  # Avant, à, et après la bifurcation
titles = [
    "Avant bifurcation ($\mu = 0.5$)",
    "Bifurcation transcritique ($\mu = 1.0$)",
    "Après bifurcation ($\mu = 1.5$)"
]

# Conditions initiales adaptées à chaque μ
conditions_initiales = [
    [(-0.2, 0.2), (0.1, 0.1), (0.4, -0.3), (1.0, 0.5)],  # μ = 0.5
    [(-0.2, 0.2), (0.1, 0.1), (0.5, -0.3), (1.2, 0.5)],  # μ = 1.0
    [(-0.2, 0.2), (0.1, 0.1), (0.5, -0.3), (1.0, 0.5)]   # μ = 1.5
]

# Couleurs distinctes pour chaque trajectoire
couleurs = ['b', 'm', 'c', 'r', 'g', 'y']

# Générer chaque figure séparément
for i, mu in enumerate(mu_values):
    # Créer une nouvelle figure
    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    
    # Ajouter le champ de vecteurs en premier (en arrière-plan)
    champ_vecteurs(mu, ax, x_min=-1.2, x_max=1.5, y_min=-1.0, y_max=1.0, density=20)
    
    # Calcul des points fixes
    point_fixe1 = (0, 0)
    point_fixe2 = (mu - 1, 0) if mu > 0 else (0, 0)
    
    # Simuler et tracer les trajectoires
    for j, (x0, y0) in enumerate(conditions_initiales[i]):
        x_vals, y_vals = simuler(mu, x0, y0, n_iter=50)
        if len(x_vals) > 1:
            # Tracer la trajectoire
            couleur = couleurs[j % len(couleurs)]
            ax.plot(x_vals, y_vals, 'o-', markersize=3, lw=1.5, 
                    color=couleur, alpha=0.9, label=f'CI: ({x0}, {y0})')
            ax.plot(x_vals[0], y_vals[0], 'go', markersize=8)  # Point initial
            
            # Ajouter des flèches directionnelles
            ajouter_fleches(ax, x_vals, y_vals, n_fleches=5, color=couleur)
    
    # Tracer les points fixes
    ax.plot(point_fixe1[0], point_fixe1[1], 'ro' if mu < 1 else 'bo', 
            markersize=10, label='Point fixe (0,0)', zorder=5)
    
    if mu != 1:  # À μ=1, les deux points fixes coïncident
        ax.plot(point_fixe2[0], point_fixe2[1], 'bo' if mu < 1 else 'ro', 
                markersize=10, label=f'Point fixe ({mu-1:.1f},0)', zorder=5)
    
    # Mise en forme
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.5)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(fontsize=9, loc='upper right')
    
    # Ajouter des informations sur la stabilité
    stability_text = {
        0.5: "Point fixe (0,0): stable\nPoint fixe (-0.5,0): instable",
        1.0: "Point fixe unique (0,0)\nsemi-stable",
        1.5: "Point fixe (0,0): instable\nPoint fixe (0.5,0): stable"
    }
    ax.text(0.05, 0.95, stability_text[mu], 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
    
    # Ajouter la dynamique de y
    ax.text(0.05, 0.05, "$y_{n+1} = \\frac{1}{2}y_n$ → $y_n \\to 0$", 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', alpha=0.2))
    
    # Sauvegarder et afficher
    plt.tight_layout()
    plt.savefig(f"bifurcation_transcritique_champ_mu_{mu:.1f}.png", dpi=300)
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Paramètres de la simulation
b = 0.3  # Paramètre b fixé
a_min = 0.1
a_max = 1.4
a_steps = 500  # Nombre de valeurs de a
n_transient = 5000  # Itérations transitoires
n_lyap = 10000  # Itérations pour calcul Lyapunov
lyap_threshold = 100  # Seuil pour les valeurs divergentes

# Initialisation
a_values = np.linspace(a_min, a_max, a_steps)
lyap_exponents = np.zeros(a_steps)

# Fonction pour calculer le Jacobien
def jacobian(x, a, b):
    return np.array([[-2*a*x, 1], 
                    [b, 0]])

# Calcul de l'exposant de Lyapunov pour chaque a
for idx, a in enumerate(a_values):
    # Initialisation de l'orbite
    x, y = 0.0, 0.0
    
    # Élimination du transitoire
    for _ in range(n_transient):
        x_new = 1 - a*x**2 + y
        y = b*x
        x = x_new
    
    # Vecteur tangent initial (normalisé)
    v = np.array([1.0, 0.0])
    v /= np.linalg.norm(v)
    
    # Somme pour l'exposant de Lyapunov
    lyap_sum = 0.0
    diverged = False
    
    # Calcul de l'exposant
    for i in range(n_lyap):
        # Mise à jour de l'orbite
        x_new = 1 - a*x**2 + y
        y = b*x
        x = x_new
        
        # Calcul du Jacobien au point courant
        J = jacobian(x, a, b)
        
        # Application du Jacobien au vecteur tangent
        v_new = J @ v
        
        # Calcul de la norme
        norm_v = np.linalg.norm(v_new)
        
        # Vérification de la divergence
        if norm_v > lyap_threshold or np.isnan(norm_v):
            diverged = True
            break
        
        # Ré-normalisation du vecteur tangent
        v = v_new / norm_v
        
        # Accumulation du log de la norme
        lyap_sum += np.log(norm_v)
    
    # Calcul de l'exposant final
    if diverged:
        lyap_exponents[idx] = np.nan
    else:
        lyap_exponents[idx] = lyap_sum / n_lyap

# Tracé du résultat
plt.figure(figsize=(10, 6))
plt.plot(a_values, lyap_exponents, 'b-', linewidth=1.5)
plt.axhline(0, color='r', linestyle='--', alpha=0.5)  # Ligne zéro

# Mise en forme
#plt.title("Exposant de Lyapunov maximal de l'application de Hénon en fonction de $a$ ($b=0.3$)", fontsize=14)
plt.xlabel("Paramètre $a$", fontsize=12)
plt.ylabel("Exposant de Lyapunov maximal ($\lambda$)", fontsize=12)
plt.grid(alpha=0.3)
plt.xlim(a_min, a_max)

# Annotation des régimes
plt.text(0.2, -0.5, "Régime périodique", fontsize=10, ha='center')
plt.text(0.6, 0.1, "Transition", fontsize=10, ha='center')
plt.text(1.1, 0.4, "Chaos", fontsize=10, ha='center')

plt.tight_layout()
#plt.savefig('exposant_lyapunov_henon.png', dpi=150)
#plt.show()

# Exposant de Lyapunov
plt.figure(figsize=(10, 6))
plt.plot(a_values, lyap_exponents, 'b-', linewidth=1.5)
plt.axhline(0, color='r', linestyle='--', alpha=0.5)
#plt.title("Exposant de Lyapunov maximal")
plt.xlabel("Paramètre $a$")
plt.ylabel("$\lambda$")
plt.grid(alpha=0.3)
plt.xlim(a_min, a_max)

plt.tight_layout()
#plt.savefig('bifurcation_lyapunov_combo.png', dpi=150)
plt.show()
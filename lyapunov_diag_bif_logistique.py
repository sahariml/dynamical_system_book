import numpy as np
import matplotlib.pyplot as plt

# Paramètres de la simulation
a_min = 0.0
a_max = 4.0
a_steps = 2000  # Nombre de valeurs de a
n_transient = 1000  # Itérations transitoires
n_lyap = 5000  # Itérations pour calcul Lyapunov

# Initialisation
a_values = np.linspace(a_min, a_max, a_steps)
lyap_exponents = np.zeros(a_steps)

# Calcul de l'exposant de Lyapunov pour chaque a
for idx, a in enumerate(a_values):
    x = 0.5  # Condition initiale
    
    # Élimination du transitoire
    for _ in range(n_transient):
        x = a * x * (1.0 - x)
    
    # Calcul de l'exposant
    lyap_sum = 0.0
    valid_points = 0
    
    for i in range(n_lyap):
        x = a * x * (1.0 - x)
        
        # Éviter les points où la dérivée n'est pas définie
        if x < 0 or x > 1:
            continue
            
        # Calcul de la dérivée |f'(x)|
        derivative = np.abs(a * (1.0 - 2*x))
        
        # Éviter les dérivées nulles ou infinies
        if derivative > 1e-10 and derivative < 1e10:
            lyap_sum += np.log(derivative)
            valid_points += 1
    
    # Calcul de l'exposant final
    if valid_points > 0:
        lyap_exponents[idx] = lyap_sum / valid_points
    else:
        lyap_exponents[idx] = np.nan

# Tracé du résultat
plt.figure(figsize=(12, 7))
plt.plot(a_values, lyap_exponents, 'b-', linewidth=0.8, alpha=0.8)
plt.axhline(0, color='r', linestyle='--', alpha=0.5)  # Ligne zéro

# Mise en forme
#plt.title("Exposant de Lyapunov de l'application logistique", fontsize=16)
plt.xlabel("Paramètre $a$", fontsize=14)
plt.ylabel("Exposant de Lyapunov ($\lambda$)", fontsize=14)
plt.grid(alpha=0.3)
plt.xlim(a_min, a_max)
plt.ylim(-2, 1)  # Plage typique pour λ

# Annotation des régimes remarquables
plt.text(3.0, -1.5, "Périodique", fontsize=12, ha='center')
plt.text(3.57, 0.5, "Chaos", fontsize=12, ha='center')
plt.text(3.83, -0.5, "Fenêtre périodique", fontsize=10, ha='center')
plt.text(2.5, 0.2, "Zone stable", fontsize=10)

# Tracé combiné avec le diagramme de bifurcation
plt.figure(figsize=(12, 10))

# Diagramme de bifurcation
plt.subplot(2, 1, 1)
x = 0.5 * np.ones_like(a_values)
for _ in range(500):  # Itérations transitoires
    x = a_values * x * (1.0 - x)
for _ in range(100):  # Itérations à conserver
    x = a_values * x * (1.0 - x)
    plt.plot(a_values, x, '.k', alpha=0.05, markersize=0.5)

#plt.title("Diagramme de bifurcation", fontsize=14)
plt.ylabel("$x_n$", fontsize=12)
plt.xlim(a_min, a_max)
plt.grid(alpha=0.2)

# Exposant de Lyapunov
plt.subplot(2, 1, 2)
plt.plot(a_values, lyap_exponents, 'b-', linewidth=0.8, alpha=0.8)
plt.axhline(0, color='r', linestyle='--', alpha=0.5)
#plt.title("Exposant de Lyapunov", fontsize=14)
plt.xlabel("Paramètre $a$", fontsize=12)
plt.ylabel("$\lambda$", fontsize=12)
plt.grid(alpha=0.3)
plt.xlim(a_min, a_max)
plt.ylim(-2, 1)
# Annotation des régimes remarquables
plt.text(3.0, -1.5, "Périodique", fontsize=12, ha='center')
plt.text(3.57, 0.5, "Chaos", fontsize=12, ha='center')
plt.text(3.83, -0.5, "Fenêtre périodique", fontsize=10, ha='center')
plt.text(2.5, 0.2, "Zone stable", fontsize=10)

plt.tight_layout()
#plt.savefig('logistic_bifurcation_lyapunov.png', dpi=150)
plt.show()
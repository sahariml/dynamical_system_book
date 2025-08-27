import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres globaux ---
b = 0.3                      # Valeur fixée de b
a_vals = np.linspace(1.0, 1.5, 200)  # Plage des valeurs de a
n_iter = 5000
transient = 1000
lyap_max = []

# --- Fonctions de Hénon et Jacobienne ---
def henon_map(x, y, a, b):
    x_next = 1 - a * x**2 + y
    y_next = b * x
    return x_next, y_next

def jacobian(x, y, a, b):
    return np.array([[-2 * a * x, 1],
                     [b,          0]])

# --- Boucle sur les valeurs de a ---
for a in a_vals:
    x, y = 0.1, 0.1
    vecs = np.identity(2)
    lyap_sum = np.zeros(2)

    for i in range(n_iter):
        x, y = henon_map(x, y, a, b)
        if i >= transient:
            J = jacobian(x, y, a, b)
            vecs = J @ vecs
            q, r = np.linalg.qr(vecs)
            lyap_sum += np.log(np.abs(np.diag(r)))
            vecs = q

    exponents = lyap_sum / (n_iter - transient)
    lyap_max.append(np.max(exponents))  # Seulement le plus grand exposant

# --- Affichage du graphique ---
plt.figure(figsize=(10, 5))
plt.plot(a_vals, lyap_max, 'b')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Exposant de Lyapunov maximal en fonction de a (b = 0.3)")
plt.xlabel("a")
plt.ylabel("Exposant de Lyapunov maximal")
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def arnold_map(x, y):
    return (x + y) % 1, (x + 2*y) % 1

# Condition initiale
x, y = 0.123456789, 0.987654321
n_iter = 1000

# Stockage des points
points = np.zeros((n_iter, 2))
points[0] = [x, y]

# Itération
for i in range(1, n_iter):
    x, y = arnold_map(x, y)
    points[i] = [x, y]

# Visualisation
plt.figure(figsize=(8, 8), dpi=150)
plt.scatter(points[:,0], points[:,1], s=1, c='blue', alpha=0.6)
#plt.title("Orbite typique du chat d'Arnold (1000 itérations)")
plt.xlabel('$x_n$')
plt.ylabel('$y_n$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.savefig('arnold_orbiteps.eps', bbox_inches='tight')
plt.show()

# Deux conditions initiales proches
x1, y1 = 0.1, 0.2
x2, y2 = 0.10001, 0.2
n_iter = 50

distances = []

for i in range(n_iter):
    x1, y1 = arnold_map(x1, y1)
    x2, y2 = arnold_map(x2, y2)
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    distances.append(dist)

# Calcul de la régression linéaire pour l'exposant de Lyapunov
log_dist = np.log(distances)
A = np.vstack([np.arange(n_iter), np.ones(n_iter)]).T
slope, _ = np.linalg.lstsq(A, log_dist, rcond=None)[0]

# Visualisation
plt.figure(figsize=(10, 6), dpi=150)
plt.semilogy(range(n_iter), distances, 'r-', label='Distance')
plt.plot(range(n_iter), np.exp(slope*np.arange(n_iter)), 'b--', 
         label=f'Pente exponentielle: {slope:.3f}')
#plt.title("Sensibilité aux conditions initiales")
plt.xlabel('Itération')
plt.ylabel('Distance (échelle log)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.savefig('arnold_sensitivity.eps', bbox_inches='tight')
plt.show()

print(f"Exposant de Lyapunov estimé: {slope:.4f} (théorique: {np.log((3+5**0.5)/2):.4f})")

# Calcul des points de période 2
A = np.array([[1, 1], [1, 2]])
A2 = A @ A
I = np.eye(2)

# Résolution de (A² - I)v ≡ 0 mod 1
solutions = []
for x in np.linspace(0, 0.999, 100):
    for y in np.linspace(0, 0.999, 100):
        v = np.array([x, y])
        if np.allclose((A2 @ v) % 1, v % 1, atol=1e-4):
            solutions.append((x, y))

# Éliminer les doublons
solutions = np.unique(np.round(solutions, 4), axis=0)

# Visualisation
plt.figure(figsize=(8, 8), dpi=150)
plt.scatter(solutions[:,0], solutions[:,1], s=80, c='red', marker='o')
plt.title("Points périodiques de période 2")
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)

# Annoter les points
for i, (x, y) in enumerate(solutions):
    plt.text(x+0.02, y+0.02, f'({x:.1f}, {y:.1f})', fontsize=9)

plt.savefig('arnold_period2.eps', bbox_inches='tight')
plt.show()

print("Points de période 2:")
for pt in solutions:
    print(f"({pt[0]:.1f}, {pt[1]:.1f})")

# Génération d'un diagramme de phase
x_vals = np.linspace(0, 1, 20)
y_vals = np.linspace(0, 1, 20)

plt.figure(figsize=(10, 10), dpi=150)
for x in x_vals:
    for y in y_vals:
        xn, yn = arnold_map(x, y)
        dx = (xn - x) * 0.1
        dy = (yn - y) * 0.31
        plt.arrow(x, y, dx, dy, 
                  head_width=0.02, 
                  head_length=0.03, 
                  fc='blue', 
                  ec='blue',
                  alpha=0.4)

#plt.title("Champ vectoriel du chat d'Arnold")
plt.xlabel('$x_n$')
plt.ylabel('$y_n$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.savefig('arnold_phase.eps', bbox_inches='tight')
plt.show()
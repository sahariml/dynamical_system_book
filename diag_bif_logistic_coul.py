import numpy as np
import matplotlib.pyplot as plt

# Paramètres
r_min, r_max = 0, 4.0   # plage de r
#r_min, r_max = 3, 4.0   # plage de r
#r_min, r_max = 3.82, 3.860   # plage de r
#r_min, r_max = 3.86, 4.0   # plage de r
points = 1000             # nombre de valeurs de r
iterations = 1000         # itérations par r
last = 200                 # nombre de points à garder après convergence

# Grille de paramètres
r_values = np.linspace(r_min, r_max, points)
x = 1e-5 * np.ones(points)  # condition initiale (petite valeur positive)

# Pour stocker les points du diagramme
R, X = [], []

for i in range(iterations):
    x = r_values * x * (1 - x)  # application logistique
    if i >= (iterations - last):
        R.extend(r_values)
        X.extend(x)

# Tracé avec couleurs selon r
plt.figure(figsize=(16, 9))
#sc = plt.scatter(R, X, c=R, cmap='jet', s=0.2, marker='.')
plt.scatter(R, X, c=X, cmap='jet', s=0.2, marker='.', rasterized=True)
#plt.colorbar(sc, label='Valeur de r')
#plt.title("Carte logistique avec coloration selon r")
plt.xlabel("$\mu$",fontsize=14)
plt.ylabel("$x_\infty$", fontsize=14)
# Sauvegarde en EPS haute résolution
#plt.savefig("diag_bif_logistic.eps", format='eps', bbox_inches='tight', pad_inches=0, dpi=100)
plt.savefig("diag_bif_logistic.eps", dpi=150)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Paramètre de la fonction
#c = -0.8 + 0.156j
c =-0.123 + 0.745j
#c=-1
#c=0
# Taille et domaine
width, height = 800, 800
xmin, xmax = -1.5, 1.5
ymin, ymax = -1.5, 1.5
max_iter = 300

# Grille complexe
x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
img = np.zeros(Z.shape, dtype=int)

for i in range(max_iter):
    mask = np.abs(Z) <= 2
    Z[mask] = Z[mask] ** 2 + c
    img[mask] += 1

plt.figure(figsize=(8, 8))
plt.imshow(img, extent=[xmin, xmax, ymin, ymax], cmap='inferno', origin='lower')
#plt.title(f"Ensemble de Julia pour c = {c}")
plt.tight_layout()
plt.xlabel('$Re(z)$')
plt.ylabel('$Im(z)$')
#plt.axis('off')
plt.savefig("julia_0.eps", dpi=300, bbox_inches='tight')
plt.show()

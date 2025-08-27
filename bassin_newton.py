import numpy as np
import matplotlib.pyplot as plt

# Paramètres du plan complexe
np1, np2 = 800, 800
ngrand = 50
tol = 1e-6  # Tolérance pour la convergence

x0, x1 = -2, 2
y0, y1 = -2, 2

# Calcul des pas
pas1 = (x1 - x0) / np1
pas2 = (y1 - y0) / np2

# Racines de z^3 - 1 = 0
racines = [1, -0.5 + 1j*np.sqrt(3)/2, -0.5 - 1j*np.sqrt(3)/2]
couleurs_base = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Rouge, Vert, Bleu

# Initialisation image
image = np.zeros((np2+1, np1+1, 3), dtype=np.float32)

# Fonction de Newton pour z^3 - 1
def newton(z):
    return z - (z**3 - 1) / (3*z**2)

# Remplissage de l'image
for i in range(np1 + 1):
    for j in range(np2 + 1):
        xa = x0 + i * pas1
        ya = y0 + j * pas2
        z = complex(xa, ya)
        
        # Éviter la division par zéro
        if abs(z) < 1e-10:
            image[np2 - j, i] = (0, 0, 0)
            continue
            
        converged = False
        for k in range(ngrand):
            z_new = newton(z)
            
            # Vérifier la convergence vers une racine
            for idx, racine in enumerate(racines):
                if abs(z_new - racine) < tol:
                    # Intensité basée sur le nombre d'itérations
                    intensity = 1.0 - (k / ngrand) ** 0.5
                    color = [c * intensity for c in couleurs_base[idx]]
                    image[np2 - j, i] = color
                    converged = True
                    break
                    
            if converged:
                break
                
            z = z_new
            
            # Éviter les overflow
            if abs(z) > 1e10:
                break
                
        if not converged:
            image[np2 - j, i] = (0, 0, 0)

# Affichage de l'image
plt.figure(figsize=(10, 10))
plt.imshow(image, extent=(x0, x1, y0, y1))
plt.axis('off')
plt.title("Bassins d'attraction pour z³ = 1")
plt.tight_layout()
plt.show()
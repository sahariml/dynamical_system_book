# Re-importation après reset
import numpy as np
import matplotlib.pyplot as plt

# Ensemble de Mandelbrot (version large)
def mandelbrot(width=1600, height=1200, max_iter=100, x_min=-2.5, x_max=1.5, y_min=-1.5, y_max=1.5):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    div_time = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        Z = Z**2 + C
        mask = (np.abs(Z) > 2) & (div_time == 0)
        div_time[mask] = i

    return x, y, div_time

x, y, mandelbrot_set = mandelbrot()

# Affichage
plt.figure(figsize=(12, 9))
plt.imshow(mandelbrot_set, extent=[x.min(), x.max(), y.min(), y.max()], cmap='hot', origin='lower')
plt.colorbar(label='Itérations avant divergence')
plt.title('Ensemble de Mandelbrot (version large)')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.tight_layout()
plt.savefig('mandelbrot_large.eps', dpi=300)
plt.show()

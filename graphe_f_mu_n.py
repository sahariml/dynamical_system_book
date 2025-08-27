import numpy as np
import matplotlib.pyplot as plt

# Fonction logistique
def f(x, mu):
    return mu * x * (1 - x)

# Itérations de la fonction
def iterate_f(func, x, mu, n):
    y = x
    for _ in range(n):
        y = func(y, mu)
    return y

# Paramètres
mu = 3.2
x = np.linspace(0, 1, 500)

# Calculs
f1 = f(x, mu)
f2 = iterate_f(f, x, mu, 2)
f4 = iterate_f(f, x, mu, 4)

# Tracé
plt.figure(figsize=(16, 9))
plt.plot(x, f1, label=r"$f_\mu(x)$", lw=2, color='blue')
plt.plot(x, f2, label=r"$f_\mu^{2}(x)$", lw=2, color='green')
plt.plot(x, f4, label=r"$f_\mu^{4}(x)$", lw=2, color='red')
plt.plot(x, x, 'k--', lw=1.2, label=r"$y = x$")

# Axes et légendes
plt.xlabel(r"$x$", fontsize=14)
#plt.ylabel(r"$f_\mu(x)$", fontsize=14)
#plt.title(fr"Graphes de $f_\mu$, $f_\mu^2$, $f_\mu^4$ avec $\mu={mu}$", fontsize=16)
plt.grid(True)
plt.legend()
plt.show()

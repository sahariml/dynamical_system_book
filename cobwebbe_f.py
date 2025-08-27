import numpy as np
import matplotlib.pyplot as plt

# Fonction logistique
def f(x, mu):
    return mu * x * (1 - x)

# Diagramme cobweb avec point initial (x0, 0)
def cobweb_plot(mu, x0, iterations, x_min=0, x_max=1.0):
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.linspace(x_min, x_max, 400)
    
    
    # Tracé de f_mu(x) et y = x
    ax.plot(x, f(x, mu), 'b', lw=3, label="$f_\mu(x)$")
    ax.plot(x, x, 'k--', lw=1.2, label="$y = x$")
    
    # Premier point : (x0, 0)
    xn = x0
    ax.plot([xn, xn], [0, f(xn, mu)], 'g', alpha=0.7)  # montée verticale
    ax.plot([xn, f(xn, mu)], [f(xn, mu), f(xn, mu)], 'g', alpha=0.7)  # horizontale
    
    # Itérations suivantes
    for _ in range(iterations - 1):
        yn = f(xn, mu)
        xn = yn
        ax.plot([xn, xn], [xn, f(xn, mu)], 'g', alpha=0.7)  # verticale
        ax.plot([xn, f(xn, mu)], [f(xn, mu), f(xn, mu)], 'g', alpha=0.7)  # horizontale
    
    # Axes
    ax.axhline(0, color='black', lw=1.2)
    ax.axvline(0, color='black', lw=1.2)
    
    # Limites et labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$f_\mu(x)$", fontsize=14)
    #ax.set_title(fr"Cobweb Plot pour $f_\mu(x) = \mu x (1 - x)$, $\mu={mu}$", fontsize=16)
    ax.legend()
    plt.grid(True)
    plt.show()

# Exemple : μ=3.5, départ à x0=0.2
mu = 2.5
x0 = 0.9
iterations = 20
cobweb_plot(mu, x0, iterations)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Use LaTeX labels
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
rc('text', usetex=True)
dpi = 100

def plot_cobweb_two_points(f, r, x0, eps=1e-5, nmax=50):
    """Cobweb plot showing two nearby trajectories."""
    x = np.linspace(0, 1, 500)
    fig = plt.figure(figsize=(600/dpi, 450/dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Plot f(x) and y=x
    ax.plot(x, f(x, r), 'k-', lw=2, label=r'$f_{\mu}(x)$')
    ax.plot(x, x, 'k--', lw=1, label=r'$y = x$')

    # Initial points
    x1 = x0
    x2 = x0 + eps

    px1, py1 = [x1], [0]
    px2, py2 = [x2], [0]

    for _ in range(nmax):
        y1 = f(x1, r)
        y2 = f(x2, r)

        # Trajectoire 1
        px1.extend([x1, y1])
        py1.extend([y1, y1])
        x1 = y1

        # Trajectoire 2
        px2.extend([x2, y2])
        py2.extend([y2, y2])
        x2 = y2

    ax.plot(px1, py1, 'b-', alpha=0.7, label=r'$x_0$')
    ax.plot(px2, py2, 'r-', alpha=0.7, label=r'$x_0 + \varepsilon$')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f_{\mu}(x)$')
    ax.set_title(r'$x_0 = {:.5f}$, $\varepsilon = {:.1e}$, $r = {:.2f}$'.format(x0, eps, r))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

class AnnotatedFunction:
    def __init__(self, func, latex_label):
        self.func = func
        self.latex_label = latex_label

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# Fonction logistique
logistic = AnnotatedFunction(lambda x, r: r * x * (1 - x), r'rx(1-x)')

# ➤ Essayons avec un r chaotique
plot_cobweb_two_points(logistic, r=3.9, x0=0.2, eps=1e-6)

import numpy as np
import matplotlib.pyplot as plt

def step(A, vec):
    return (A @ vec) % 1.0

a_vals = np.linspace(-0.5, 2.0, 601)
nit = 500     # itérations totales
trans = 200   # transitoire à jeter
points_per_a = 200  # nombre d'itérations tracées après transitoire

# point initial (peut varier)
x0 = np.array([0.123456, 0.654321])

plt.figure(figsize=(9,6))
for a in a_vals:
    A = np.array([[1.0, a],[1.0, 1.0 + a]])
    v = x0.copy()
    # itérations transitoires
    for _ in range(trans):
        v = step(A, v)
    # collecte
    xs = []
    for _ in range(points_per_a):
        v = step(A, v)
        xs.append(v[0])  # on trace seulement la coordonnée x modulo 1
    # tracer: pour réduire le surplotting, on trace chaque x comme point
    plt.scatter([a]*len(xs), xs, s=0.2, marker='.')  # ',k' = très petit marker
plt.xlabel('$a$')
plt.ylabel('$x_\infty$')
#plt.title("Diagramme style bifurcation pour la famille A_a")
plt.ylim(0,1)
plt.xlim(a_vals.min(), a_vals.max())
plt.grid(True)
#plt.savefig('bifurcation_style_arnold.png', dpi=300)
plt.show()




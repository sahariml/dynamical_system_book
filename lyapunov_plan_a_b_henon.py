import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def henon_map(x, y, a, b):
    x_next = 1 - a * x**2 + y
    y_next = b * x
    return x_next, y_next

def compute_lyapunov_exponent(a, b, N=1000, discard=100):
    x, y = 0.0, 0.0
    delta = 1e-8
    le_sum = 0.0

    # initial perturbation vector
    dx, dy = delta, 0.0

    for i in range(N + discard):
        # Current point
        x1 = x

        # Apply map
        x, y = henon_map(x, y, a, b)

        # Tangent map (Jacobian at x1, y1)
        J = np.array([[-2 * a * x1, 1],
                      [b, 0]])

        # Apply linearized map to perturbation
        d = np.dot(J, [dx, dy])
        norm_d = np.linalg.norm(d)
        dx, dy = d / norm_d  # renormalize

        if i >= discard:
            le_sum += np.log(norm_d)

    return le_sum / N

# Grid of parameters
a_vals = np.linspace(1.0, 1.4, 300)
b_vals = np.linspace(0.2, 0.4, 300)
LE = np.zeros((len(b_vals), len(a_vals)))

print("Computing Lyapunov exponents...")
for i, b in enumerate(tqdm(b_vals)):
    for j, a in enumerate(a_vals):
        LE[i, j] = compute_lyapunov_exponent(a, b)

# Plot
plt.figure(figsize=(16, 9))
plt.imshow(LE, extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label='Lyapunov exponent')
plt.xlabel('$a$')
plt.ylabel('$b$')
#plt.title('Lyapunov Exponent Diagram of the Hénon Map')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def max_lyap_of_matrix(A):
    vals = np.linalg.eigvals(A)
    return max(np.log(np.abs(vals)))

a_vals = np.linspace(-1.5, 2.5, 801)
lyap = [max_lyap_of_matrix(np.array([[1, a],[1, 1+a]])) for a in a_vals]

plt.figure(figsize=(8,4))
plt.plot(a_vals, lyap)
plt.axhline(0, color='k', linewidth=0.6, linestyle='--')
plt.xlabel('$a$')
plt.ylabel('$\max \{\log(|\lambda|)\}$')
#plt.title('Exposant (log) maximal des valeurs propres de A_a')
plt.grid(True)
plt.savefig('lyapunov_vs_a.png', dpi=300)
plt.show()

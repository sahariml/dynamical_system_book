import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'application de Hénon
a = 1.4
b = 0.3
n_iterations = 100

# Conditions initiales
X1, Y1 = [0.0], [0.0]       # Première condition
X2, Y2 = [0.001], [0.001]  # Deuxième condition (proche)

# Calcul des itérations
for n in range(n_iterations - 1):
    # Première trajectoire
    x1_next = 1 - a * X1[n]**2 + Y1[n]
    y1_next = b * X1[n]
    X1.append(x1_next)
    Y1.append(y1_next)
    
    # Deuxième trajectoire
    x2_next = 1 - a * X2[n]**2 + Y2[n]
    y2_next = b * X2[n]
    X2.append(x2_next)
    Y2.append(y2_next)

# Figure 1 : Évolution de X_n
plt.figure(figsize=(10, 6))
plt.plot(X1, 'b-', linewidth=1.5, label='$(X_0,Y_0)=(0,0)$')
plt.plot(X2, 'r--', linewidth=1.5, label='$(X_0,Y_0)=(0.001,0.001)$')
#plt.title('Évolution de $X_n$ pour deux conditions initiales proches', fontsize=14)
plt.xlabel('$n$', fontsize=12)
plt.ylabel('$X_n$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(0, n_iterations)
plt.ylim(min(min(X1), min(X2)) - 0.1, max(max(X1), max(X2)) + 0.1)
plt.tight_layout()
#plt.savefig('henon_Xn_evolution.png', dpi=150)
#plt.show()

# Figure 2 : Évolution de Y_n
plt.figure(figsize=(10, 6))
plt.plot(Y1, 'b-', linewidth=1.5, label='$(X_0,Y_0)=(0,0)$')
plt.plot(Y2, 'r--', linewidth=1.5, label='$(X_0,Y_0)=(0.001,0.001)$')
#plt.title('Évolution de $Y_n$ pour deux conditions initiales proches', fontsize=14)
plt.xlabel('$n$', fontsize=12)
plt.ylabel('$Y_n$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(0, n_iterations)
plt.ylim(min(min(Y1), min(Y2)) - 0.05, max(max(Y1), max(Y2)) + 0.05)
plt.tight_layout()
#plt.savefig('henon_Yn_evolution.png', dpi=150)
#plt.show()

# Figure 3 : Divergence des trajectoires (optionnelle)
plt.figure(figsize=(10, 6))
diff_X = np.abs(np.array(X1) - np.array(Y1))
diff_Y = np.abs(np.array(X2) - np.array(Y2))

plt.semilogy(diff_X, 'b-', label='|$X_{n}-X_{n}$|')
plt.semilogy(diff_Y, 'r-', label='|$Y_{n}-Y_{n}$|')
#plt.title('Divergence des trajectoires (échelle logarithmique)', fontsize=14)
plt.xlabel('$n$', fontsize=12)
plt.ylabel('Différence absolue', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3, which='both')
plt.xlim(0, n_iterations)
plt.tight_layout()
#plt.savefig('henon_divergence.png', dpi=150)
plt.show()
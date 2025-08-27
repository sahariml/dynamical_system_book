import matplotlib.pyplot as plt

# Parameters
a, b = 1.4, 0.3
n_iter = 5000
delta=0.1
# Initial conditions
x0, y0 = 0, 0
#x0_perturbed, y0_perturbed = x0+delta, y0+delta

# Iteration function
def henon_map(x, y, a, b):
    return 1 - a * x**2 + y, b * x

x, y = [x0], [y0]
#x_perturbed, y_perturbed = [x0_perturbed], [y0_perturbed]

for _ in range(n_iter):
    xn, yn = henon_map(x[-1], y[-1], a, b)
    x.append(xn)
    y.append(yn)
    
    #xn_p, yn_p = henon_map(x_perturbed[-1], y_perturbed[-1], a, b)
 #   x_perturbed.append(xn_p)
 #   y_perturbed.append(yn_p)

# Plotting orbits
plt.plot(x[2:], y[2:], 'b.', markersize=2)
#plt.plot(x_perturbed[2:], y_perturbed[2:], 'r.', markersize=2,label='Orbite perturbee')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$y$',fontsize=14)
plt.legend()
#plt.title("Application de Hénon")
plt.show()
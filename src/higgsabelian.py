import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lambda_val = 1.0  # Coupling constant
v = 1.0  # Vacuum expectation value
phi_values = np.linspace(-2, 2, 100)
phi_grid = np.meshgrid(phi_values, phi_values)

# Higgs Potential
higgs_potential = (lambda_val / 4) * (phi_grid[0]**2 - v**2 / 2)**2

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Abelian Higgs Model: Higgs Potential')
    ax.set_xlabel('Higgs Field (phi)')
    ax.set_ylabel('Potential Energy')
    ax.set_zlabel('Potential Energy')
    ax.plot_surface(phi_grid[0], phi_grid[1], higgs_potential, cmap='viridis')
    plt.show()
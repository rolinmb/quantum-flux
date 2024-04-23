import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

mu_squared = 1.0  # Mass term coefficient
lambda_val = 1.0  # Self-interaction coefficient

phi_values = np.linspace(-2, 2, 100)
v_values = np.linspace(0, 2, 100)
phi_grid, v_grid = np.meshgrid(phi_values, v_values)

def get_higgs_potential(phi, v):
    return 0.5 * mu_squared * v**2 + 0.25 * lambda_val * v**4 + 0.5 * mu_squared * phi**2

def update(frame):
    ax.clear()
    t = frame * 0.1  # Time parameter
    phi_values = np.linspace(-2, 2, 100)
    v_values = np.linspace(0, 2, 100)
    phi_grid, v_grid = np.meshgrid(phi_values, v_values)
    higgs_potential = get_higgs_potential(phi_grid, v_grid - t)
    ax.plot_surface(phi_grid, v_grid, higgs_potential, cmap='viridis')
    ax.set_title(f'Higgs Potential (t={t:.1f})')
    ax.set_xlabel('Higgs Field (phi)')
    ax.set_ylabel('Vacuum Expectation Value (v)')
    ax.set_zlabel('Potential Energy')

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=range(50), interval=100)
    plt.show()
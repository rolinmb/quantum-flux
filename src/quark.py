import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_SIZE = 100
X_MIN, X_MAX = -5, 5
Y_MIN, Y_MAX = -5, 5
T_MIN, T_MAX = 0, 10
DT = 0.1
mu = 0  # Mean position
sigma = 1  # Standard deviation
k = 2 * np.pi / (X_MAX - X_MIN)  # Wavenumber

def fermionic_wavefunction(x, y, t):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) * np.exp(-0.5 * ((y - mu) / sigma)**2) * np.sin(k * x - t)

if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, GRID_SIZE)
    y = np.linspace(Y_MIN, Y_MAX, GRID_SIZE)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        t = frame * DT
        Z = np.abs(fermionic_wavefunction(X, Y, t))**2
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_title(f'Evolution of Fermionic Wavefunction (Quark), t={t:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')

    animation = FuncAnimation(fig, update, frames=np.arange(T_MIN, T_MAX, DT), interval=50)
    plt.show()
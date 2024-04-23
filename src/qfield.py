import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

GRID_SIZE = 80
X_MIN, X_MAX = -5, 5
Y_MIN, Y_MAX = -5, 5
T_MIN, T_MAX = 0, 10
DT = 0.0025
DX = (X_MAX - X_MIN) / GRID_SIZE
DY = (Y_MAX - Y_MIN) / GRID_SIZE

SIGMA = 0.777 # Gaussian Wave Packet Initial Width
k0 = 7 # Initial Wavenumber (look into de Broglie relation: p=ℏk)

def V(x, y): # Potential function (harmonic oscillator)
    return 0.5 * (x**2 + y**2)

x = np.linspace(X_MIN, X_MAX, GRID_SIZE)
y = np.linspace(Y_MIN, Y_MAX, GRID_SIZE)
X, Y = np.meshgrid(x, y)
psi = np.exp(-(X**2 + Y**2) / (2 * SIGMA**2)) * np.exp(1j * k0 * (X + Y))
psi_next = np.zeros_like(psi)

def evolve():
    global psi, psi_next
    for i in range(1, GRID_SIZE - 1):
        for j in range(1, GRID_SIZE - 1):
            psi_next[i, j] = psi[i, j] - 1j * (
                (psi[i+1, j] - 2*psi[i, j] + psi[i-1, j]) / DX**2 +
                (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1]) / DY**2
            ) * DT / 2 - DT * V(X[i, j], Y[i, j]) * psi[i, j]
    psi = psi_next.copy()

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), np.real(psi), np.abs(psi)**2, length=0.005, normalize=True)
        ax.set_title(f'Schrödinger Equation 3D Time evolution (Frame {frame})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        evolve()

    animation = FuncAnimation(fig, update, frames=np.arange(T_MIN, T_MAX, DT), interval=50)
    plt.show()
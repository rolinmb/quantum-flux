import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_SIZE = 100
X_MIN, X_MAX = -5, 5
T_MIN, T_MAX = 0, 10
DT = 0.01
DX = (X_MAX - X_MIN) / GRID_SIZE

SIGMA = 1
k0 = 5

# Potential function (harmonic oscillator)
def V(x):
    return 0.5 * x**2

if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, GRID_SIZE)
    psi = np.exp(-(x**2) / (2 * SIGMA**2)) * np.exp(1j * k0 * x)
    psi_next = np.zeros_like(psi)
    
    def evolve():
        global psi, psi_next
        for i in range(1, GRID_SIZE - 1):
            psi_next[i] = psi[i] - 1j * (psi[i+1] - 2*psi[i] + psi[i-1]) * DT / (2 * DX**2) - DT * V(x[i]) * psi[i]
        psi = psi_next.copy()

    def update(frame):
        ax.cla()
        ax.plot(x, np.abs(psi)**2, color='b')
        ax.set_title(f'Schrödinger Equation 2D Time evolution (Frame {frame})')
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability Density')
        evolve()

    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update, frames=np.arange(T_MIN, T_MAX, DT), interval=50)
    plt.show()
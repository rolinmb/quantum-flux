import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_SIZE = 100
X_MIN, X_MAX = -10, 10
T_MIN, T_MAX = 0, 20
DT = 0.01
DX = (X_MAX - X_MIN) / GRID_SIZE

SIGMA = 1
k0 = 5

# Potential function (harmonic oscillator)
def V(x):
    return 0.5 * x**2

if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, GRID_SIZE)
    
    # Initialize two wave packets
    psi1 = np.exp(-((x - 2)**2) / (2 * SIGMA**2)) * np.exp(1j * k0 * x)
    psi2 = np.exp(-((x + 2)**2) / (2 * SIGMA**2)) * np.exp(1j * k0 * x)
    
    def evolve(psi):
        psi_next = np.zeros_like(psi)
        for i in range(1, GRID_SIZE - 1):
            psi_next[i] = psi[i] - 1j * (psi[i+1] - 2*psi[i] + psi[i-1]) * DT / (2 * DX**2) - DT * V(x[i]) * psi[i]
        return psi_next

    def update(frame):
        ax.cla()
        
        # Evolve both wave packets
        global psi1, psi2
        psi1 = evolve(psi1)
        psi2 = evolve(psi2)
        
        # Plot both wave packets
        ax.plot(x, np.abs(psi1)**2, color='b', label='Wave Packet 1')
        ax.plot(x, np.abs(psi2)**2, color='r', label='Wave Packet 2')
        
        ax.set_title(f'Collision of Two Wave Packets (Frame {frame})')
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability Density')
        ax.legend()

    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update, frames=np.arange(T_MIN, T_MAX, DT), interval=50)
    plt.show()
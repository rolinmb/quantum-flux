import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from matplotlib.colors import Normalize

GRID_SIZE = 1000
X_MIN, X_MAX = -5, 5
N_STATES = 10

def harmonic_oscillator_potential(x):
    return 0.5 * x**2

def probability_density_n(x, n):
    prefactor = 1 / np.sqrt(2**n * np.math.factorial(n)) * (1 / np.pi)**0.25
    return prefactor * np.exp(-x**2 / 2) * eval_hermite(n, x)

if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, GRID_SIZE)
    fig, ax = plt.subplots()

    for n in range(N_STATES):
        psi_n = probability_density_n(x, n)
        probability_density = np.abs(psi_n)**2
        norm = Normalize(vmin=0, vmax=np.max(probability_density))
        for i in range(len(x)):
            color = plt.cm.viridis(norm(probability_density[i]))
            ax.plot(x[i], probability_density[i] + n, color=color, marker='o', markersize=1)

    ax.plot(x, harmonic_oscillator_potential(x), color='black', linestyle='--', label='Potential Energy')
    ax.legend()
    ax.set_title('Probability Densities for Quantum Harmonic Oscillator Eigenstates')
    ax.set_xlabel('Position')
    ax.set_ylabel('Eigenstate Index')
    plt.show()
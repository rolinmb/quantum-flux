import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

L = 10  # Size of the spatial grid
T = 10  # Total time
Nx = Ny = Nz = 50  # Number of grid points in each dimension
Nt = 100  # Number of time steps
dx = dy = dz = L / (Nx - 1)  # Spatial step size
dt = T / Nt  # Time step size
c = 0.5  # Speed of propagation (for simplicity)
phi = np.zeros((Nx, Ny, Nz))
phi[Nx // 2, Ny // 2, Nz // 2] = 1  # Initial condition (e.g., a single excitation)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
z = np.linspace(0, L, Nz)
X, Y, Z = np.meshgrid(x, y, z)

def update(frame):
    global phi
    phi_new = np.zeros_like(phi)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                laplacian = (phi[i+1, j, k] + phi[i-1, j, k] - 2*phi[i, j, k]) / dx**2 + \
                            (phi[i, j+1, k] + phi[i, j-1, k] - 2*phi[i, j, k]) / dy**2 + \
                            (phi[i, j, k+1] + phi[i, j, k-1] - 2*phi[i, j, k]) / dz**2
                phi_new[i, j, k] = phi[i, j, k] + c**2 * dt**2 * laplacian
    phi = phi_new
    grad_phi_x, grad_phi_y, grad_phi_z = np.gradient(phi, dx, dy, dz)
    norm = np.sqrt(grad_phi_x**2 + grad_phi_y**2 + grad_phi_z**2)
    norm[norm == 0] = 1  # Avoid division by zero
    grad_phi_x /= norm
    grad_phi_y /= norm
    grad_phi_z /= norm
    ax.clear()
    ax.set_title(f'Time step {frame+1}')
    ax.quiver(X, Y, Z, grad_phi_x, grad_phi_y, grad_phi_z, length=0.2)

if __name__ == "__main__":
    ani = FuncAnimation(fig, update, frames=Nt, interval=100)
    plt.show()
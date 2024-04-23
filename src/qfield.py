import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

grid_size = 50
num_steps = 50
time_step = 0.1
diffusion_coefficient = 0.1
grid = np.zeros((grid_size, grid_size, grid_size))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(step):
    global grid
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                grid[i, j, k] += diffusion_coefficient * (np.random.rand() - 0.5) # diffusion process

    ax.clear()
    ax.set_title(f'Time step {step}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(*np.where(grid > 0), c='b', alpha=0.1)

if __name__ == "__main__":
    ani = FuncAnimation(fig, update, frames=num_steps, interval=100)
    plt.show()
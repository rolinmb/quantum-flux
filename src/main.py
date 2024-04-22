import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

GRIDSIZE = 60
dt = 0.1

def update_field(field):
    laplacian = np.zeros_like(field)
    for i in range(3):
        laplacian += np.roll(field, 1, axis=i) + np.roll(field, -1, axis=i)
    laplacian -= 6 * field
    field += dt * laplacian
    return field

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=45)
    field = np.random.randn(GRIDSIZE, GRIDSIZE, GRIDSIZE)

    def update_frame(frame):
        global field
        ax.clear()
        field = update_field(field)
        x, y, z = np.meshgrid(np.arange(GRIDSIZE), np.arange(GRIDSIZE), np.arange(GRIDSIZE))
        norm = mcolors.Normalize(vmin=np.min(field), vmax=np.max(field))
        ax.scatter(x, y, z, c=field.flatten(), cmap="coolwarm", norm=norm)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Quantum Field Vaccum Fluctuations (Frame {})".format(frame))
    
    animate = FuncAnimation(fig, update_frame, frames=100, interval=1000)
    plt.show()

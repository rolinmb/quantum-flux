import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

init_spinor = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

"""
def plot_spinor(spinor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = spinor[:, 0]
    y = spinor[:, 1]
    z = spinor[:, 2]
    ax.plot(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
"""
def update(frame, line):
    angle = 0.1 * frame
    rotmat= np.array([
	[np.cos(angle), -np.sin(angle), 0],
	[np.sin(angle), np.cos(angle), 0],
	[0, 0, 1]
    ])
    newvec = np.dot(rotmat, init_spinor)
    line.set_data(newvec[:2, 0], newvec[:2, 1])
    line.set_3d_properties(newvec[:2, 2])
    return line

if __name__ == "__main__":
    #plot_spinor(init_spinor)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot(init_spinor[:2, 0], init_spinor[:2, 1], init_spinor[:2, 2])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ani = FuncAnimation(fig, update, frames=360, fargs=(line,), blit=True, interval=50)
    plt.show()

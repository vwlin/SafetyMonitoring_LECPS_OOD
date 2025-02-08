import matplotlib.pyplot as plt
import numpy as np

WIDTH = 1.5
LENGTH = 20
# vert walls: -0.75, 0.75, 17.75, 19.25
# horz walls: -10, -8.5, 8.5, 10
VERT_WALL_Xs = np.array([-WIDTH/2, WIDTH/2, LENGTH-WIDTH-WIDTH/2, LENGTH-WIDTH/2])
HORZ_WALL_Ys = np.array([-LENGTH/2, -LENGTH/2+WIDTH, LENGTH/2-WIDTH, LENGTH/2])

def plot_halls():

    # left vertical hall
    plt.plot(np.array([-WIDTH/2, -WIDTH/2]), np.array([-LENGTH/2, LENGTH/2]), 'k', linewidth=3)
    plt.plot(np.array([WIDTH/2, WIDTH/2]), np.array([-LENGTH/2+WIDTH, LENGTH/2-WIDTH]), 'k', linewidth=3)

    # right vertical hall
    plt.plot(np.array([LENGTH-WIDTH/2, LENGTH-WIDTH/2]), np.array([-LENGTH/2, LENGTH/2]), 'k', linewidth=3)
    plt.plot(np.array([LENGTH-WIDTH-WIDTH/2, LENGTH-WIDTH-WIDTH/2]), np.array([-LENGTH/2+WIDTH, LENGTH/2-WIDTH]), 'k', linewidth=3)

    # top horizontal hall
    plt.plot(np.array([-WIDTH/2, LENGTH-WIDTH/2]), np.array([LENGTH/2, LENGTH/2]), 'k', linewidth=3)
    plt.plot(np.array([WIDTH/2, LENGTH-WIDTH-WIDTH/2]), np.array([LENGTH/2-WIDTH, LENGTH/2-WIDTH]), 'k', linewidth=3)

    # bottom horizontal hall
    plt.plot(np.array([-WIDTH/2, LENGTH-WIDTH/2]), np.array([-LENGTH/2, -LENGTH/2]), 'k', linewidth=3)
    plt.plot(np.array([WIDTH/2, LENGTH-WIDTH-WIDTH/2]), np.array([-LENGTH/2+WIDTH, -LENGTH/2+WIDTH]), 'k', linewidth=3)

    # # SANITY CHECK
    # # bottom left
    # plt.scatter(-1.5/2, -10, c='r')
    # plt.scatter(1.5/2, -10+1.5, c='r')
    # # top left
    # plt.scatter(-1.5/2, 10, c='r')
    # plt.scatter(1.5/2, 10-1.5, c='r')
    # # top right
    # plt.scatter(20-1.5/2, 10, c='r')
    # plt.scatter(20-4.5/2, 10-1.5, c='r')
    # # bottom right
    # plt.scatter(20-1.5/2, -10, c='r')
    # plt.scatter(20-4.5/2, -10+1.5, c='r')

def plot_halls_with_safety_margin(margin):

    plot_halls()

    # left vertical hall
    plt.plot(np.array([-WIDTH/2 + margin, -WIDTH/2 + margin]), np.array([-LENGTH/2 + margin, LENGTH/2 - margin]), '0.8', linewidth=1)
    plt.plot(np.array([WIDTH/2 - margin, WIDTH/2 - margin]), np.array([-LENGTH/2+WIDTH - margin, LENGTH/2-WIDTH + margin]), '0.8', linewidth=1)

    # right vertical hall
    plt.plot(np.array([LENGTH-WIDTH/2 - margin, LENGTH-WIDTH/2 - margin]), np.array([-LENGTH/2 + margin, LENGTH/2 - margin]), '0.8', linewidth=1)
    plt.plot(np.array([LENGTH-WIDTH-WIDTH/2 + margin, LENGTH-WIDTH-WIDTH/2 + margin]), np.array([-LENGTH/2+WIDTH - margin, LENGTH/2-WIDTH + margin]), '0.8', linewidth=1)

    # top horizontal hall
    plt.plot(np.array([-WIDTH/2 + margin, LENGTH-WIDTH/2 - margin]), np.array([LENGTH/2 - margin, LENGTH/2 - margin]), '0.8', linewidth=1)
    plt.plot(np.array([WIDTH/2 - margin, LENGTH-WIDTH-WIDTH/2 + margin]), np.array([LENGTH/2-WIDTH + margin, LENGTH/2-WIDTH + margin]), '0.8', linewidth=1)

    # bottom horizontal hall
    plt.plot(np.array([-WIDTH/2 + margin, LENGTH-WIDTH/2 - margin]), np.array([-LENGTH/2 + margin, -LENGTH/2 + margin]), '0.8', linewidth=1)
    plt.plot(np.array([WIDTH/2 - margin, LENGTH-WIDTH-WIDTH/2 + margin]), np.array([-LENGTH/2+WIDTH - margin, -LENGTH/2+WIDTH - margin]), '0.8', linewidth=1)
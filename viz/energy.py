"""Script to visualize total energy of positive definite matrices and eval"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np


def generate_spd_matrix(
    theta: float = None, evals: tuple = None, degrees: bool = False
) -> np.ndarray:
    assert len(evals) == 2, "only 2-dim matrices"
    if degrees:
        theta = np.deg2rad(theta)
    e1, e2 = evals
    assert e1 > 0 and e2 > 0, "Eigenvalues must be positive"
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return Q @ np.diag([e1, e2]) @ Q.T


def compute_energy(A: np.ndarray, X, Y) -> np.ndarray:
    """Compute energy of a matrix"""
    # check for symmetry
    if np.allclose(A, A.T):
        a11, a12, a22 = A[0, 0], A[0, 1], A[1, 1]
        Z = a11 * X**2 + 2 * a12 * X * X + a22 * Y**2
    else:
        vect = np.stack([X, Y])
        Z = vect.T @ A @ vect
    return Z


def plot_enegery(axes: tuple, cmap: str = "viridis", alpha=0.8):
    """Plot the energy in a 3d space"""
    X, Y, Z = axes
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha)
    ax.contour(X, Y, Z, zdir="z", offset=0, cmap=cmap)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--e", nargs="+", type=int)

    args = parser.parse_args()
    e_values = tuple(args.e)

    A = generate_spd_matrix(10, e_values, degrees=True)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)

    # we obtain a grid of points in x, y axes
    X, Y = np.meshgrid(x, y)
    Z = compute_energy(A, X, Y)
    plot_enegery((X, Y, Z))

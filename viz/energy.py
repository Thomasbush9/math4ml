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
        Z = a11 * X**2 + 2 * a12 * X * Y + a22 * Y**2
    else:
        vect = np.stack([X, Y])
        Z = vect.T @ A @ vect
    return Z


def plot_energy(
    axes: tuple,
    evecs: np.ndarray,
    evals: np.ndarray,
    gradient: np.ndarray = None,
    *,
    cmap: str = "viridis",
    alpha: float = 0.8,
    c_axis: float | None = None,
) -> None:
    """
    3D surface + contour-at-z=0 + ellipse axes at level c_axis.
    - evecs must be the matrix returned by eigh (columns are eigenvectors).
    - evals must be the eigenvalues from eigh (ascending).
    """
    X, Y, Z = axes
    l1, l2 = float(evals[0]), float(evals[1])
    v1, v2 = evecs[:, 0], evecs[:, 1]  # <-- columns, not rows

    # choose a positive visible level for axis lengths
    if c_axis is None:
        Zpos = Z[Z > 0]
        c_axis = float(np.median(Zpos)) if Zpos.size else float(Z.max() * 0.25)
    assert c_axis > 0, "c_axis must be positive"

    a1 = np.sqrt(c_axis / l1)
    a2 = np.sqrt(c_axis / l2)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha)

    # project contours onto the base plane (z=0) for clarity
    ax.contour(X, Y, Z, zdir="z", offset=0.0, cmap=cmap)

    # draw ellipse axes in the base plane from a single origin
    origin = np.array([0.0, 0.0, 0.0])
    ax.quiver(*origin, a1 * v1[0], a1 * v1[1], 0.0)
    ax.quiver(*origin, -a1 * v1[0], -a1 * v1[1], 0.0)
    ax.quiver(*origin, a2 * v2[0], a2 * v2[1], 0.0)
    ax.quiver(*origin, -a2 * v2[0], -a2 * v2[1], 0.0)
    # plot the gradient on the contour plot
    if gradient is not None:
        skip = (slice(None, None, 5), slice(None, None, 5))  # plot every 5th vector
        ax.quiver(
            X[skip],
            Y[skip],
            np.zeros_like(Z[skip]),
            gradient[0][skip],
            gradient[1][skip],
            np.zeros_like(Z[skip]),
            color="red",
            length=0.5,
            normalize=True,
        )

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--e", nargs="+", type=int)

    args = parser.parse_args()
    e_values = tuple(args.e)

    A = generate_spd_matrix(10, e_values, degrees=True)
    _, evect = np.linalg.eigh(A)

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)

    # we obtain a grid of points in x, y axes
    X, Y = np.meshgrid(x, y)
    Z = compute_energy(A, X, Y)
    # compute the gradient:
    V = np.stack([X, Y]).reshape((2, X.shape[-1] ** 2))
    gradient = (2 * (A @ V)).reshape((2, X.shape[-1], Y.shape[-1]))  # dE/dx = 2Ax
    Gx, Gy = gradient[0], gradient[1]

    plot_energy((X, Y, Z), evect.T, e_values, gradient)

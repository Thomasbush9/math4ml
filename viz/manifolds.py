import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import ndarray


def rank(b: np.ndarray, tol: float = 1e-10) -> int:
    assert np.ndim(b) >= 2, "Array must have two dim"
    assert b.shape[0] == 3, "Array must have shape (3, k)"

    # compute singular values
    theta = np.linalg.svd(b, compute_uv=False)
    theta_max = theta.max() if len(theta) > 0 else 0.0

    rank = np.sum(np.where(theta >= tol * theta_max, 1, 0))
    return rank


def as_basis(A: np.ndarray):
    assert A.shape[0] == 3, "all basis vectors must be 3D"
    assert A.shape[-1] in list(range(4)), "k must be in 0, 1, 2, 3"

    # check independence
    r = rank(A)
    assert r == A.shape[-1], (
        f"Basis vectors are linearly dependent (rank {r} < {A.shape[-1]})"
    )
    return A


def sample_affine(p0: np.ndarray, A: np.ndarray, n=50, span=1.0):
    "Generates points in the affine p=p_0 + alpha*A"
    assert p0.shape == (3,), "p0 must be a 3d point"
    assert A.shape[0] == 3, "A must have shape (3, k)"

    k = A.shape[1]

    if k == 2:
        u = np.linspace(-span, span, n)
        v = np.linspace(-span, span, n)
        uv, vv = np.meshgrid(u, v, indexing="ij")
        uv = uv.ravel()
        vv = vv.ravel()
        grid = np.column_stack((uv, vv))
    elif k == 3:
        u = np.linspace(-span, span, n)
        v = np.linspace(-span, span, n)
        w = np.linspace(-span, span, n)
        uv, vv, wv = np.meshgrid(u, v, w, indexing="ij")
        uv = uv.ravel()
        vv = vv.ravel()
        wv = wv.ravel()
        grid = np.column_stack((uv, vv, wv))
    elif k == 1:
        grid = np.linspace(-span, span, n).reshape((n, 1))
    else:
        return p0

    # compute points:
    points = p0 + grid @ A.T
    return points


def plot_affine(
    p0: np.ndarray,
    A: np.ndarray,
    n: int = 50,
    span: float = 1.0,
    ax=None,
    show_frame=True,
    alpha_surface=0.5,
    subsample=5000,
):
    "Visualize the sampled manifold in 3D"
    P = sample_affine(p0, A, n, span)
    k = A.shape[1]
    if not ax:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    # start plotting:

    if k == 0:
        ax.scatter(*p0)
    elif k == 1:  # line
        t = np.dot((P - p0), A[:, 0]) / np.dot(A[:, 0], A[:, 0])
        order = np.argsort(t)
        P = P[order]
        # plot the line:
        ax.plot(P[:, 0], P[:, 1], P[:, 2])
        ax.scatter(*p0)
    elif k == 2:
        P = P.reshape((n, n, 3))
        X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
        # plotting
        ax.plot_surface(X, Y, Z, alpha=alpha_surface)
        ax.scatter(*p0)

    elif k == 3:
        # subsample the points
        step = max(1, len(P) // subsample)
        sub = P[::step]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=2, alpha=0.5)
        ax.scatter(*p0)
    else:
        raise ValueError(f"k must be 0, 1, 2, or 3 (got {k})")

    return ax

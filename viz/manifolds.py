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

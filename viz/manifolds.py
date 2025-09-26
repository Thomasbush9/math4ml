import matplotlib.pyplot as plt
import numpy as np


def rank(b: np.ndarray, tol: float = 1e-10) -> int:
    assert np.ndim(b) >= 2, "Array must have two dim"
    assert b.shape[0] == 3, "Array must have shape (3, k)"

    # compute singular values
    theta = np.linalg.svd(b, compute_uv=False)
    theta_max = theta.max() if len(theta) > 0 else 0.0

    rank = np.sum(np.where(theta >= tol * theta_max, 1, 0))
    return rank

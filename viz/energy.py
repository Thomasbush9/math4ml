"""Script to visualize total energy of positive definitive matrices and eval"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvals


def generate_psd_matrix(theta: float = None, evals: tuple = None) -> np.ndarray:
    assert len(evals) == 2, "only 2-dim matrices"
    e1, e2 = evals
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return Q.T @ np.diag((e1, e2)) @ Q


if __name__ == "__main__":
    A = generate_psd_matrix(10, (1, 2))
    print(A.shape)

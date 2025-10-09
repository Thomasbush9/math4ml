import numpy as np np


def eigen_decomposition(A:np.ndarray):
    """Compute the eigen decomposition of a square matrix A.
    Args:
        A (np.ndarray): A square matrix.
    Returns:
        tuple: A tuple containing the V, S and V-1 matrices
    """
    assert A.ndim == 2, "Only square matrices"
    assert A.shape[0] == A.shape[1], "only square matrices"

    evals, evect = np.linalg.eig(A)

    V = evect
    S = np.diag(evals)
    return (V, S, np.linalg.inv(V))



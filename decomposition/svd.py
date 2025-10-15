import numpy as np


def svd_vanilla(A: np.ndarray):
    """SVD implementation non optimized"""

    # make A symmetric:
    m, n = A.shape
    if m < n:
        B = A @ A.T
        idx_s = [np.arange(m), np.arange(m)]
    else:
        B = A.T @ A
        idx_s = [np.arange(n), np.arange(n)]

    e_values, e_vect = np.linalg.eigh(B)
    singular_values = np.sqrt(np.where(e_values > 0, e_values, 0))
    # now we need to sort the singular values in non-increasing and get the idx
    idx = np.argsort(singular_values[::-1])
    singular_values = singular_values[idx]
    V = e_vect[:, idx]
    # now we have only to compute U
    U = A @ V / singular_values
    U_partial, _ = np.linalg.qr(U)
    Q, _ = np.linalg.qr(U_partial, mode="complete")
    U = np.hstack((U, Q[:, 10:]))
    S = np.zeros((m, n))
    S[idx_s] = np.diag(singular_values)

    return U, S, V.T


if __name__ == "__main__":
    A = np.random.randn(16, 10)
    U, S, V = svd_vanilla(A)
    print(U.shape, S.shape, V.shape)
    print(np.allclose(A, U @ S @ V))

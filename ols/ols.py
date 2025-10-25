import numpy as np


def pseudo_inverse(A: np.ndarray) -> np.ndarray:
    """Function to generate the pseudo inverse of a matrix using SVD"""
    U, SIGMA, V_t = np.linalg.svd(A)
    return (U.T, 1 / SIGMA, V_t.T)

def minimize_norm(A:np.ndarray, norm:str)->np.ndarray:
    """Function to minimize the given norm of a matrix A"""
    U_t, SIGMA_inv, V = pseudo_inverse(A)
    if norm == 'l2':
        return U_t @ np.diag(SIGMA_inv) @ V
    else:
        raise NotImplementedError(f"Norm '{norm}' not implemented.")

if __name__ == "__main__":
    A = np.random.randn(12, 8)
    U_t, SIGMA_inv, V = pseudo_inverse(A)
    # test it:
    A_pseudo_inv = U_t @ np.diag(SIGMA_inv) @ V
    print(A @ pseudo_inv)

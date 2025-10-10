import numbers

import matplotlib.pyplot as plt
import numpy as np


def eigen_decomposition(A: np.ndarray):
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


def PCA(A: np.ndarray, k: int = 2, plotting: bool = False):
    # diagonalize the data:
    D = A - np.mean(A, axis=0)
    # covariance matrix of features:
    n = D.shape[0]
    C = 1 / (n - 1) * np.dot(D.T, D)
    eig_vals, eig_vect = np.linalg.eigh(C)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vect = eig_vect[:, idx]

    if (
        isinstance(k, numbers.Real)
        and not isinstance(k, (bool, np.bool_))
        and not isinstance(k, (int, np.integer))
    ):
        total_var = eig_vals.sum()
        if total_var <= 0:
            count = 1
        else:
            evr = eig_vals / total_var
            cum = np.cumsum(evr)
            eps = 1e-12
            thr = min(max(float(k), eps), 1.0)  # clamp to (0,1]
            j = int(np.argmax(cum >= thr)) if np.any(cum >= thr) else len(cum) - 1
            count = j + 1
        selected_eig_vals = eig_vals[:count]
        selected_eig_vect = eig_vect[:, :count]
    else:
        # Integer-k path (or numpy integer)
        if not isinstance(k, (int, np.integer)):
            raise TypeError(
                "k must be an int (#components) or float (variance threshold in (0,1])."
            )
        count = int(k)
        if count < 1 or count > eig_vals.shape[0]:
            raise ValueError(f"k must be in [1, {eig_vals.shape[0]}], got {k}.")
        selected_eig_vals = eig_vals[:count]
        selected_eig_vect = eig_vect[:, :count]
    if plotting:
        plt.figure()
        plt.plot(cum)
        plt.axhline(k, color="r", label="90% variance")
        plt.title("Cumulative plot PCA")
        plt.show()

    proj_d = D @ selected_eig_vect
    return selected_eig_vals, selected_eig_vect, proj_d


if __name__ == "__main__":
    # ---- Test 1: Rank-1 staircase matrix ----
    A = np.arange(16).reshape(4, 4)
    vals, vecs, Z = PCA(A, 0.8)  # threshold selection
    # Expect exactly 1 component (rank-1 after centering)
    assert vals.shape == (1,), f"Expected 1 component, got {vals.shape[0]}"
    assert Z.shape == (4, 1), f"Projected shape mismatch: {Z.shape}"
    # Non-zero variance in the projected scores
    assert np.var(Z) > 0, "Projected variance unexpectedly zero for rank-1 data."
    # First PC direction should be uniform (up to sign); check low coefficient variance
    coeff_var = np.var(vecs[:, 0] / (np.mean(np.abs(vecs[:, 0])) + 1e-12))
    assert coeff_var < 0.5, (
        "PC1 not close to a uniform direction (unexpected for this dataset)."
    )

    # ---- Test 2: Threshold selection includes crossing component ----
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 10))
    # Get full eigenvalues to compute manual count
    full_vals, full_vecs, _ = PCA(A, k=10)
    evr = (
        full_vals / np.sum(full_vals)
        if np.sum(full_vals) > 0
        else np.zeros_like(full_vals)
    )
    cum = np.cumsum(evr)
    for thr in [0.5, 0.8, 0.95]:
        j = int(np.argmax(cum >= thr)) if np.any(cum >= thr) else len(cum) - 1
        manual_count = j + 1
        vals_t, vecs_t, Z_t = PCA(A, k=thr)
        assert vals_t.shape[0] == manual_count, (
            f"Threshold {thr}: expected {manual_count} PCs, got {vals_t.shape[0]}."
        )
        assert vecs_t.shape == (A.shape[1], manual_count), (
            "Eigenvector shape mismatch for threshold selection."
        )
        assert Z_t.shape == (A.shape[0], manual_count), (
            "Projection shape mismatch for threshold selection."
        )

    # ---- Test 3: Orthonormality and projected covariance diagonal ----
    A = rng.standard_normal((150, 8))
    vals_k, vecs_k, Z_k = PCA(A, k=5)
    # Orthonormal columns
    VtV = vecs_k.T @ vecs_k
    assert np.allclose(VtV, np.eye(5), atol=1e-6), (
        "Selected eigenvectors are not orthonormal."
    )
    # Covariance of Z equals diag(selected eigenvalues)
    n = A.shape[0]
    CovZ = (Z_k.T @ Z_k) / (n - 1)
    assert np.allclose(CovZ, np.diag(vals_k), atol=1e-5), (
        "Projected covariance not matching selected eigenvalues."
    )

    # ---- Test 4: SVD consistency (eigs of C vs S^2/(n-1)) ----
    A = rng.standard_normal((120, 7))
    D = A - A.mean(axis=0)
    U, S, VT = np.linalg.svd(D, full_matrices=False)
    sv_eigs = (S**2) / (A.shape[0] - 1)
    vals_full, vecs_full, _ = PCA(A, k=A.shape[1])
    # Sort both for comparison
    assert np.allclose(np.sort(vals_full), np.sort(sv_eigs), rtol=1e-5, atol=1e-8), (
        "Eigs(C) != S^2/(n-1)."
    )

    # ---- Test 5: Degenerate data (all rows identical) ----
    A = np.ones((10, 5)) * 3.14
    vals_d, vecs_d, Z_d = PCA(A, 0.9)
    # Expect at least one component; all eigenvalues ~ 0 and projection zeros
    assert vals_d.shape[0] >= 1, (
        "Should return at least one component for degenerate data."
    )
    assert np.allclose(Z_d, 0), "Projection should be zero for zero-variance data."

    # ---- Test 6: Integer-k branch shapes and covariance ----
    A = rng.standard_normal((100, 6))
    for kk in [1, 3, 6]:
        vals_i, vecs_i, Z_i = PCA(A, k=kk)
        assert vals_i.shape == (kk,), f"int-k: eigenvalues shape mismatch for k={kk}"
        assert vecs_i.shape == (A.shape[1], kk), (
            f"int-k: eigenvectors shape mismatch for k={kk}"
        )
        assert Z_i.shape == (A.shape[0], kk), (
            f"int-k: projection shape mismatch for k={kk}"
        )
        CovZi = (Z_i.T @ Z_i) / (A.shape[0] - 1)
        assert np.allclose(CovZi, np.diag(vals_i), atol=1e-5), (
            f"int-k: projected covariance mismatch for k={kk}"
        )

    print("All PCA tests passed ✅")

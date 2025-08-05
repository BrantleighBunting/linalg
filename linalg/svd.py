# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import numpy as np


def svd(A: np.ndarray, tol: float = 1e-12):
    """
        Economy-size Singular Value Decomposition using the Eckart–Young-Mirsky
    construction described in introductory linear-algebra texts.

        For an m-by-n real matrix (m ≥ n) this routine returns three objects:
            U : m-by-n matrix whose columns are orthonormal
            s : length-n vector of singular values, sorted in descending order
            Vt: n-by-n matrix whose rows are orthonormal  (V.T)

        Algorithm outline
        -----------------
        1.  Form A.T @ A, a symmetric n-by-n matrix.
        2.  Solve the symmetric eigenproblem A.T @ A * v = lambda @ v with numpy.linalg.eigh.
            Eigenvectors v become the right-singular vectors; eigenvalues lambda ≥ 0
            give singular values sigma = sqrt(lambda).
        3.  Build the first rank(A) columns of U via u = (1/sigma) A @ v.
            If the matrix is rank-deficient, complete U with any orthonormal
            basis for the remaining subspace so that the final U has exactly
            n orthonormal columns.
        4.  Return U, the singular value vector s, and V.T.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # Handle the wide-matrix case by transposing and swapping the roles
    # of left and right singular vectors.
    if m < n:
        Vt, s, Ut = svd(A.T, tol)
        return Ut.T, s, Vt.T

    # Step-1: build the normal-equations matrix ATA
    ATA = A.T @ A

    # Step-2: eigen-decomposition of the symmetric matrix
    # eigh returns eigenvalues in ascending order, so we flip later
    eigenvalues, V = np.linalg.eigh(ATA)

    # Sort eigenpairs so that singular values come out largest first.
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Singular values are the square roots of the eigenvalues.
    s = np.sqrt(np.clip(eigenvalues, 0.0, None))

    # Numerical rank: how many singular values are clearly non-zero?
    rank = np.sum(s > tol)

    # Step-3: build the first 'rank' left-singular vectors.
    U_cols = []
    for j, sigma in enumerate(s):
        if sigma > tol:
            u = A @ V[:, j] / sigma
            U_cols.append(u)

    # Step-3b: if rank < n, finish U with an orthonormal complement
    if rank < n:
        # Start with random orthonormal columns (QR produces orthonormal Q)
        Q, _ = np.linalg.qr(np.random.randn(m, n - rank))
        # Remove any component lying in the span of the existing U columns
        for u in U_cols:
            Q -= u[:, None] * (u @ Q)
        # Re-orthogonalise to get a clean complement
        Q, _ = np.linalg.qr(Q)
        # Append each new column to the list
        U_cols.extend(Q[:, k] for k in range(n - rank))

    # Pack columns into the final U matrix
    U = np.column_stack(U_cols)

    # Return U, the singular value vector, and V transpose.
    return U, s, V.T

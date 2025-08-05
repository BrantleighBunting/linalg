# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

from typing import Tuple

import numpy as np

from .utils import EPS


def qr(A: np.ndarray, reorth: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified Gram-Schmidt orthogonalization (QR decomposition)
    Parameters:
    A : ndarray
        Full column rank input matrix.
    reorth : ndarray
        Run a second Gram-Schmidt pass to recover orthogonality
    Returns:
    Q : ndarray
        Orthogonal column matrix
    R : ndarray
        Upper-triangular column matrix
    """
    A = A.astype(float, copy=True)
    m, n = A.shape
    Q = np.zeros_like(A)
    R = np.zeros((n, n))

    def _mgs(V):
        for j in range(n):
            v = V[:, j].copy()
            for k in range(j):
                R[k, j] = Q[:, k] @ v
                v -= R[k, j] * Q[:, k]
            R[j, j] = np.linalg.norm(v)
            if R[j, j] < EPS:
                raise ValueError("Input vectors are linearly dependent")
            Q[:, j] = v / R[j, j]
        return Q.copy()

    Q = _mgs(A)
    if reorth:
        Q = _mgs(Q)

    return Q, R


def householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the QR decomposition of an m-by-n matrix A using
    Householder transformations. (m ≥ n)

    A = QR
    H = Q - tau * w * transpose(w)
    tau = 2 / transpose(w) * w'

    Parameters
    ----------
    A : (m, n) ndarray, m >= n

    Returns
    -------
    Q : (m, n) ndarray | orthonormal columns
    R : (n, n) ndarray | upper-triangular
    """
    A = A.astype(float, copy=True)
    m, n = A.shape
    Q = np.eye(max(m, n))
    R = A.copy().astype(float)

    for j in range(n):
        # ---- build the reflector for column j --------------------------------
        x = R[j:, j]
        norm_x = np.linalg.norm(x)
        if norm_x < EPS:  # already zero
            continue
        # w = x + sign(x0) ‖x‖ e₁
        w = x.copy()
        w[0] += np.copysign(norm_x, x[0])
        w /= np.linalg.norm(w)  # ‖w‖ = 1
        w = w.reshape(-1, 1)  # column
        tau = 2  # because w is unit-norm

        # ---- apply H = I – τ w wᵀ  to R (from the left) ----------------------
        R[j:, :] -= tau * w @ (w.T @ R[j:, :])
        # ---- accumulate Q = Q Hᵀ (Hᵀ = H)  -----------------------------------
        Q[:, j:] -= Q[:, j:] @ w @ (tau * w).T

    # economic Q (m × n)
    Q = Q[:, :n]

    # force exact upper-triangular shape / zero tiny noise
    R[np.tril_indices(n, -1)] = 0.0
    # keep only the square part
    R = R[:n, :n]
    return Q, R


def least_squares_qr(A: np.ndarray, b: np.ndarray):
    """
    Solve min ‖Ax – b‖₂ using a thin QR factorisation (A = QR).

    Returns:
    x : (n, ) ndarray
        The least squares solution to Ax = b
    """
    m, n = A.shape
    Q, R = qr(A)
    y = Q.T @ b

    if m > n:
        x_ls = np.linalg.solve(R[:n, :], y[:n])
    else:
        x_ls = np.linalg.solve(R, y)
    return x_ls.ravel()


def least_squares_householder_qr(A: np.ndarray, b: np.ndarray):
    """
    Solve min ‖Ax – b‖₂ using (economic) Householder QR
    decomposition (A = QR). Works for tall or square
    full-rank A.

    Returns:
    x : (n, ) ndarray
        The least squares solution to Ax = b
    """
    Q, R = householder_qr(A)
    y = Q.T @ b
    return np.linalg.solve(R, y)


def random_nonsingular_qr(n, seed=None) -> np.ndarray:
    """
    QR trick (random orthogonal × random non-zero scale)

    QR Decomposition:
        A matrix A can be decomposed into the product of an
        orthogonal matrix Q and an upper triangular matrix
        R (A = QR)

    Returns
    -------
    Matrix with float64 dtype
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q, _R = qr(A)  # Q is orthogonal, det ≠ 0
    scales = rng.uniform(0.5, 10.0, size=n)  # strictly non-zero
    return np.asarray(Q * scales)  # broadcast scales into columns

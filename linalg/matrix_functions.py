# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BUSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import logging

import numpy as np

from .elimination import forward_eliminate
from .qr import qr
from .utils import permutation_sign

logger = logging.getLogger(__name__)


def det(A):
    """
    Calculate the determinant of n-by-n matrix A using elimination
    """
    A = A.astype(float, copy=True)
    m, n = A.shape
    if m != n:
        raise ValueError("The determinant is undefined for non-square matrices.")
    U, _c, _pivots, _free, perm = forward_eliminate(A)
    sign = permutation_sign(perm[:n])
    diag_prod = float(np.prod(np.diag(U)))
    return sign * diag_prod


def rank_numpy(A):
    return np.linalg.matrix_rank(A)


def adj(A: np.ndarray):
    """
    Adjugate (classical adjoint) of a square matrix A.

    Fast path (det ≠ 0): adj(A) = det(A) · A^{-1}
    Slow path (det = 0): cofactor expansion (still O(n³) each det call)
    """
    A = A.astype(float, copy=True)
    m, n = A.shape
    if m != n:
        raise ValueError("A must be a square matrix")

    d = det(A)
    if d == 0:
        logger.warning("adj(): falling back to cofactor expansion – O(n!)")
        # singular matrix, calculate cofactors
        C = np.empty_like(A)

        for i in range(n):
            for j in range(n):
                minor = A[np.arange(n) != i][:, np.arange(n) != j]
                C[i, j] = ((-1) ** (i + j)) * det(minor)
        return C.T

    # If A is nonsingular, use QR
    Q, R = qr(A)
    ain = np.linalg.solve(R, Q.T)
    return d * ain

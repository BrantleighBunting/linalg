#!/usr/bin/python3
# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

"""
Projection operations
"""

import numpy as np


def project_onto_colspace(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Find p = A x, the orthogonal projection of b onto
    the column-space of A.
    Returns
    -------
    p : ndarray, shape (m, k) if b is (m,k) or (m,)

    TODO (Phase-4, step 5):
        Replace normal-equations path with QR once householder_qr is ready,
        e.g. Q, _ = householder_qr(A);  return Q @ (Q.T @ b)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # If we are 1D, make this a column matrix
    if b.ndim == 1:
        b = b[:, None]

    r = np.linalg.matrix_rank(A)
    if r < A.shape[1]:
        print("The columns of A are not independent, falling back to pseudo-inverse")
        return A @ (np.linalg.pinv(A) @ b)
    aT = np.transpose(A)
    ata = aT @ A
    # TODO: use householder QR here
    x = np.linalg.solve(ata, aT @ b)
    p = A @ x

    # using np.inv is safe here because ata is square
    # P = (A @ (np.linalg.inv(ata) @ aT))
    # p =  P @ b

    return p

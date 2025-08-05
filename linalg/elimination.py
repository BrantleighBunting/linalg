# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import logging
from typing import List, Optional, Tuple

import numpy as np

from .utils import scale_tol

logger = logging.getLogger(__name__)


def forward_eliminate(
    A: np.ndarray,
    b: Optional[np.ndarray] = None,
    pivot: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int], List[int], List[int]]:
    """
    Row-echelon reduction with partial pivoting on an m by n matrix A.

    Parameters
    ----------
    A : np.ndarray               (m, n)
        Coefficient matrix (MUST be ndarray).
    b : np.ndarray | None        (m,) or (m, k)
        Optional right-hand side; same row swaps & updates applied.
    pivot : bool
        If False, no row swaps are performed (rarely useful).

    Returns
    -------
    U      : np.ndarray          (m, n)
        Row-echelon form of A (upper-trapezoidal, not reduced).
    c      : np.ndarray | None
        b after identical row ops (None if b was None).
    pivots : list[int]
        Column indices where pivots were placed; len = rank(A).
    free : list[int]
        Column indices where free variables were placed
    perm   : list[int]
        Final row order: row i of U comes from original row perm[i].
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a NumPy ndarray")
    if b is not None and not isinstance(b, np.ndarray):
        raise TypeError("b must be a NumPy ndarray or None")

    U = A.astype(float, copy=True)
    m, n = U.shape

    if b is not None:
        c = np.atleast_2d(b.astype(float)).T if b.ndim == 1 else b.astype(float)
    else:
        c = None

    pivot_tol = scale_tol(U)

    perm = list(range(max(m, n)))  # Identity Permutation
    pivots: List[int] = []
    free: List[int] = []

    row = 0
    for col in range(n):
        if row == m:
            free.extend(range(col, n))
            break
        # The computation we perform will be more stable if we
        # pick the largest possible number for the pivot column.
        # This is done by taking the largest absolute value of
        # each column member, k and below, and choosing the
        # maximum of those values.
        col_slice = np.abs(U[row:, col])
        max_idx = int(col_slice.argmax())
        max_val = col_slice[max_idx]

        if max_val <= pivot_tol:  # column is numerically zero
            free.append(col)
            continue  # go to next column

        pivot_row = row + max_idx

        # If the pivot row is not our current row, record
        # the permutation, the same operation must be applied
        # to b as well
        if pivot and pivot_row != row:
            U[[row, pivot_row]] = U[[pivot_row, row]]
            if c is not None:
                c[[row, pivot_row]] = c[[pivot_row, row]]
            perm[row], perm[pivot_row] = perm[pivot_row], perm[row]

        pivots.append(col)

        # Eliminate entries below the pivot
        # Loop through rows beneath current pivot
        factors = U[row + 1 :, col] / U[row, col]
        U[row + 1 :, col:] -= factors[:, None] * U[row, col:]
        if c is not None:
            c[row + 1 :, :] -= factors[:, None] * c[row, :]

        row += 1  # move to next pivot row

    return U, c, pivots, free, perm


def back_substitute(U: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    U : (n, n) ndarray
        Upper-triangular matrix (output of forward_eliminate,
        possibly with tiny non-zero super-diagonal entries).
    c : (n,) or (n,k) ndarray
        RHS after identical row operations.
    Returns
    -------
    x : (n,) or (n,k) ndarray
        Solution(s) of Ux = c.
    Raises
    ------
    ValueError : if the system is inconsistent or rank-deficient.
    """
    U = np.asarray(U, dtype=float)
    c = np.asarray(c, dtype=float)

    if c.ndim == 1:
        # (n,)  →  (n,1)
        c = c[:, None]
    n, k = c.shape
    x = np.zeros((n, k), dtype=float)
    tol = scale_tol(U)

    for i in reversed(range(n)):
        pivot = U[i, i]
        if abs(pivot) <= tol:
            if np.any(np.abs(c[i]) > tol):
                raise ValueError("inconsistent system (no solution)")
            else:
                raise ValueError("rank deficient (infinitely many solutions)")

        s = c[i] - U[i, i + 1 :] @ x[i + 1 :]
        x[i] = s / pivot

    # flatten if k == 1, regardless of ndim
    if x.shape[1] == 1:
        # (n,1)  →  (n,)
        return x.ravel()
    return x


def gaussian_solve(A: np.ndarray, b: np.ndarray, pivot=True):
    try:
        U, c, pivots, free, perm = forward_eliminate(A, b, pivot=pivot)
        return back_substitute(U, c)
    except ValueError as e:
        if "inconsistent" in str(e):
            # No solution
            raise
        logger.debug(
            f"{e}; Matrix is rank deficient but consistent, falling back to least squares..."
        )
        # rank-deficient but consistent  → least-squares
        return np.linalg.lstsq(A, b, rcond=None)[0]


def rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Return the reduced row-echelon form R of A and the
    pivot column list. R has the same shape as A.

    Parameters
    ----------
    A   : (m,n) ndarray

    Returns
    -------
    R       : (m,n) ndarray  (RREF)
    pivots  : list[int]      pivot column indices
    """
    # Perform forward elimination, U = R
    U, _c, pivots, _free, _perm = forward_eliminate(A, pivot=True)

    R = U.copy()
    tol = scale_tol(R)

    # backward sweep: one pass per pivot, from bottom to top
    for r, col in reversed(list(enumerate(pivots))):
        piv_val = R[r, col]
        if abs(piv_val) > tol:
            R[r] /= piv_val  # scale pivot row → 1

        # zero out entries above the pivot
        for i in range(r):
            factor = R[i, col]
            if abs(factor) > tol:
                R[i] -= factor * R[r]

    # zero out tiny noise
    R[np.abs(R) < tol] = 0.0
    return R, pivots


def rank_elimination(A):
    """Matrix rank is the number of pivot columns"""
    pivots = forward_eliminate(A)[2]
    return len(pivots)


def nullspace_basis_elimination(A):
    """
    Constructs a matrix N whose columns form a basis of the nullspace of A

    Returns
    -------
    N : (n, n-r) ndarray
        Columns form a basis of N(A). If A is full rank (r = n) the returned
        array has shape (n, 0).
    """
    U, c, pivots, free, perm = forward_eliminate(A)
    m, n = A.shape
    r = len(pivots)
    if not free:
        # If we have full rank, only Z = {zero vector} is the nullspace
        return np.zeros((n, 0))

    R = U[:r, pivots]
    N = np.zeros((n, len(free)))

    # construct one basis vector per free column
    for k, j in enumerate(free):
        z = np.zeros(n)
        z[j] = 1.0
        rhs = -U[:r, j]
        x_piv = np.zeros(r)

        # back-substitute through R
        for i in reversed(range(r)):
            x_piv[i] = (rhs[i] - R[i, i + 1 :] @ x_piv[i + 1 :]) / R[i, i]

        z[pivots] = x_piv
        N[:, k] = z

    return N

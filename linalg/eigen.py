# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

from typing import Optional

import numpy as np


def power_iteration(
    A: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-10,
    v0: Optional[np.ndarray] = None,
    return_history: bool = False,
):
    """
    Estimate the dominant eigenvalue (by magnitude) and its eigenvector
    using the Power Iteration method.

    Stops when the residual norm ||A v - λ v||_2 falls below `tol`
    or when `max_iter` is reached.

    Parameters
    ----------
    A : (n,n) ndarray
        Real square matrix.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the residual.
    v0 : (n,) ndarray or None
        Optional initial guess. If None, random normal is used.
    return_history : bool
        If True, also return (num_iters, residual_history).

    Returns
    -------
    lam : float
        Estimated dominant eigenvalue.
    v : (n,) ndarray
        Corresponding eigenvector (unit norm).
    (iters, hist) : optional
        Iteration count and residual array if return_history=True.
    """
    A = np.asarray(A, float)
    m, n = A.shape
    if m != n:
        raise ValueError("Power iteration requires a square matrix.")

    # init vector
    if v0 is None:
        v = np.random.randn(n)
    else:
        v = np.asarray(v0, float).copy()
        if v.shape != (n,):
            raise ValueError("v0 must be shape (n,).")
    v /= np.linalg.norm(v)  # normalize

    lam = 0.0
    iters = 0
    hist = []
    for iters in range(max_iter):
        w = A @ v
        norm_w = np.linalg.norm(w)
        if norm_w < tol:
            # A maps current v to ~0; matrix may be singular.
            lam = 0.0
            break
        v = w / norm_w
        lam_new = v @ (A @ v)  # Rayleigh quotient
        resid = np.linalg.norm(A @ v - lam_new * v)
        hist.append(resid)
        lam = lam_new
        if resid < tol:
            break
    return (lam, v, iters, np.array(hist)) if return_history else (lam, v)


def matrix_power_eig(
    A: np.ndarray, k: int, *, tol=1e-10, cond_thresh=1e12
) -> np.ndarray:
    """
    Compute A^k using eigendecomposition when possible, else fallback to numpy.

    Parameters
    ----------
    A : (n,n) ndarray
        Square matrix.
    k : int
        Integer power (can be negative or zero).
    tol : float
        Tolerance for "imaginary part is zero" checks.
    cond_thresh : float
        If cond(V) > cond_thresh, fall back to np.linalg.matrix_power.

    Returns
    -------
    Ak : (n,n) ndarray
        A raised to the k-th power.
    """
    A = np.asarray(A)
    n, m = A.shape
    if n != m:
        raise ValueError("matrix_power_eig only defined for square matrices.")
    if k == 0:
        return np.eye(n, dtype=A.dtype)
    if k < 0:
        # invert first (will raise if singular)
        A_inv = np.linalg.inv(A)
        return matrix_power_eig(A_inv, -k, tol=tol, cond_thresh=cond_thresh)

    # Try eigendecomposition
    eigvals, V = np.linalg.eig(A)
    # Check conditioning
    try:
        condV = np.linalg.cond(V)
    except np.linalg.LinAlgError:  # V may be singular
        return np.linalg.matrix_power(A, k)

    if not np.isfinite(condV) or condV > cond_thresh:
        # Poorly conditioned eigenvectors -> fallback
        return np.linalg.matrix_power(A, k)

    # Raise eigenvalues to power
    Dk = np.diag(eigvals**k)

    # Solve instead of invert
    # Solve V * X = I -> X = inv(V)
    X = np.linalg.solve(V, np.eye(n, dtype=A.dtype))
    Ak = V @ Dk @ X

    # If original A was real and imag is tiny, drop it
    if np.isrealobj(A) and np.max(np.abs(Ak.imag)) < tol:
        Ak = Ak.real

    return Ak

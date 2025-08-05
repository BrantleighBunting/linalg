# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import numpy as np

EPS: float = 1e-12


def scale_tol(A: np.ndarray) -> float:
    """Return an absolute tolerance scaled to the matrix magnitude."""
    return EPS * max(1.0, np.linalg.norm(A, ord=np.inf))


def permutation_sign(perm: list[int]) -> float:
    """Return +1 or –1 depending on permutation parity."""
    visited = [False] * len(perm)
    cycles = 0
    for i in range(len(perm)):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
    swaps = len(perm) - cycles  # n − #cycles
    return -1.0 if swaps & 1 else 1.0


def random_nonsingular_upper(n, low=-100, high=100, seed=None) -> np.ndarray:
    """
    Build a matrix U that is upper-triangular with random entries
    everywhere and put only non-zero values on its diagonal

    Returns
    -------
    Matrix with float64 dtype
    """
    rng = np.random.default_rng(seed)
    U = rng.uniform(low, high, size=(n, n))
    # enforce upper-triangular
    U = np.triu(U)
    # replace any accidental zeros on the diagonal
    diag = rng.uniform(low if low != 0 else 1, high, size=n)
    U[np.diag_indices(n)] = diag
    return np.asarray(U)

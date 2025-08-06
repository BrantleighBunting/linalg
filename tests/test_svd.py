# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BUSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import numpy as np
import pytest

from linalg.svd import svd


@pytest.mark.parametrize("m,n", [(8, 5), (20, 20), (50, 10)])
def test_reconstruction_and_orthogonality(m, n):
    """U Σ Vᵀ must reconstruct A and U, V must be orthonormal."""
    rng = np.random.default_rng(seed=m + n)
    A = rng.normal(size=(m, n))

    U, s, Vt = svd(A)
    Σ = np.diag(s)

    # 1  Reconstruction ‖A - UΣVᵀ‖
    recon_err = np.linalg.norm(U @ Σ @ Vt - A, ord=2)
    assert recon_err < 1e-10

    # 2  Orthonormality
    assert np.allclose(U.T @ U, np.eye(n), atol=1e-10)
    assert np.allclose(Vt @ Vt.T, np.eye(n), atol=1e-10)


def _align_signs(X, Y):
    """Flip columns of X so that X[:,i] · Y[:,i] ≥ 0 (helps compare eigendirections)."""
    sign = np.sign(np.sum(X * Y, axis=0))
    sign[sign == 0] = 1.0  # avoid zeros
    return X * sign


@pytest.mark.parametrize("m,n", [(12, 7), (30, 15)])
def test_against_numpy_svd(m, n):
    """Singular values must match NumPy’s; left/right spaces must match up to sign."""
    rng = np.random.default_rng(seed=4 * m + n)
    A = rng.standard_normal(size=(m, n))

    # NumPy “truth”
    U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    # Strang implementation
    U_my, s_my, Vt_my = svd(A)

    # 1  Singular values (sorted descending)
    assert np.allclose(s_my, s_np, rtol=1e-10, atol=1e-12)

    # 2  Column spaces (sign ambiguity only)
    U_my_aligned = _align_signs(U_my, U_np)
    Vt_my_aligned = _align_signs(Vt_my.T, Vt_np.T).T  # align rows → take T
    assert np.allclose(U_my_aligned, U_np, atol=1e-8)
    assert np.allclose(Vt_my_aligned, Vt_np, atol=1e-8)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_rank_deficient(k):
    """
    Rank-deficient matrices: make last k cols zero, Strang code
    must still return (r, …) SVD where σ_{r:} ≈ 0.
    """
    rng = np.random.default_rng(123 + k)
    A = rng.normal(size=(10, 7))
    if k:
        A[:, -k:] = 0.0

    U, s, Vt = svd(A)
    Σ = np.diag(s)
    err = np.linalg.norm(U @ Σ @ Vt - A)
    assert err < 1e-10

    # Check trailing singular values ~ 0
    r = 7 - k
    assert np.all(s[:r] > 1e-12)
    assert np.all(s[r:] < 1e-12)

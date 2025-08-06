# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BUSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import numpy as np
import pytest

from linalg.eigen import matrix_power_eig, power_iteration


def test_matrix_power_matches_numpy_random():
    rng = np.random.default_rng(0)
    for _ in range(10):
        A = rng.normal(size=(5, 5))
        for k in [0, 1, 2, 5, -1, -3]:
            Ak_eig = matrix_power_eig(A, k)
            Ak_np = np.linalg.matrix_power(A, k)
            np.testing.assert_allclose(Ak_eig, Ak_np, rtol=1e-8, atol=1e-10)


def test_matrix_power_defective_warns_or_fallbacks():
    # Jordan-like (defective) example
    A = np.array([[1, 1], [0, 1]], dtype=float)
    Ak_eig = matrix_power_eig(A, 5)
    Ak_np = np.linalg.matrix_power(A, 5)
    np.testing.assert_allclose(Ak_eig, Ak_np, rtol=1e-8, atol=1e-10)


def test_matrix_power_complex_eigs_back_to_real():
    # Rotation matrix (complex eigenvalues)
    theta = 0.3
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Ak_eig = matrix_power_eig(A, 7)
    Ak_np = np.linalg.matrix_power(A, 7)
    np.testing.assert_allclose(Ak_eig, Ak_np, rtol=1e-8, atol=1e-10)


def test_power_iteration_sym_psd():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(40, 40))
    A = M.T @ M  # symmetric PSD
    lam, v = power_iteration(A, tol=1e-12, max_iter=5000)
    rq = v @ (A @ v)
    assert np.isclose(lam, rq, atol=1e-10)
    resid = np.linalg.norm(A @ v - lam * v)
    assert resid < 1e-8


def test_power_iteration_non_square_raises():
    A = np.random.randn(3, 4)
    with pytest.raises(ValueError):
        # if your power_iteration does not raise, adapt this
        _ = power_iteration(A)


def test_power_iteration_scaling():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 30))
    alpha = 7.3
    v0 = rng.normal(size=(30,))  # SAME start for both
    lam1, v1 = power_iteration(A, v0=v0)
    lam2, v2 = power_iteration(alpha * A, v0=v0)

    sign = np.sign(v1 @ v2) or 1.0
    assert np.allclose(sign * v2, v1, atol=1e-6)
    assert np.isclose(lam2, alpha * lam1, rtol=1e-6, atol=1e-8)


def test_power_iteration_random():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(50, 50))
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigvals))
    lam_true = eigvals[idx]
    v_true = eigvecs[:, idx]

    lam_est, v_est = power_iteration(A, max_iter=2000, tol=1e-12)

    # fix phase (sign/complex phase); we assume real here, so just sign:
    sign = np.sign(v_true @ v_est) or 1.0
    v_est_aligned = sign * v_est

    # check eigenvalue estimate
    assert np.isclose(lam_est, lam_true, rtol=1e-6, atol=1e-8)
    # check vector (direction only)
    assert np.allclose(v_est_aligned, v_true, atol=1e-5)
    # residual small
    resid = np.linalg.norm(A @ v_est - lam_est * v_est)
    assert resid < 1e-8


def test_power_iteration_diagonal():
    A = np.diag([5.0, 2.0, -1.0])
    lam, v = power_iteration(A, max_iter=1000, tol=1e-12)
    # true dominant eigenvalue magnitude is 5, sign is +5
    assert np.isclose(lam, 5.0, atol=1e-9)
    # eigenvector should align with e1 (up to sign)
    e1 = np.array([1.0, 0.0, 0.0])
    assert np.allclose(np.abs(v), e1, atol=1e-6)

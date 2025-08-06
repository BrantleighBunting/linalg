# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BUSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import logging

import numpy as np

from linalg.elimination import (
    back_substitute,
    forward_eliminate,
    gaussian_solve,
    nullspace_basis_elimination,
    rank_elimination,
    rref,
)
from linalg.qr import random_nonsingular_qr
from linalg.utils import EPS, random_nonsingular_upper

TEST_ITERATIONS = 50
logger = logging.getLogger(__name__)


def test_forward_eliminate_basic_n_by_n():
    n = 200
    A = np.random.randn(n, n)
    x0 = np.random.randn(n)
    b = A @ x0

    U, c, pivots, free, perm = forward_eliminate(A, b)
    x = back_substitute(U, c)
    assert np.allclose(x, x0, rtol=1e-8, atol=EPS)


def test_elimination_random_nonsingular_upper_triangular():
    n = TEST_ITERATIONS

    for i in range(n):
        logger.debug("==============================")
        while True:
            A = random_nonsingular_upper(n)
            if np.linalg.matrix_rank(A) == n:
                break
        logger.debug(f"\nRunning Test\nRandom Nonsingular Upper:\n{A}\n")

        # Generate a random vector x (the true solution)
        x_true = np.random.rand(n)

        # Calculate b using the equation Ax = b
        b = np.dot(A, x_true)

        # Solve the system of equations using numpy.linalg.solve
        x_calculated = np.linalg.solve(A, b)
        u_calculated = gaussian_solve(A, b)
        logger.debug(
            f"\n==== Results ====\nOurs:\n{u_calculated}\nNumpy:\n{x_calculated}"
        )

        # Compare the residual (r = b - Ax), this judges numerical
        # correctness in a way that is independent of conditioning
        # and unknown scaling, then take the inifinty norm
        # NOTE: least squares makes this as small as algebraically possible
        res_np = np.linalg.norm(A @ x_calculated - b, ord=np.inf)
        res_lu = np.linalg.norm(A @ u_calculated - b, ord=np.inf)
        np.testing.assert_allclose(res_lu, res_np, rtol=1e-10, atol=EPS)
        logger.debug("==============================")


def test_elimination_random_nonsingular_qr():
    n = TEST_ITERATIONS

    for i in range(n):
        logger.debug("==============================")
        while True:
            A = random_nonsingular_qr(n)
            if np.linalg.matrix_rank(A) == n:
                break

        logger.debug(f"\nRunning Test\n{A}\n")

        # Generate a random vector x (the true solution)
        x_true = np.random.rand(n)

        # Calculate b using the equation Ax = b
        b = np.dot(A, x_true)

        # Solve the system of equations using numpy.linalg.solve
        x_calculated = np.linalg.solve(A, b)
        u_calculated = gaussian_solve(A, b)

        logger.debug(
            f"\n==== Results ====\nOurs:\n{u_calculated}\nNumpy:\n{x_calculated}"
        )
        np.testing.assert_allclose(
            x_calculated,
            u_calculated,
            rtol=5e-8,  # relative tolerance
            atol=EPS,  # absolute tolerance
            verbose=True,
        )
        logger.debug("==============================")


def test_nullspace_basis_elimination():
    n = TEST_ITERATIONS
    A = random_nonsingular_qr(n)
    logger.debug(f"\nRunning Test\n{A}\n")

    A = np.random.randn(6, 10)  # rank ≤ 6
    N = nullspace_basis_elimination(A)
    assert np.allclose(A @ N, 0, atol=1e-10)
    # Assert that we are satisfying the rank nullity theorem
    assert N.shape[1] == A.shape[1] - np.linalg.matrix_rank(A)


def test_rref_idempotent():
    m, n = 6, 8
    A = np.random.randn(m, n)
    R1, piv = rref(A)
    logger.debug(f"RREF\n{R1}\n")
    R2, _ = rref(R1)  # RREF of an RREF is itself
    assert np.allclose(R1, R2, atol=1e-10)


def test_rref_pivot_structure():
    A = np.random.randn(5, 7)
    R, pivots = rref(A)
    # each pivot column should be e_i
    for r, c in enumerate(pivots):
        ei = np.zeros_like(R[:, c])
        ei[r] = 1
        assert np.allclose(R[:, c], ei, atol=1e-10)


def test_rank_agreement():
    for _ in range(100):
        A = np.random.randn(8, 6)
        assert rank_elimination(A) == np.linalg.matrix_rank(A, tol=EPS)

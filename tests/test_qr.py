# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import logging

import numpy as np

from linalg.qr import (
    householder_qr,
    least_squares_householder_qr,
    least_squares_qr,
    qr,
)
from linalg.utils import random_nonsingular_upper

TEST_ITERATIONS = 50
logger = logging.getLogger(__name__)


def test_least_squares_qr():
    n = TEST_ITERATIONS

    for i in range(n):
        logger.debug("==============================")
        while True:
            A = random_nonsingular_upper(n)
            if np.linalg.matrix_rank(A) == n:
                break

        # Generate a random vector x (the true solution)
        x_true = np.random.rand(n)

        # Calculate b using the equation Ax = b
        b = np.dot(A, x_true)

        x_np, *_ = np.linalg.lstsq(A, b, rcond=None)
        x_ours = least_squares_qr(A, b)
        x_householder = least_squares_householder_qr(A, b)

        res_np = np.linalg.norm(A @ x_np - b, ord=np.inf)
        res_ours = np.linalg.norm(A @ x_ours - b, ord=np.inf)
        res_householder = np.linalg.norm(A @ x_householder - b, ord=np.inf)
        assert res_ours <= res_np * (1 + 1e-8)
        assert res_householder <= res_np * (1 + 1e-8)


def test_orthogonality_qr():
    V = np.random.randn(100, 10)
    Q, _ = qr(V, reorth=True)
    identity = Q.T @ Q
    assert np.allclose(identity, np.eye(10), atol=1e-10)


def test_orthogonality_householder_qr():
    V = np.random.randn(100, 10)
    Q, _ = householder_qr(V)
    identity = Q.T @ Q
    assert np.allclose(identity, np.eye(10), atol=1e-10)

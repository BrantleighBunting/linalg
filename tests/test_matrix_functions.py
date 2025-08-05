# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import math

import numpy as np

from linalg.matrix_functions import adj, det


def test_determinants():
    A = np.random.randn(100, 100)
    our_det = det(A)
    numpy_det = np.linalg.det(A)
    assert math.isclose(our_det, numpy_det, abs_tol=1e-8)


def test_adjugate():
    A = np.random.randn(10, 10)

    our_adj = adj(A)
    numpy_adj = np.linalg.det(A) * np.linalg.inv(A)
    print(our_adj)
    print(numpy_adj)
    assert np.allclose(our_adj, numpy_adj, atol=1e-8)

# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import numpy as np

from linalg.projections import project_onto_colspace


def test_projections():
    A = np.array(
        [
            [1, 0],
            [1, 1],
            [1, 2],
        ]
    )
    b = np.array(
        [
            [6],
            [0],
            [0],
        ]
    )

    p = project_onto_colspace(A, b)
    np.testing.assert_allclose(
        p,
        np.array(
            [
                [5],
                [2],
                [-1],
            ]
        ),
        verbose=True,
    )

    # Check residuals
    res = np.linalg.norm(A @ np.linalg.lstsq(A, b, rcond=None)[0] - b, np.inf)
    res_proj = np.linalg.norm(p - b, np.inf)
    assert abs(res - res_proj) < 1e-12

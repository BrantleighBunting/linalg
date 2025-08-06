# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BUSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

"""
linalg
======

A small, educational linear–algebra toolkit that grows step-by-step
alongside a transformer-from-scratch project.

Public API
~~~~~~~~~~
- Decompositions
    - `qr`, `householder_qr`
    - `svd`
- Matrix utilities
    - `det`, `adj`, `matrix_power_eig`
- Linear systems
    - `gaussian_solve`, `least_squares_qr`,
      `least_squares_householder_qr`
- Iterative methods
    - `power_iteration`
- Rank / null-space tools
    - `rank_elimination`, `nullspace_basis_elimination`

Everything else lives in sub-modules and is **not** considered part of the
stable interface.

Example
-------
>>> import numpy as np, linalg as la
>>> A = np.random.randn(5, 3)
>>> Q, R = la.qr(A)
>>> np.allclose(Q @ R, A)
True
"""

from importlib.metadata import version as _pkg_version

from .eigen import (
    matrix_power_eig,
    power_iteration,
)
from .elimination import (
    back_substitute,
    forward_eliminate,
    gaussian_solve,
    nullspace_basis_elimination,
    rank_elimination,
)
from .matrix_functions import (
    adj,
    det,
    rank_numpy,
)
from .projections import project_onto_colspace

# ---------------------------------------------------------------------
# Re-export the high-level functions users are expected to call.
# Each of these names is implemented in one of the internal sub-modules.
# ---------------------------------------------------------------------
from .qr import (
    householder_qr,
    least_squares_householder_qr,
    least_squares_qr,
    qr,
    random_nonsingular_qr,
)
from .svd import svd
from .utils import permutation_sign, random_nonsingular_upper, scale_tol

__all__ = [
    "qr",
    "householder_qr",
    "least_squares_qr",
    "least_squares_householder_qr",
    "random_nonsingular_qr",
    "power_iteration",
    "matrix_power_eig",
    "forward_eliminate",
    "back_substitute",
    "gaussian_solve",
    "rank_elimination",
    "nullspace_basis_elimination",
    "det",
    "rank_numpy",
    "adj",
    "project_onto_colspace",
    "svd",
    "scale_tol",
    "permutation_sign",
    "random_nonsingular_upper",
]

# ---------------------------------------------------------------------
# Version string (helps “pip show linalg”, Sphinx, etc.)
# ---------------------------------------------------------------------
try:  # installed via pip / build backend
    __version__ = _pkg_version(__name__)
except Exception:  # running from a checkout
    __version__ = "0.0.0.dev0"

# ---------------------------------------------------------------------
# Optional: lightweight default logging config so users see warnings
# only if they deliberately enable them.
# ---------------------------------------------------------------------
import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

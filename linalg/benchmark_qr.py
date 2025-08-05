#!/usr/bin/python3
# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import json
import platform
import time

import numpy as np
import pandas as pd
from row_reduction import (
    back_substitute,
    forward_eliminate,
    householder_qr,
    least_squares_householder_qr,
    least_squares_qr,
    qr,
)

np.random.seed(0)
REPEATS = 5  # median of 5 runs leads to stable numbers
sizes = [(300, 300), (1000, 1000), (5000, 1000)]


def wall(f, *args, **kwargs):
    t0 = time.perf_counter()
    f(*args, **kwargs)
    return time.perf_counter() - t0


records = []
for m, n in sizes:
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # reference
    t_np = min(wall(np.linalg.lstsq, A, b, rcond=None) for _ in range(REPEATS))
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    r_ref = np.linalg.norm(A @ x_ref - b, np.inf)

    # benchmark Gaussian solver (square only)
    if m == n:
        t_gauss = min(
            wall(lambda: back_substitute(*forward_eliminate(A, b)[:2]))
            for _ in range(REPEATS)
        )
        x_gauss = back_substitute(*forward_eliminate(A, b)[:2])
        r_gauss = np.linalg.norm(A @ x_gauss - b, np.inf)
        records.append(("GE", f"{m}x{n}", t_gauss, t_gauss / t_np, r_gauss / r_ref))

    # Modified Gram-Schmidt QR
    t_mgs = min(wall(qr, A) for _ in range(REPEATS))
    Q, R = qr(A)
    ortho = np.linalg.norm(Q.T @ Q - np.eye(n), np.inf)
    x_mgs = least_squares_qr(A, b)
    r_mgs = np.linalg.norm(A @ x_mgs - b, np.inf)
    records.append(("MGS-QR", f"{m}×{n}", t_mgs, t_mgs / t_np, r_mgs / r_ref, ortho))

    # ---------- Householder QR ---------------------------------
    t_hh = min(wall(householder_qr, A) for _ in range(REPEATS))
    Qh, Rh = householder_qr(A)
    ortho2 = np.linalg.norm(Qh.T @ Qh - np.eye(n), np.inf)
    x_hh = least_squares_householder_qr(A, b)
    r_hh = np.linalg.norm(A @ x_hh - b, np.inf)
    records.append(("HH-QR", f"{m}×{n}", t_hh, t_hh / t_np, r_hh / r_ref, ortho2))

df = pd.DataFrame(
    records,
    columns=["kernel", "size", "sec", "sec/NumPy", "residual/NumPy", "orth_err"],
)
print(df.to_markdown(index=False))

df.to_csv("bench_results.csv", index=False)

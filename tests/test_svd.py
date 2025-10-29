# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BUSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

import numpy as np
import pytest

from linalg.svd import svd, pca


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


# ----- PCA Tests -----


def make_lowrank_data(n=200, d=10, r=3, noise=0.05, seed=0):
    """
    Generate approximately rank-r data with descending latent variances.
    Shapes: X is (n_samples=n, n_features=d).
    """
    assert 1 <= r <= min(n, d), "r must be between 1 and min(n, d)"
    rng = np.random.default_rng(seed)

    # latent factors (n×r) with descending std per component
    Z = rng.normal(size=(n, r))
    scales = np.geomspace(3.0, 0.3, r)  # length r, strictly decreasing
    Z *= scales[None, :]  # broadcast scale each latent dim

    # feature loadings (d×r)
    W = rng.normal(size=(d, r))

    # construct X and add small noise
    X = Z @ W.T + noise * rng.normal(size=(n, d))
    return X


def test_shapes_and_types():
    X = make_lowrank_data(n=50, d=8, r=3, seed=1)
    k = 3
    pcs, scores, ev, evr, total_var, mean_ = pca(X, k)

    assert pcs.shape == (X.shape[1], k)
    assert scores.shape == (X.shape[0], k)
    assert ev.shape == (k,)
    assert evr.shape == (k,)
    assert mean_.shape == (X.shape[1],)


def test_centering_zero_mean():
    X = make_lowrank_data(n=100, d=12, r=4, seed=2)
    k = 4
    pcs, scores, ev, evr, total_var, mean_ = pca(X, k)

    # Verify the input was centered along features used for projection
    Xc = X - mean_
    col_means = Xc.mean(axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-12)


def test_pcs_columns_orthonormal():
    X = make_lowrank_data(n=120, d=9, r=3, seed=3)
    k = 5
    pcs, *_ = pca(X, k)
    I = pcs.T @ pcs
    assert np.allclose(I, np.eye(k), atol=1e-10)


def test_reconstruction_error_small():
    X = make_lowrank_data(n=150, d=10, r=3, noise=0.01, seed=4)
    k = 3
    pcs, scores, *_, mean_ = pca(X, k)
    Xc = X - mean_
    X_hat = scores @ pcs.T
    # With truly low-rank data, reconstruction from top-k should be very close
    rel_err = np.linalg.norm(Xc - X_hat, ord="fro") / np.linalg.norm(Xc, ord="fro")
    assert rel_err < 0.05  # generous but meaningful threshold


def test_variance_accounting_total_equals_sum_when_full_rank():
    X = make_lowrank_data(n=80, d=7, r=7, noise=0.0, seed=5)  # exactly rank 7 (w.h.p.)
    k = min(X.shape)
    pcs, scores, ev, evr, total_var, mean_ = pca(X, k)
    assert np.isclose(ev.sum(), total_var, rtol=1e-10, atol=1e-12)


def test_scores_match_U_times_S():
    X = make_lowrank_data(n=60, d=11, r=4, seed=6)
    k = 4
    pcs, scores, *_, mean_ = pca(X, k)
    Xc = X - mean_
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores_svd = U[:, :k] * S[:k]  # column-scale U by top singular values
    assert np.allclose(scores, scores_svd, atol=1e-8)


def test_explained_variance_ratio_monotone_nonincreasing():
    X = make_lowrank_data(n=140, d=15, r=5, seed=7)
    k = 10
    *_, evr, _total_var, _mean = pca(X, k)
    # ratios should be non-increasing
    diffs = np.diff(evr)
    assert np.all(diffs <= 1e-12)  # allow tiny numerical wiggle


def test_explained_variance_matches_score_variances():
    X = make_lowrank_data(n=200, d=12, r=4, seed=8)
    k = 6
    pcs, scores, ev, *_ = pca(X, k)
    # column-wise sample variance of scores equals explained_variance
    # (since PCs are orthonormal and X is centered)
    sample_vars = scores.var(axis=0, ddof=1)
    assert np.allclose(sample_vars, ev, rtol=1e-8, atol=1e-10)


def test_topk_captures_majority_of_variance_on_lowrank():
    X = make_lowrank_data(n=180, d=20, r=3, noise=0.02, seed=9)
    k = 3
    _, _, ev, evr, total_var, _ = pca(X, k)
    # On low-rank data the top-r should dominate
    assert ev.sum() / total_var > 0.85
    assert evr.sum() > 0.85


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_valid_k_values(k):
    X = make_lowrank_data(n=100, d=6, r=3, seed=10)
    # Ensure it runs for various k up to min(n, d)
    pcs, scores, ev, evr, total_var, mean_ = pca(X, k)
    assert pcs.shape == (X.shape[1], k)
    assert scores.shape == (X.shape[0], k)


def test_reconstruction_identity_when_k_equals_min_dim():
    X = make_lowrank_data(n=50, d=5, r=5, noise=0.0, seed=11)
    k = min(X.shape)
    pcs, scores, *_, mean_ = pca(X, k)
    Xc = X - mean_
    X_hat = scores @ pcs.T
    assert np.allclose(Xc, X_hat, atol=1e-10)

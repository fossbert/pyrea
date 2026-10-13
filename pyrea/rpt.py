"""Residual post-translational (RPT) activity (Alvarez et al., Nat Genet 2016)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.linalg import LinAlgError, solveh_banded
from scipy.stats import rankdata

__all__ = ["viper_rpt"]

_METHODS = ("rank", "lineal", "spline")


def viper_rpt(activity: pd.DataFrame, expression: pd.DataFrame, method: str = "rank",
              spar: float = 1.2) -> pd.DataFrame:
    """Part of VIPER activity not explained by the coding gene's expression.

    For each protein, activity is regressed on the expression of its gene
    across samples; the residuals are the RPT activity. Matches
    ``viper::viperRPT`` (without weights / ``robust``) and is computed without R.

    Parameters
    ----------
    activity : pd.DataFrame
        VIPER activity, proteins x samples.
    expression : pd.DataFrame
        Gene expression, genes x samples.
    method : {"rank", "lineal", "spline"}
        ``"rank"`` (default) regresses the within-gene ranks on each other
        (as in the paper); ``"lineal"`` is ordinary linear regression of the
        raw values; ``"spline"`` fits a cubic smoothing spline of activity on
        expression (``smooth.spline`` with ``spar`` and a knot at every distinct
        expression value, the default of ``viper::viperRPT``), i.e. it also
        removes a non-linear dependence on expression.
    spar : float
        Smoothing parameter of the spline method (R's ``spar``; larger = smoother,
        a straight line in the limit). Ignored by the other methods.

    Returns
    -------
    pd.DataFrame
        RPT activity on the genes and samples shared by both inputs.

    Raises
    ------
    ValueError
        If sample names are not unique in ``activity`` or ``expression`` (collapse
        duplicates first, e.g. ``df.T.groupby(level=0).mean().T``), or the method is unknown.
    """
    if method not in _METHODS:
        raise ValueError("method must be one of %s" % ", ".join(repr(m) for m in _METHODS))
    for name, df in (("activity", activity), ("expression", expression)):
        dup = df.columns[df.columns.duplicated()].unique()
        if len(dup):
            raise ValueError(f"{name} has duplicated sample names (e.g. {list(dup[:3])}); "
                             "collapse them first, e.g. df.T.groupby(level=0).mean().T")
    genes = activity.index.intersection(expression.index, sort=False)
    samples = activity.columns.intersection(expression.columns, sort=False)
    y = activity.loc[genes, samples].to_numpy(dtype=float)
    x = expression.loc[genes, samples].to_numpy(dtype=float)

    if method == "spline":
        return pd.DataFrame(np.vstack([_spline_residuals(xi, yi, spar) for xi, yi in zip(x, y)])
                            if len(genes) else np.empty((0, len(samples))), index=genes, columns=samples)
    if method == "rank":
        y, x = rankdata(y, axis=1), rankdata(x, axis=1)

    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sxx = (xc ** 2).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(sxx > 0, (xc * yc).sum(axis=1, keepdims=True) / sxx, 0.0)
    return pd.DataFrame(yc - slope * xc, index=genes, columns=samples)


def _linear_residuals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc, yc = x - x.mean(), y - y.mean()
    sxx = (xc ** 2).sum()
    return yc - (xc @ yc / sxx if sxx > 0 else 0.0) * xc


def _spline_residuals(x: np.ndarray, y: np.ndarray, spar: float) -> np.ndarray:
    """Residuals of ``smooth.spline(x, y, spar=spar, all.knots=TRUE)`` (unit weights).

    Port of R's penalised B-spline fit: cubic B-splines with a knot at every distinct
    (rescaled) x, penalty ``lambda * integral f''^2`` with ``lambda = r * 256**(3*spar - 1)``,
    where ``r`` is the ratio of the traces of the data and penalty matrices.
    """
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    tol = 1e-6 * iqr
    if not tol > 0:  # R refuses to fit; fall back to the straight line
        return _linear_residuals(x, y)
    key = np.round((x - x.mean()) / tol)
    ukeys, first, inv = np.unique(key, return_index=True, return_inverse=True)
    nx = len(ukeys)
    if nx < 4:
        return _linear_residuals(x, y)
    ux = x[first]  # x of each distinct key (sorted by key = sorted by x)
    wbar = np.bincount(inv, minlength=nx).astype(float)
    ybar = np.bincount(inv, weights=y, minlength=nx) / wbar
    xbar = (ux - ux[0]) / (ux[-1] - ux[0])

    k = 3
    t = np.concatenate([[xbar[0]] * k, xbar, [xbar[-1]] * k])
    nk = nx + 2
    B = BSpline.design_matrix(xbar, t, k).tocsr()
    BtWB = (B.T @ sparse.diags(wbar) @ B).tocsr()

    # penalty: second derivative of the spline is a degree-1 spline on t[2:-2]
    d1 = k / (t[k + 1: nk + k] - t[1: nk])                        # (nk-1)
    D1 = sparse.diags([-d1, d1], [0, 1], shape=(nk - 1, nk))
    t1 = t[1:-1]
    d2 = (k - 1) / (t1[k: nk - 1 + k - 1] - t1[1: nk - 1])        # (nk-2)
    D2 = sparse.diags([-d2, d2], [0, 1], shape=(nk - 2, nk - 1))
    h = np.diff(xbar)
    g = 0.5 / np.sqrt(3.0)
    pts = np.concatenate([xbar[:-1] + h * (0.5 - g), xbar[:-1] + h * (0.5 + g)])
    wq = np.concatenate([h / 2, h / 2])
    Q = BSpline.design_matrix(pts, t[2:-2], 1).tocsr() @ (D2 @ D1)
    Sigma = (Q.T @ sparse.diags(wq) @ Q).tocsr()
    # R's sgram.f integrates the quadratic term of each interval with 0.3330 instead of 1/3;
    # reproduce that so that the fit matches smooth.spline to numerical precision
    m = len(h)
    jump = (Q[m:] - Q[:m]) / (2 * g)
    Sigma = (Sigma - (1 / 3 - 0.333) * (jump.T @ sparse.diags(h) @ jump)).tocsr()

    ratio = BtWB.diagonal()[2: nk - 3].sum() / Sigma.diagonal()[2: nk - 3].sum()
    lam = ratio * 256.0 ** (3.0 * spar - 1.0)
    A = (BtWB + lam * Sigma).tocsr()
    ab = np.zeros((k + 1, nk))
    for d in range(k + 1):
        ab[k - d, d:] = A.diagonal(d)
    try:
        coef = solveh_banded(ab, B.T @ (wbar * ybar))
    except LinAlgError:  # penalty so large that the system is numerically singular: the fit is the line
        return _linear_residuals(x, y)
    fit = B @ coef
    return y - fit[inv]

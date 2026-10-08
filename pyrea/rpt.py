"""Residual post-translational (RPT) activity (Alvarez et al., Nat Genet 2016)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = ["viper_rpt"]


def viper_rpt(activity: pd.DataFrame, expression: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    """Part of VIPER activity not explained by the coding gene's expression.

    For each protein, activity is regressed on the expression of its gene
    across samples (ordinary least squares); the residuals are the RPT
    activity. Matches ``viper::viperRPT`` with ``method`` ``"rank"`` or
    ``"lineal"`` (the spline variant is not implemented) and is computed
    without R.

    Parameters
    ----------
    activity : pd.DataFrame
        VIPER activity, proteins x samples.
    expression : pd.DataFrame
        Gene expression, genes x samples.
    method : {"rank", "lineal"}
        ``"rank"`` regresses the within-gene ranks on each other (as in the
        paper); ``"lineal"`` uses the raw values.

    Returns
    -------
    pd.DataFrame
        RPT activity on the genes and samples shared by both inputs.
    """
    if method not in ("rank", "lineal"):
        raise ValueError("method must be 'rank' or 'lineal'")
    genes = activity.index.intersection(expression.index, sort=False)
    samples = activity.columns.intersection(expression.columns, sort=False)
    y = activity.loc[genes, samples].to_numpy(dtype=float)
    x = expression.loc[genes, samples].to_numpy(dtype=float)
    if method == "rank":
        y, x = rankdata(y, axis=1), rankdata(x, axis=1)

    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sxx = (xc ** 2).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(sxx > 0, (xc * yc).sum(axis=1, keepdims=True) / sxx, 0.0)
    return pd.DataFrame(yc - slope * xc, index=genes, columns=samples)

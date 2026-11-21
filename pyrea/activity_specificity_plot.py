"""Figures for a :class:`pyrea.MultiActivity`: cohorts x sets and the forest plot of a pair.

Clean panels on an Axes that can be passed (``ax=``); the marks are added afterwards (:func:`add_top`; ``ax=None`` uses the current
axes, ``**text_kwargs`` go to matplotlib).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_matrix", "add_top", "plot_forest"]


def plot_matrix(ma, value: str = "padj_ext_percentile_max", ax=None, cmap: str | None = None, title: str | None = None):
    """Cohorts (rows) x sets (columns) heatmap of one column of ``ma.table``. The default shows the worst partner percentile of a member
    (adjusted for the covariates if there are any): small = every member is among the best partners of the others, drawn dark. Other
    columns (``rho_mean``, ``partial_rho_mean``, ...) are drawn with the sequential colour map (large = dark)."""
    mat = ma.matrix(value)
    shown = value if (value != "padj_ext_percentile_max" or "padj_ext_percentile_max" in ma.table.columns) else "ext_percentile_max"
    if cmap is None:
        cmap = "Blues_r" if "percentile" in shown else "Blues"
    if ax is None:
        _, ax = plt.subplots(figsize=(1.8 + 0.6 * mat.shape[1], 1.0 + 0.25 * mat.shape[0]))
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(dtype=float)), cmap=cmap, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    ax.set_title(shown if title is None else title, loc="left", fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    ax._pyrea_activity = (mat, shown)
    return ax


def add_top(ma, ax=None, top_frac: float = 0.05, marker: str = "*", **text_kwargs):
    """After :func:`plot_matrix` of a percentile: put ``marker`` into the cells whose worst partner percentile is at most ``top_frac``."""
    ax = plt.gca() if ax is None else ax
    info = getattr(ax, "_pyrea_activity", None)
    if info is None:
        raise ValueError("Draw the matrix first (plot_matrix) on this axes")
    mat, shown = info
    pct = ("padj_ext_percentile_max" if "padj_ext_percentile_max" in ma.table.columns else "ext_percentile_max")
    perc = ma.matrix(pct)
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    mid = (finite.min() + finite.max()) / 2 if len(finite) else 0
    dark_when_small = "percentile" in shown
    for i, c in enumerate(mat.index):
        for j, s in enumerate(mat.columns):
            if np.isfinite(perc.loc[c, s]) and perc.loc[c, s] <= top_frac and np.isfinite(vals[i, j]):
                dark = (vals[i, j] < mid) if dark_when_small else (vals[i, j] > mid)
                props = dict(ha="center", va="center", fontsize=8, color="white" if dark else "black")
                props.update(text_kwargs)
                ax.text(j, i, marker, **props)
    return ax


def plot_forest(ma, tf1: str, tf2: str, variant: str = "full", adjusted: bool = False, ax=None, title: str | None = None):
    """Forest plot of the correlation of one pair: one row per cohort (point = Spearman or partial correlation, bar = 95 % interval from
    Fisher's z, variance inflated for Spearman) and the pooled random-effects estimate at the bottom (diamond), with ``I2``."""
    pa = ma.pairs[(ma.pairs["variant"] == variant) & (((ma.pairs["tf1"] == tf1) & (ma.pairs["tf2"] == tf2)) |
                                                     ((ma.pairs["tf1"] == tf2) & (ma.pairs["tf2"] == tf1)))]
    if pa.empty:
        raise ValueError(f"The pair {tf1} | {tf2} was not evaluated (variant {variant!r})")
    col = "partial_rho" if adjusted else "rho"
    if col not in pa.columns:
        raise ValueError("The analysis has no covariates (no partial correlations)")
    q = len(ma.settings.get("covariates") or []) if adjusted else 0
    r = pa[col].to_numpy(dtype=float)
    n = pa["n"].to_numpy(dtype=float)
    se = np.sqrt(1.06 / (n - 3 - q))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    meta = ma.meta(variant, adjusted).query("(tf1 == @pa.tf1.iloc[0]) and (tf2 == @pa.tf2.iloc[0])").iloc[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 1.0 + 0.22 * (len(pa) + 2)))
    y = np.arange(len(pa))[::-1] + 1.5
    ax.hlines(y, lo, hi, color="0.55", lw=1)
    ax.plot(r, y, "o", color="#1f77b4", ms=4)
    ax.fill([meta["ci_low"], meta["rho_pooled"], meta["ci_high"], meta["rho_pooled"]], [0.5, 0.8, 0.5, 0.2], color="#d55e00", lw=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_yticks(list(y) + [0.5], list(pa["cohort"]) + ["pooled"], fontsize=7)
    ax.set_xlabel("partial Spearman correlation" if adjusted else "Spearman correlation", fontsize=8)
    ax.set_title(f"{tf1} | {tf2}  (I$^2$ = {meta['I2']:.0%})" if title is None else title, loc="left", fontsize=8)
    return ax

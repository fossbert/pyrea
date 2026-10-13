"""Does protein activity show an effect that expression alone does not (or contradicts)?

For an alteration (mutation, amplification, subtype, ...) the same two-group comparison is made
on three scales of the same gene: its mRNA expression, its VIPER global activity (G) and its
residual post-translational activity (RPT, activity beyond the expression). The effect on each
scale is the rank-based Cliff's delta (``2 * AUC - 1``, in [-1, 1]), which does not depend on the
units of the scales and is therefore comparable between them.

"Significant on one scale, not on the other" is not evidence of a difference (the difference
between significant and not significant is not itself significant). The difference of the effects
is therefore estimated directly, with a paired bootstrap over the samples (the three scales are
measured on the same samples and are correlated, activity being computed from expression).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, rankdata

__all__ = ["activity_vs_expression", "CONCORDANCE_CATEGORIES", "CONCORDANCE_COLORS"]

CONCORDANCE_CATEGORIES = ("concordant", "activity_only", "expression_only", "opposite", "unclear", "none")
CONCORDANCE_COLORS = {"concordant": "#1f77b4", "activity_only": "#d62728", "expression_only": "#ff7f0e",
                      "opposite": "#9467bd", "unclear": "#8c8c8c", "none": "#d9d9d9"}


def _cliff(c: np.ndarray, w: np.ndarray) -> float:
    r = rankdata(np.concatenate([c, w]))
    nc, nw = len(c), len(w)
    return 2 * (r[:nc].sum() - nc * (nc + 1) / 2) / (nc * nw) - 1


def _boot_cliff(c: np.ndarray, w: np.ndarray, ic: np.ndarray, iw: np.ndarray) -> np.ndarray:
    """Cliff's delta of every bootstrap resample (rows of ``ic`` / ``iw`` index ``c`` / ``w``)."""
    v = np.concatenate([c[ic], w[iw]], axis=1)
    r = rankdata(v, axis=1)
    nc, nw = c.size, w.size
    return 2 * (r[:, :nc].sum(axis=1) - nc * (nc + 1) / 2) / (nc * nw) - 1


def _boot_p(b: np.ndarray) -> float:
    """Two-sided bootstrap p-value of 'the statistic is 0'."""
    return float(min(1.0, 2 * min((np.sum(b <= 0) + 1), (np.sum(b >= 0) + 1)) / (len(b) + 1)))


def activity_vs_expression(mutations: pd.DataFrame,
                           activity: pd.DataFrame,
                           expression: pd.DataFrame,
                           rpt: Optional[pd.DataFrame] = None,
                           key: Optional[Mapping[str, str]] = None,
                           min_mut: int = 10,
                           min_wt: int = 10,
                           n_boot: int = 1000,
                           alpha: float = 0.05,
                           min_effect: float = 0.2,
                           seed: int = 0) -> pd.DataFrame:
    """Compare the effect of an alteration on a gene's expression, activity (G) and RPT.

    Parameters
    ----------
    mutations : pd.DataFrame
        Binary rows x samples (1 = altered, 0 = reference, NaN = not profiled, excluded).
    activity, expression : pd.DataFrame
        VIPER activity (G) and mRNA expression, genes x samples. Both are required: a row is only
        analysed for genes present in both.
    rpt : pd.DataFrame, optional
        RPT activity (see :func:`pyrea.viper_rpt`); adds ``delta_RPT`` and ``beyond_expression``.
    key : mapping row id -> gene, optional
        Rows that are not gene names (e.g. a fusion) are analysed on the mapped gene.
    min_mut, min_wt : int
        Minimum group sizes; rows below are omitted.
    n_boot : int
        Bootstrap resamples (carriers and reference samples resampled separately, the same
        samples for all scales).
    alpha : float
        Level for the Benjamini-Hochberg q-values (per scale, over all rows) and for the
        bootstrap intervals (``1 - alpha`` percentile intervals).
    min_effect : float
        Smallest |Cliff's delta| counted as an effect (0.147 / 0.33 / 0.474 are the usual
        small / medium / large thresholds; the default 0.2 is a small-to-medium effect).
    seed : int
        Seed of the bootstrap.

    Returns
    -------
    pd.DataFrame
        One row per analysed id with

        - ``n_mut``, ``n_wt``
        - ``delta_expr``, ``delta_G`` (and ``delta_RPT``): Cliff's delta, > 0 = higher in the altered
          samples, with Mann-Whitney ``p_*`` and BH ``q_*``
        - ``delta_diff`` = ``delta_G - delta_expr`` with ``diff_low`` / ``diff_high`` (bootstrap interval)
          and ``p_diff``: do activity and expression differ in direction or size?
        - ``abs_diff`` = ``|delta_G| - |delta_expr|`` with ``abs_low`` / ``abs_high`` and ``p_abs``: is the
          activity effect larger (> 0) or smaller (< 0) than the expression effect?
        - ``category``:

          ``concordant``      both scales show an effect (q < alpha, |delta| >= min_effect) in the same direction
          ``opposite``        both show an effect, in opposite directions
          ``activity_only``   only G shows an effect and the interval of ``abs_diff`` is above 0
          ``expression_only`` only expression shows an effect and the interval of ``abs_diff`` is below 0
          ``unclear``         only one scale shows an effect but the difference is not established
          ``none``            no effect on either scale

        - ``beyond_expression`` (with ``rpt``): RPT shows an effect in the direction of G, i.e. the
          activity effect is not just the expression effect again.

    Notes
    -----
    Samples with a missing value on any scale are left out for that row. Categories depend on the
    two thresholds ``alpha`` and ``min_effect`` and are a guide for sorting results; read the effects
    and intervals. A signal found only in the activity needs checks before it is believed: copy-number
    effects on neighbouring regulon genes, subtype composition (see :func:`pyrea.stratified_comparison`),
    and replication in another cohort.

    Examples
    --------
    ::

        res = pr.activity_vs_expression(mutations, vpres, emat, rpt=rpt)
        res.sort_values("abs_diff", ascending=False)[["delta_expr", "delta_G", "delta_RPT", "category"]]

        # scatter of the effects with cbrviz (colour = category, diagonal = same effect)
        from cbrviz import XYview
        cats = [c for c in pr.CONCORDANCE_CATEGORIES if c in set(res["category"])]
        xv = XYview(res[["delta_expr", "delta_G"]], s1=res["category"], s1_order=cats,
                    s1_colors=[pr.CONCORDANCE_COLORS[c] for c in cats], highlight=list(res.index[:5]))
    """
    traits = {"expr": expression, "G": activity}
    if rpt is not None:
        traits["RPT"] = rpt
    samples = mutations.columns
    for t in traits.values():
        samples = samples.intersection(t.columns, sort=False)
    rng = np.random.default_rng(seed)
    z = 1 - alpha

    rows = {}
    for rid in mutations.index:
        gene = rid if key is None else key.get(rid, rid)
        if any(gene not in t.index for t in traits.values()):
            continue
        grp = mutations.loc[rid, samples].to_numpy(dtype=float)
        vals = {n: t.loc[gene, samples].to_numpy(dtype=float) for n, t in traits.items()}
        ok = np.isfinite(grp) & np.all([np.isfinite(v) for v in vals.values()], axis=0)
        c, w = ok & (grp == 1), ok & (grp == 0)
        nc, nw = int(c.sum()), int(w.sum())
        if nc < min_mut or nw < min_wt:
            continue
        rec = {"n_mut": nc, "n_wt": nw}
        vc = {n: v[c] for n, v in vals.items()}
        vw = {n: v[w] for n, v in vals.items()}
        for n in traits:
            rec[f"delta_{n}"] = _cliff(vc[n], vw[n])
            rec[f"p_{n}"] = float(mannwhitneyu(vc[n], vw[n]).pvalue)
        ic = rng.integers(0, nc, (n_boot, nc))
        iw = rng.integers(0, nw, (n_boot, nw))
        boot = {n: _boot_cliff(vc[n], vw[n], ic, iw) for n in traits}
        d = boot["G"] - boot["expr"]
        a = np.abs(boot["G"]) - np.abs(boot["expr"])
        lo, hi = (1 - z) / 2 * 100, (1 + z) / 2 * 100
        rec.update(delta_diff=rec["delta_G"] - rec["delta_expr"],
                   diff_low=np.percentile(d, lo), diff_high=np.percentile(d, hi), p_diff=_boot_p(d),
                   abs_diff=abs(rec["delta_G"]) - abs(rec["delta_expr"]),
                   abs_low=np.percentile(a, lo), abs_high=np.percentile(a, hi), p_abs=_boot_p(a))
        rows[rid] = rec

    cols = ["n_mut", "n_wt"] + [f"delta_{n}" for n in traits] + [f"p_{n}" for n in traits]
    out = pd.DataFrame.from_dict(rows, orient="index")
    if out.empty:
        return pd.DataFrame(columns=cols + ["category"])
    for n in traits:
        out[f"q_{n}"] = false_discovery_control(out[f"p_{n}"].to_numpy(), method="bh") if len(out) > 1 \
            else out[f"p_{n}"].to_numpy()

    def present(n):
        return ((out[f"q_{n}"] < alpha) & (out[f"delta_{n}"].abs() >= min_effect)).to_numpy()

    pe, pg = present("expr"), present("G")
    same = np.sign(out["delta_expr"]) == np.sign(out["delta_G"])
    cat = np.full(len(out), "none", dtype=object)
    cat[pe & pg & same.to_numpy()] = "concordant"
    cat[pe & pg & ~same.to_numpy()] = "opposite"
    g_only, e_only = pg & ~pe, pe & ~pg
    cat[g_only] = np.where(out["abs_low"].to_numpy()[g_only] > 0, "activity_only", "unclear")
    cat[e_only] = np.where(out["abs_high"].to_numpy()[e_only] < 0, "expression_only", "unclear")
    out["category"] = cat
    if rpt is not None:
        out["beyond_expression"] = present("RPT") & (np.sign(out["delta_RPT"]) == np.sign(out["delta_G"])).to_numpy()

    order = (["n_mut", "n_wt"] + [f"delta_{n}" for n in traits] + [f"p_{n}" for n in traits]
             + [f"q_{n}" for n in traits]
             + ["delta_diff", "diff_low", "diff_high", "p_diff", "abs_diff", "abs_low", "abs_high", "p_abs",
                "category"] + (["beyond_expression"] if rpt is not None else []))
    return out[order]

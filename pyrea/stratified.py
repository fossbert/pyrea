"""Effect of an alteration within strata (e.g. molecular subtypes) and differences between them.

Pooling samples of different strata mixes the effect of the alteration with baseline differences
between the strata (VIPER activity with ``method="mad"`` is relative to the whole cohort). The
functions here compare mutant and wild-type samples *within* each stratum and then ask whether
the effect differs between strata (interaction). No R needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import mannwhitneyu, rankdata

__all__ = ["stratified_comparison", "differential_activity", "StratifiedComparison"]


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x == y), via ranks."""
    n1, n2 = len(x), len(y)
    r = rankdata(np.concatenate([x, y]))
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def _boot_auc(x: np.ndarray, y: np.ndarray, rng, n_boot: int) -> np.ndarray:
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = _auc(rng.choice(x, len(x)), rng.choice(y, len(y)))
    return out


def _prepare(group: pd.Series, strata: pd.Series, samples: pd.Index, min_group: int):
    """Align group/strata to the samples; keep strata with >= min_group mutants and wild types."""
    g = pd.Series(group).reindex(samples).astype(float)
    s = pd.Series(strata).reindex(samples)
    ok = g.notna() & s.notna()
    keep = []
    for name, idx in s[ok].groupby(s[ok]).groups.items():
        if (g[idx] == 1).sum() >= min_group and (g[idx] == 0).sum() >= min_group:
            keep.append(name)
    ok &= s.isin(keep)
    return g, s, ok, keep


def _design_f(ranks: np.ndarray, mut: np.ndarray, strata_idx: np.ndarray, k: int):
    """F-test (and p) for the mutation x stratum interaction in a rank-based linear model.

    ranks : (n, m) matrix of ranks, one column per tested feature (same design for all).
    Model: rank ~ stratum + mut + mut:stratum; restricted: without the interaction.
    """
    n = len(mut)
    S = np.eye(k)[strata_idx][:, 1:]                       # k-1 stratum dummies
    base = np.column_stack([np.ones(n), S, mut])
    full = np.column_stack([base, S * mut[:, None]])

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, ranks, rcond=None)
        res = ranks - X @ beta
        return (res ** 2).sum(axis=0), np.linalg.matrix_rank(X)

    rss_r, rank_r = rss(base)
    rss_f, rank_f = rss(full)
    df1, df2 = rank_f - rank_r, n - rank_f
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
    return stat, f_dist.sf(stat, df1, df2)


@dataclass
class StratifiedComparison:
    """Result of :func:`stratified_comparison`.

    Attributes
    ----------
    effects : pd.DataFrame
        Index (stratum, trait): ``n_mut``, ``n_wt``, ``auc`` (probability that a mutant has a higher
        value than a wild type of the same stratum; 0.5 = no effect), bootstrap 95 % interval
        ``ci_low`` / ``ci_high`` and Mann-Whitney ``p``.
    interaction : pd.DataFrame
        Index trait: ``p_interaction`` of the mutation x stratum term of a rank-based linear
        model (all strata together; small p = the effect differs between strata).
    pairwise : pd.DataFrame
        Index (trait, stratum1, stratum2): difference of the AUCs (stratum1 - stratum2) with
        bootstrap interval and two-sided bootstrap ``p``.
    """

    effects: pd.DataFrame
    interaction: pd.DataFrame
    pairwise: pd.DataFrame


def stratified_comparison(group: pd.Series,
                          strata: pd.Series,
                          target: str,
                          activity: pd.DataFrame,
                          rpt: Optional[pd.DataFrame] = None,
                          expression: Optional[pd.DataFrame] = None,
                          min_group: int = 3,
                          n_boot: int = 1000,
                          seed: int = 0) -> StratifiedComparison:
    """Effect of an alteration on one protein within strata, and whether it differs between them.

    Parameters
    ----------
    group : pd.Series
        Alteration per sample: 1 = present, 0 = absent (wild type), NaN = not evaluable.
    strata : pd.Series
        Stratum label per sample (e.g. molecular subtype); NaN = excluded.
    target : str
        Gene whose G-activity (``activity``), RPT-activity (``rpt``) and expression
        (``expression``) are compared; a row in each matrix.
    activity, rpt, expression : pd.DataFrame
        Genes x samples.
    min_group : int
        Strata need at least this many mutant and wild-type samples; others are left out.
    n_boot : int
        Bootstrap replicates for the intervals (resampling within mutants and wild types).
    seed : int

    Returns
    -------
    StratifiedComparison

    Notes
    -----
    Mutants are compared with wild types *of the same stratum*, so baseline differences between
    strata do not matter for ``effects``. ``interaction`` uses global ranks and stratum main effects.
    Neither is corrected for multiple testing. With few mutants the intervals are wide: read
    ``auc`` with its interval, not only the p-value.

    Examples
    --------
    ::

        res = pr.stratified_comparison(arid1a, subtype, "ARID1A", vpres, rpt, expression=emat)
        res.effects.round(2)         # per subtype and trait
        res.interaction              # does the effect differ between the subtypes?
        res.pairwise.round(2)        # EBV vs GS etc.
    """

    traits = {"G": activity}
    if rpt is not None:
        traits["RPT"] = rpt
    if expression is not None:
        traits["expression"] = expression
    for name, mat in traits.items():
        if target not in mat.index:
            raise ValueError(f"Target {target!r} not in the {name} matrix")

    samples = activity.columns
    for mat in traits.values():
        samples = samples.intersection(mat.columns, sort=False)
    g, s, ok, keep = _prepare(group, strata, samples, min_group)
    if len(keep) == 0:
        raise ValueError("No stratum has enough mutant and wild-type samples")

    rng = np.random.default_rng(seed)
    eff, boots = {}, {}
    for trait, mat in traits.items():
        v = mat.loc[target, samples]
        for st in keep:
            m = ok & (s == st)
            x, y = v[m & (g == 1)].to_numpy(float), v[m & (g == 0)].to_numpy(float)
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            b = _boot_auc(x, y, rng, n_boot)
            boots[(trait, st)] = b
            eff[(st, trait)] = {"n_mut": len(x), "n_wt": len(y), "auc": _auc(x, y),
                                "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5),
                                "p": mannwhitneyu(x, y, alternative="two-sided").pvalue}
    effects = pd.DataFrame.from_dict(eff, orient="index")
    effects.index = pd.MultiIndex.from_tuples(effects.index, names=["stratum", "trait"])

    inter = {}
    if len(keep) >= 2:
        idx = ok.to_numpy()
        codes = pd.Categorical(s[ok], categories=keep).codes
        mut = (g[ok] == 1).to_numpy(float)
        for trait, mat in traits.items():
            y = rankdata(mat.loc[target, samples][ok].to_numpy(float))[:, None]
            inter[trait] = float(_design_f(y, mut, codes, len(keep))[1][0])
    interaction = pd.DataFrame({"p_interaction": pd.Series(inter, dtype=float)})
    interaction.index.name = "trait"

    prow = {}
    for trait in traits:
        for a, b in combinations(keep, 2):
            d = boots[(trait, a)] - boots[(trait, b)]
            diff = effects.loc[(a, trait), "auc"] - effects.loc[(b, trait), "auc"]
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            prow[(trait, a, b)] = {"auc_diff": diff, "ci_low": np.percentile(d, 2.5),
                                   "ci_high": np.percentile(d, 97.5), "p": max(p, 1 / n_boot)}
    pairwise = pd.DataFrame.from_dict(prow, orient="index")
    if len(pairwise):
        pairwise.index = pd.MultiIndex.from_tuples(pairwise.index, names=["trait", "stratum1", "stratum2"])
    return StratifiedComparison(effects, interaction, pairwise)


def differential_activity(group: pd.Series,
                          activity: pd.DataFrame,
                          strata: Optional[pd.Series] = None,
                          min_group: int = 3) -> pd.DataFrame:
    """Compare mutant and wild-type samples on every row of ``activity`` (all regulators).

    The proteome-wide version of :func:`stratified_comparison`: for each stratum and each
    regulator the AUC (mutant > wild type, within the stratum), the Mann-Whitney p-value and the
    Benjamini-Hochberg q-value; with two or more strata also the rank-based interaction p and q.

    Parameters
    ----------
    group : pd.Series
        1 = alteration present, 0 = wild type, NaN = not evaluable; index = sample ids.
    activity : pd.DataFrame
        Regulators x samples (e.g. ``vpres`` or ``rpt``).
    strata : pd.Series, optional
        Stratum per sample. Without it all samples form one stratum called ``"all"``.
    min_group : int
        Minimum mutants and wild types per stratum.

    Returns
    -------
    pd.DataFrame
        Index regulator; columns ``<stratum>:auc``, ``<stratum>:p``, ``<stratum>:q`` and, with
        several strata, ``interaction:p``, ``interaction:q``. Sorted by interaction p (or by
        the p of the single stratum).

    Examples
    --------
    ::

        res = pr.differential_activity(arid1a, vpres, subtype)
        res.query("`STAD_EBV:q` < 0.1 and `interaction:p` < 0.01")
    """

    if strata is None:
        strata = pd.Series("all", index=activity.columns)
    samples = activity.columns
    g, s, ok, keep = _prepare(group, strata, samples, min_group)
    if len(keep) == 0:
        raise ValueError("No stratum has enough mutant and wild-type samples")

    out = {}
    for st in keep:
        m = ok & (s == st)
        X = activity.loc[:, (m & (g == 1)).to_numpy()].to_numpy(float)
        Y = activity.loc[:, (m & (g == 0)).to_numpy()].to_numpy(float)
        res = mannwhitneyu(X, Y, axis=1, alternative="two-sided")
        out[f"{st}:auc"] = res.statistic / (X.shape[1] * Y.shape[1])
        out[f"{st}:p"] = res.pvalue
        out[f"{st}:q"] = _bh(res.pvalue)

    if len(keep) >= 2:
        sel = ok.to_numpy()
        codes = pd.Categorical(s[ok], categories=keep).codes
        mut = (g[ok] == 1).to_numpy(float)
        R = rankdata(activity.loc[:, sel].to_numpy(float), axis=1).T
        p = _design_f(R, mut, codes, len(keep))[1]
        out["interaction:p"], out["interaction:q"] = p, _bh(p)

    res = pd.DataFrame(out, index=activity.index)
    sort = "interaction:p" if "interaction:p" in res else f"{keep[0]}:p"
    return res.sort_values(sort)


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN stay NaN)."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(pv)
    adj[order] = np.minimum(ranked, 1.0)
    q[ok] = adj
    return q

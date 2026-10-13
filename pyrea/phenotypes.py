"""Compare alteration groups on the activity of a target protein (phenocopy analysis).

Question answered: do samples with alteration B (e.g. a gene fusion) resemble samples with
alteration A (e.g. a mutation of the target gene) in the activity of a protein? Built on the
mutant phenotype score of :mod:`pyrea.mps`; no R needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

from .mps import DEFAULT_LR, _larger_abs, lr_to_rl, relative_likelihood

__all__ = ["compare_phenotypes", "PhenotypeComparison"]


@dataclass
class PhenotypeComparison:
    """Result of :func:`compare_phenotypes`.

    Attributes
    ----------
    scores : pd.DataFrame
        One row per sample: ``group`` (name; the reference name; ``"overlap"`` for samples in
        several groups; NaN if not evaluable), the traits ``G`` / ``RPT`` of the target and one
        column ``MPS[<group>]`` per group (the MPS scale defined by that group against the
        reference). For plotting.
    summary : pd.DataFrame
        Index (group, score): n, mean, median, ``auc`` (probability that a group sample has a
        higher value than a reference sample; 0.5 = no difference), a bootstrap interval
        ``auc_low`` / ``auc_high`` and ``p`` (Mann-Whitney U, two-sided) against the reference.
        Includes the reference itself (n only).
    pairwise : pd.DataFrame
        Index (score, group1, group2) for every pair of groups: n1, n2, ``auc`` (group1 > group2),
        ``p``, ``auc_low`` / ``auc_high``.
    cross : pd.DataFrame
        Index (scale, group): how the samples of ``group`` score on the MPS scale defined by
        ``scale``: n, mean RL, ``frac_mutant_phenotype`` (share above the likelihood-ratio
        threshold), ``auc`` / ``auc_low`` / ``auc_high`` / ``p`` against the reference, and ``in_sample`` (True where
        group == scale: the scale was fitted on these samples, so the value is optimistic).
        Phenocopy: if group B scores high on the scale defined by A (and A on B's scale), the two
        alterations put the target into the same state. See the README for how to read it.
    """

    scores: pd.DataFrame
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    cross: pd.DataFrame


def _nan_stat(func, x: np.ndarray) -> float:
    """``func`` over the finite values of ``x``; NaN (without a numpy warning) if there are none."""
    x = x[np.isfinite(x)]
    return float(func(x)) if len(x) else np.nan


def _mwu(x: np.ndarray, y: np.ndarray):
    """AUC (P(x > y) + ties/2) and two-sided Mann-Whitney p; NaN if a group is empty."""
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    u = mannwhitneyu(x, y, alternative="two-sided")
    return u.statistic / (len(x) * len(y)), u.pvalue


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + ties / 2 over the finite values; NaN if a group is empty."""
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    r = rankdata(np.concatenate([x, y]))
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def _interval(values, alpha: float):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


_REF = "\0reference"
_CHUNK = 2_000_000  # elements of the largest (resamples x points x data) kernel array per block


def _auc_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC (P(x > y) + ties / 2) for every row of ``x`` (B x n1) against ``y`` (B x n2).

    Rows with a missing value fall back to :func:`_auc`, which drops missing values."""
    out = np.full(len(x), np.nan)
    if x.shape[1] == 0 or y.shape[1] == 0:
        return out
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if ok.any():
        n1, n2 = x.shape[1], y.shape[1]
        r = rankdata(np.concatenate([x[ok], y[ok]], axis=1), axis=1)
        out[ok] = (r[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n2)
    for i in np.flatnonzero(~ok):
        out[i] = _auc(x[i], y[i])
    return out


def _rl_rows(x: np.ndarray, m: np.ndarray, w: np.ndarray, bw_adjust: float) -> np.ndarray:
    """Row-wise :func:`pyrea.relative_likelihood`: ``x`` (B x k) scored on the KDEs of ``m`` / ``w`` (B x n).

    Same estimator as ``scipy.stats.gaussian_kde`` (Scott's bandwidth times ``bw_adjust``), evaluated for
    all resamples at once. Rows whose groups have fewer than two distinct values are NaN."""
    def dens(d):
        h = d.std(axis=1, ddof=1) * d.shape[1] ** (-0.2) * bw_adjust
        usable = (np.ptp(d, axis=1) > 0) & (d.shape[1] >= 2)
        hs = np.where(usable, h, 1.0)[:, None]
        z = (x[:, :, None] - d[:, None, :]) / hs[:, :, None]
        return np.exp(-0.5 * z * z).mean(axis=2) / (hs * np.sqrt(2 * np.pi)), usable
    pm, um = dens(m)
    pw, uw = dens(w)
    den = pm + pw
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = np.where(den > 0, (pm - pw) / den, np.nan)
    rl[~(um & uw)] = np.nan
    return rl


def _bootstrap_auc(scores, exclusive, refv, names, traits, min_group, n_boot, alpha, seed, bw_adjust):
    """Bootstrap intervals for all AUCs of :func:`compare_phenotypes`.

    Every group and the reference are resampled separately. The MPS scale of a group is
    estimated again from each resample (group vs reference), so the uncertainty of the scale is
    part of the interval, not only the sampling of the scored samples.
    Returns ``(summary, pairwise, cross)``, dicts of ``(low, high)``.
    """
    rng = np.random.default_rng(seed)
    vals = {t: scores[t].to_numpy(dtype=float) for t in traits}
    finite = np.all([np.isfinite(v) for v in vals.values()], axis=0)
    pos = {k: np.flatnonzero(exclusive[k].to_numpy() & finite) for k in names}
    pos[_REF] = np.flatnonzero(refv & finite)
    pairs = list(combinations(names, 2))
    order = list(names) + [_REF]
    summ, pair, cross = {}, {}, {}
    for t in traits:
        for g in names:
            summ[(g, t)] = []
        for a, b in pairs:
            pair[(t, a, b)] = []
    for k in names:
        for g in names:
            summ[(g, f"MPS[{k}]")] = cross[(k, g)] = []
        for a, b in pairs:
            pair[(f"MPS[{k}]", a, b)] = []

    width = max(sum(len(p) for p in pos.values()) * max(len(p) for p in pos.values()), 1)
    block = max(1, min(n_boot, _CHUNK // width))
    done = 0
    while done < n_boot:
        bc = min(block, n_boot - done)
        done += bc
        idx = {g: (p[rng.integers(0, len(p), (bc, len(p)))] if len(p) else np.empty((bc, 0), dtype=int))
               for g, p in pos.items()}
        for t in traits:
            v = vals[t]
            for g in names:
                summ[(g, t)].append(_auc_rows(v[idx[g]], v[idx[_REF]]))
            for a, b in pairs:
                pair[(t, a, b)].append(_auc_rows(v[idx[a]], v[idx[b]]))
        cat = np.concatenate([idx[g] for g in order], axis=1)
        cuts = np.cumsum([idx[g].shape[1] for g in order])[:-1]
        for k in names:
            if len(pos[k]) < min_group:
                nan = np.full(bc, np.nan)
                for g in names:
                    summ[(g, f"MPS[{k}]")].append(nan)
                for a, b in pairs:
                    pair[(f"MPS[{k}]", a, b)].append(nan)
                continue
            rl = np.full(cat.shape, np.nan)
            for t in traits:
                v = vals[t]
                rl = _larger_abs(rl, _rl_rows(v[cat], v[idx[k]], v[idx[_REF]], bw_adjust))
            xs = dict(zip(order, np.split(rl, cuts, axis=1)))
            for g in names:
                summ[(g, f"MPS[{k}]")].append(_auc_rows(xs[g], xs[_REF]))
            for a, b in pairs:
                pair[(f"MPS[{k}]", a, b)].append(_auc_rows(xs[a], xs[b]))

    ci = lambda d: {key: _interval(np.concatenate(v) if v else [], alpha) for key, v in d.items()}
    return ci(summ), ci(pair), ci(cross)


def compare_phenotypes(groups: Mapping[str, pd.Series],
                       target: str,
                       activity: pd.DataFrame,
                       rpt: Optional[pd.DataFrame] = None,
                       reference: str = "neither",
                       lr: float = DEFAULT_LR,
                       min_group: int = 3,
                       bw_adjust: float = 1.0,
                       n_boot: int = 1000,
                       alpha: float = 0.05,
                       seed: int = 0) -> PhenotypeComparison:
    """Compare alteration groups on the activity of ``target``.

    Samples are assigned to the groups, and only to one: a sample in several groups is labelled
    ``"overlap"`` and left out of all comparisons. The *reference* is every sample that is 0 in
    **all** groups (NaN in any group, i.e. not profiled, excludes a sample from the reference).
    This keeps carriers of the other alteration out of the wild-type group, which is what a
    plain per-gene MPS does not do.

    For every group an MPS scale is defined (group vs reference, activity G and RPT combined as in
    :func:`pyrea.mutant_phenotype_score`) and all other groups are scored on it. If B is a
    phenocopy of A, B scores high on A's scale and A scores high on B's scale (``cross``).

    Parameters
    ----------
    groups : mapping of name -> pd.Series
        Binary per sample (1 = alteration present, 0 = absent, NaN = not profiled), index =
        sample ids, e.g. ``{"RHOA mut": mut_row, "ARHGAP fusion": fusion_row}``.
    target : str
        Protein whose activity is compared (a row of ``activity``).
    activity, rpt : pd.DataFrame
        G-activity and (optionally) RPT-activity, proteins x samples.
    reference : str
        Label of the reference group.
    lr : float
        Likelihood ratio for the mutant phenotype (3 = RL > 0.5).
    min_group : int
        Groups with fewer samples get no MPS scale (NaN); a warning names them (``n=0``: nobody carries
        the alteration, e.g. no amplification events of that gene in the export).
    bw_adjust : float
        Multiplier of the KDE bandwidth of the MPS scales (see :func:`pyrea.relative_likelihood`).
    n_boot : int
        Bootstrap resamples for the AUC intervals (``auc_low`` / ``auc_high``); 0 switches them
        off (the columns are then NaN). Every group and the reference are resampled separately;
        for the MPS scales the scale is estimated again in each resample.
    alpha : float
        The intervals are the ``alpha / 2`` and ``1 - alpha / 2`` percentiles (default 95 %).
    seed : int
        Seed of the bootstrap.

    Returns
    -------
    PhenotypeComparison

    Examples
    --------
    ::

        groups = {"RHOA mut": mut.loc["RHOA"], "ARHGAP fusion": fus.loc["CLDN-ARHGAP"]}
        cmp = pr.compare_phenotypes(groups, "RHOA", vpres, rpt)
        cmp.summary                    # G, RPT and MPS per group vs. reference
        cmp.cross.round(2)             # does the fusion score like a RHOA mutation (and v.v.)?
        cmp.scores.boxplot("G", by="group")
    """

    if target not in activity.index:
        raise ValueError(f"Target {target!r} has no activity (not in the regulon)")
    traits = {"G": activity}
    if rpt is not None:
        traits["RPT"] = rpt

    G = pd.DataFrame({k: pd.Series(v) for k, v in groups.items()}).astype(float)
    samples = G.index
    for t in traits.values():
        samples = samples.intersection(t.columns, sort=False)
    G = G.loc[samples]

    member = G.eq(1)
    ref = G.eq(0).all(axis=1)
    overlap = member.sum(axis=1) > 1
    exclusive = {k: member[k] & ~overlap for k in G}

    label = pd.Series(np.nan, index=samples, dtype=object)
    label[ref] = reference
    for k, m in exclusive.items():
        label[m] = k
    label[overlap] = "overlap"

    scores = pd.DataFrame({"group": label})
    for name, tr in traits.items():
        scores[name] = tr.loc[target, samples].to_numpy(dtype=float)

    refv = ref.to_numpy()
    if not refv.any():
        raise ValueError("No reference samples: no sample is 0 in all groups. Check the groups "
                         "(a group that is 1 or NaN everywhere leaves no reference).")
    small = {k: int(m.sum()) for k, m in exclusive.items() if m.sum() < min_group}
    if small:
        warnings.warn("Groups with fewer than min_group=%d samples get no MPS scale (their scores are NaN): %s"
                      % (min_group, ", ".join(f"'{k}' (n={n})" for k, n in small.items())
                         + ("; n=0 means nobody carries the alteration" if 0 in small.values() else "")),
                      UserWarning, stacklevel=2)
    for k, m in exclusive.items():
        rl = np.full(len(samples), np.nan)
        if m.sum() >= min_group:
            for name in traits:
                v = scores[name].to_numpy()
                rl = _larger_abs(rl, relative_likelihood(v, v[m.to_numpy()], v[refv], bw_adjust))
        scores[f"MPS[{k}]"] = rl

    cols = [c for c in scores.columns if c != "group"]
    names = list(G.columns)

    rows = {}
    for c in cols:
        x = scores[c].to_numpy()
        rows[(reference, c)] = {"n": int(refv.sum()), "mean": _nan_stat(np.mean, x[refv]), "median": _nan_stat(np.median, x[refv])}
        for k in names:
            m = exclusive[k].to_numpy()
            auc, p = _mwu(x[m], x[refv])
            rows[(k, c)] = {"n": int(m.sum()), "mean": _nan_stat(np.mean, x[m]), "median": _nan_stat(np.median, x[m]),
                            "auc": auc, "p": p}
    if n_boot > 0:
        b_summ, b_pair, b_cross = _bootstrap_auc(scores, exclusive, refv, names, list(traits), min_group,
                                                 n_boot, alpha, seed, bw_adjust)
    else:
        b_summ, b_pair, b_cross = {}, {}, {}
    nan2 = (np.nan, np.nan)
    for (k, c), rec in rows.items():
        if k != reference:
            rec["auc_low"], rec["auc_high"] = b_summ.get((k, c), nan2)
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index = pd.MultiIndex.from_tuples(summary.index, names=["group", "score"])

    prow = {}
    for c in cols:
        x = scores[c].to_numpy()
        for a, b in combinations(names, 2):
            ma, mb = exclusive[a].to_numpy(), exclusive[b].to_numpy()
            auc, p = _mwu(x[ma], x[mb])
            lo, hi = b_pair.get((c, a, b), nan2)
            prow[(c, a, b)] = {"n1": int(ma.sum()), "n2": int(mb.sum()), "auc": auc, "p": p,
                               "auc_low": lo, "auc_high": hi}
    pairwise = pd.DataFrame.from_dict(prow, orient="index")
    if len(pairwise):
        pairwise.index = pd.MultiIndex.from_tuples(pairwise.index, names=["score", "group1", "group2"])

    thr = lr_to_rl(lr)
    crow = {}
    for k in names:
        x = scores[f"MPS[{k}]"].to_numpy()
        for b in names:
            m = exclusive[b].to_numpy()
            auc, p = _mwu(x[m], x[refv])
            vals = x[m][np.isfinite(x[m])]
            lo, hi = b_cross.get((k, b), nan2)
            crow[(k, b)] = {"n": int(m.sum()), "mean_rl": vals.mean() if len(vals) else np.nan,
                            "frac_mutant_phenotype": (vals > thr).mean() if len(vals) else np.nan,
                            "auc": auc, "auc_low": lo, "auc_high": hi, "p": p, "in_sample": k == b}
    cross = pd.DataFrame.from_dict(crow, orient="index")
    cross.index = pd.MultiIndex.from_tuples(cross.index, names=["scale", "group"])
    return PhenotypeComparison(scores, summary, pairwise, cross)

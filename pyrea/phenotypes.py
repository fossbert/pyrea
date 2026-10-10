"""Compare alteration groups on the activity of a target protein (phenocopy analysis).

Question answered: do samples with alteration B (e.g. a gene fusion) resemble samples with
alteration A (e.g. a mutation of the target gene) in the activity of a protein? Built on the
mutant phenotype score of :mod:`pyrea.mps`; no R needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

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
        higher value than a reference sample; 0.5 = no difference) and ``p`` (Mann-Whitney U,
        two-sided) against the reference. Includes the reference itself (n only).
    pairwise : pd.DataFrame
        Index (score, group1, group2) for every pair of groups: n1, n2, ``auc`` (group1 > group2),
        ``p``.
    cross : pd.DataFrame
        Index (scale, group): how the samples of ``group`` score on the MPS scale defined by
        ``scale``: n, mean RL, ``frac_mutant_phenotype`` (share above the likelihood-ratio
        threshold), ``auc`` / ``p`` against the reference, and ``in_sample`` (True where
        group == scale: the scale was fitted on these samples, so the value is optimistic).
        Phenocopy: if group B scores high on the scale defined by A (and A on B's scale), the two
        alterations put the target into the same state. See the README for how to read it.
    """

    scores: pd.DataFrame
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    cross: pd.DataFrame


def _mwu(x: np.ndarray, y: np.ndarray):
    """AUC (P(x > y) + ties/2) and two-sided Mann-Whitney p; NaN if a group is empty."""
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    u = mannwhitneyu(x, y, alternative="two-sided")
    return u.statistic / (len(x) * len(y)), u.pvalue


def compare_phenotypes(groups: Mapping[str, pd.Series],
                       target: str,
                       activity: pd.DataFrame,
                       rpt: Optional[pd.DataFrame] = None,
                       reference: str = "neither",
                       lr: float = DEFAULT_LR,
                       min_group: int = 3) -> PhenotypeComparison:
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
        Groups with fewer samples get no MPS scale (NaN).

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
    for k, m in exclusive.items():
        rl = np.full(len(samples), np.nan)
        if m.sum() >= min_group:
            for name in traits:
                v = scores[name].to_numpy()
                rl = _larger_abs(rl, relative_likelihood(v, v[m.to_numpy()], v[refv]))
        scores[f"MPS[{k}]"] = rl

    cols = [c for c in scores.columns if c != "group"]
    names = list(G.columns)

    rows = {}
    for c in cols:
        x = scores[c].to_numpy()
        rows[(reference, c)] = {"n": int(refv.sum()), "mean": np.nanmean(x[refv]), "median": np.nanmedian(x[refv])}
        for k in names:
            m = exclusive[k].to_numpy()
            auc, p = _mwu(x[m], x[refv])
            rows[(k, c)] = {"n": int(m.sum()), "mean": np.nanmean(x[m]) if m.any() else np.nan,
                            "median": np.nanmedian(x[m]) if m.any() else np.nan, "auc": auc, "p": p}
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index = pd.MultiIndex.from_tuples(summary.index, names=["group", "score"])

    prow = {}
    for c in cols:
        x = scores[c].to_numpy()
        for a, b in combinations(names, 2):
            ma, mb = exclusive[a].to_numpy(), exclusive[b].to_numpy()
            auc, p = _mwu(x[ma], x[mb])
            prow[(c, a, b)] = {"n1": int(ma.sum()), "n2": int(mb.sum()), "auc": auc, "p": p}
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
            crow[(k, b)] = {"n": int(m.sum()), "mean_rl": vals.mean() if len(vals) else np.nan,
                            "frac_mutant_phenotype": (vals > thr).mean() if len(vals) else np.nan,
                            "auc": auc, "p": p, "in_sample": k == b}
    cross = pd.DataFrame.from_dict(crow, orient="index")
    cross.index = pd.MultiIndex.from_tuples(cross.index, names=["scale", "group"])
    return PhenotypeComparison(scores, summary, pairwise, cross)

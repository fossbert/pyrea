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
from scipy.special import ndtr
from scipy.stats import mannwhitneyu, rankdata

from .mps import DEFAULT_LR, DEFAULT_RL_METHOD, RL_METHODS, _larger_abs, lr_to_rl, relative_likelihood

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
    target, method, reference, groups, lr, alpha, n_boot :
        The settings of the comparison (target protein, RL definition, reference label, group names in
        the order given, likelihood-ratio threshold, interval level, bootstrap resamples). The figure
        methods use them for titles and colours.

    Methods
    -------
    ``plot()`` (four-panel overview), ``plot_scores()``, ``plot_scale_scatter()``, ``plot_cross()``,
    ``plot_auc()``: clean figures on an Axes that can be passed; ``add_group_stats()``, ``add_pair_stats()``,
    ``add_scatter_stats()``, ``add_cross_stats()``, ``add_auc_p()``: statistics added afterwards, selectively;
    see :mod:`pyrea.phenotypes_plot`. ``scale_xy()``: the data of the scatter for a custom plot, e.g. with
    ``cbrviz.XYview``.
    """

    scores: pd.DataFrame
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    cross: pd.DataFrame
    target: str = ""
    method: str = DEFAULT_RL_METHOD
    reference: str = "neither"
    groups: tuple = ()
    lr: float = DEFAULT_LR
    alpha: float = 0.05
    n_boot: int = 0

    # ------------------------------------------------------------------ figures (see pyrea.phenotypes_plot)
    # The panels are clean. Each draws on a matplotlib Axes that can be passed (``ax=``) and returns it; statistics are
    # added afterwards, selectively, with the ``add_*`` methods (``ax=None``: current axes), as in cbrviz.StripBox.
    @property
    def score_names(self) -> list:
        """Names of the scores available for the panels: ``"G"``, ``"RPT"`` (if given) and ``"MPS[<group>]"`` per group."""
        return [c for c in self.scores.columns if c != "group"]

    def plot(self, scale_a: Optional[str] = None, scale_b: Optional[str] = None, score: str = "G", figsize=None,
             axes=None, stats: bool = False, colors: Optional[dict] = None, pairs: Optional[list] = None,
             scores: Optional[list] = None, groups: Optional[list] = None):
        """Four-panel overview (``plot_scores``, ``plot_scale_scatter``, ``plot_cross``, ``plot_auc``).

        Returns the figure, or the list of axes if ``axes`` (4, or 2 with a single group) is given.
        ``stats=True`` adds all statistics (for a figure that goes into a manuscript). ``scores`` selects the rows of
        the forest plot (see ``score_names``), ``groups`` the groups shown in all panels."""
        from .phenotypes_plot import plot_phenocopy
        return plot_phenocopy(self, scale_a, scale_b, score, figsize, axes, stats, colors, pairs, scores, groups)

    def plot_scores(self, score: str = "G", ax=None, seed: int = 0, title: Optional[str] = None,
                    colors: Optional[dict] = None, groups: Optional[list] = None, reference: bool = True):
        """Distribution of ``score`` (``"G"``, ``"RPT"``, ``"MPS[<group>]"``) per group, one point per sample;
        ``groups`` selects and orders the groups, ``reference=False`` hides the reference."""
        from .phenotypes_plot import plot_scores
        return plot_scores(self, score, ax, seed, title, colors, groups, reference)

    def plot_scale_scatter(self, scale_a: Optional[str] = None, scale_b: Optional[str] = None, ax=None,
                           title: Optional[str] = None, colors: Optional[dict] = None, groups: Optional[list] = None):
        """Every sample on the MPS scale of group A (x) and of group B (y); ``groups`` restricts the groups drawn."""
        from .phenotypes_plot import plot_scale_scatter
        return plot_scale_scatter(self, scale_a, scale_b, ax, title, colors, groups)

    def plot_cross(self, ax=None, cmap: str = "RdBu_r", title: Optional[str] = None, groups: Optional[list] = None):
        """Matrix of ``cross``: AUC against the reference per (scale, group), in-sample cells framed;
        ``groups`` restricts the matrix to some groups."""
        from .phenotypes_plot import plot_cross
        return plot_cross(self, ax, cmap, title, groups)

    def plot_auc(self, ax=None, scores: Optional[list] = None, pairwise: bool = False, title: Optional[str] = None,
                 colors: Optional[dict] = None, groups: Optional[list] = None):
        """Forest plot of the AUCs with their bootstrap intervals; ``scores`` (see ``score_names``) and ``groups``
        select the rows."""
        from .phenotypes_plot import plot_auc
        return plot_auc(self, ax, scores, pairwise, title, colors, groups)

    def add_group_stats(self, score: str = "G", ax=None, ci: bool = True, colors: Optional[dict] = None,
                        exact_p: bool = False, **text_kwargs):
        """After ``plot_scores``: AUC (with interval) and Mann-Whitney p of every group against the reference."""
        from .phenotypes_plot import add_group_stats
        return add_group_stats(self, score, ax, ci, colors, exact_p, **text_kwargs)

    def add_pair_stats(self, group_a: str, group_b: str, score: str = "G", ax=None, yoffset: float = 0.08,
                       auc: bool = True, exact_p: bool = False, line_kwargs: Optional[dict] = None, **text_kwargs):
        """After ``plot_scores``: bracket between two groups with the AUC and p of their comparison."""
        from .phenotypes_plot import add_pair_stats
        return add_pair_stats(self, group_a, group_b, score, ax, yoffset, auc, exact_p, line_kwargs, **text_kwargs)

    def add_scatter_stats(self, ax=None, scale_a: Optional[str] = None, scale_b: Optional[str] = None,
                          loc: str = "upper left", **text_kwargs):
        """After ``plot_scale_scatter``: the AUC of every group on both scales as a small text box."""
        from .phenotypes_plot import add_scatter_stats
        return add_scatter_stats(self, ax, scale_a, scale_b, loc, **text_kwargs)

    def add_cross_stats(self, ax=None, ci: bool = True, share: bool = True, **text_kwargs):
        """After ``plot_cross``: AUC, interval and share with the mutant phenotype in the cells."""
        from .phenotypes_plot import add_cross_stats
        return add_cross_stats(self, ax, ci, share, **text_kwargs)

    def add_auc_p(self, ax=None, exact_p: bool = False, **text_kwargs):
        """After ``plot_auc``: the Mann-Whitney p of every row at the right edge."""
        from .phenotypes_plot import add_auc_p
        return add_auc_p(self, ax, exact_p, **text_kwargs)

    def scale_xy(self, scale_a: Optional[str] = None, scale_b: Optional[str] = None):
        """Data for a custom scatter (e.g. ``cbrviz.XYview``): ``(xy, group)``.

        ``xy`` has the columns ``scale_A`` and ``scale_B`` (RL of every sample on the MPS scale of the two
        groups), ``group`` is the group label per sample; samples that are in several groups or not evaluable
        are dropped.
        """
        names = list(self.groups)
        if len(names) < 2:
            raise ValueError("Need at least two groups")
        a = names[0] if scale_a is None else scale_a
        b = names[1] if scale_b is None else scale_b
        for g in (a, b):
            if g not in names:
                raise ValueError(f"Unknown group {g!r}; groups are {names}")
        sc = self.scores
        keep = sc["group"].isin([self.reference] + names)
        xy = pd.DataFrame({"scale_A": sc.loc[keep, f"MPS[{a}]"], "scale_B": sc.loc[keep, f"MPS[{b}]"]})
        return xy.dropna(), sc.loc[xy.dropna().index, "group"]


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


def _rl_rows(x: np.ndarray, m: np.ndarray, w: np.ndarray, bw_adjust: float,
             method: str = DEFAULT_RL_METHOD) -> np.ndarray:
    """Row-wise :func:`pyrea.relative_likelihood`: ``x`` (B x k) scored on the KDEs of ``m`` / ``w`` (B x n).

    Same estimators as ``scipy.stats.gaussian_kde`` (Scott's bandwidth times ``bw_adjust``), evaluated for
    all resamples at once. Rows whose groups have fewer than two distinct values are NaN."""
    def band(d):
        h = d.std(axis=1, ddof=1) * d.shape[1] ** (-0.2) * bw_adjust
        usable = (np.ptp(d, axis=1) > 0) & (d.shape[1] >= 2)
        return np.where(usable, h, 1.0)[:, None], usable

    hm, um = band(m)
    hw, uw = band(w)
    if method == "density":
        def dens(d, h):
            z = (x[:, :, None] - d[:, None, :]) / h[:, :, None]
            return np.exp(-0.5 * z * z).mean(axis=2) / (h * np.sqrt(2 * np.pi))
        pm, pw = dens(m, hm), dens(w, hw)
    else:
        o = np.where(m.mean(axis=1) >= w.mean(axis=1), 1.0, -1.0)[:, None]
        xo, mo, wo = o * x, o * m, o * w
        pm = ndtr((xo[:, :, None] - mo[:, None, :]) / hm[:, :, None]).mean(axis=2)
        pw = 1.0 - ndtr((xo[:, :, None] - wo[:, None, :]) / hw[:, :, None]).mean(axis=2)
    den = pm + pw
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = np.where(den > 0, (pm - pw) / den, np.nan)
    rl[~(um & uw)] = np.nan
    return rl


def _bootstrap_auc(scores, exclusive, refv, names, traits, min_group, n_boot, alpha, seed, bw_adjust,
                   method=DEFAULT_RL_METHOD):
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
                rl = _larger_abs(rl, _rl_rows(v[cat], v[idx[k]], v[idx[_REF]], bw_adjust, method))
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
                       method: str = DEFAULT_RL_METHOD,
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
    bw_adjust, method :
        Bandwidth multiplier and RL definition of the MPS scales (see :func:`pyrea.relative_likelihood`).
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

    if method not in RL_METHODS:
        raise ValueError("method must be one of %s" % ", ".join(repr(m) for m in RL_METHODS))
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
                rl = _larger_abs(rl, relative_likelihood(v, v[m.to_numpy()], v[refv], bw_adjust, method))
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
                                                 n_boot, alpha, seed, bw_adjust, method)
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
    return PhenotypeComparison(scores, summary, pairwise, cross, target=target, method=method, reference=reference,
                               groups=tuple(names), lr=lr, alpha=alpha, n_boot=n_boot)

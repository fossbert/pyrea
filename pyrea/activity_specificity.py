"""Is a set of regulators more closely linked than other regulators in the *activity* of a cohort?

The activity counterpart of :func:`pyrea.regulon_specificity`: instead of the overlap of the target tails, the Spearman correlation of
the VIPER activities across the samples of a cohort is evaluated, for the pairs inside every set and for the set as a whole:

- **Partner ranks** (``extension``): the activity of every regulator of the network is regressed on the activities of the other members
  (their *base*); the members are ranked among **all** regulators by the R² of this regression (for a pair: by the squared correlation, or
  by the signed correlation with ``direction="positive"`` / ``"negative"``). Rank 1 for both members of a pair: *mutual best partner*.
- **Adjustment** (``covariates``, e.g. tumour purity): all statistics are given without and with adjustment. The adjusted ones are partial
  Spearman correlations (the ranks of the activities are regressed on the ranks of the covariates; the correlation of the residuals, p-value with
  ``n - 2 - q`` degrees of freedom for ``q`` covariates), and the ranks are those of the residuals. Whatever the stroma (or the purity)
  drives in all regulators of a module disappears there.
- **Activity variants** (``variants``): the correlations of the pairs are also computed for further activity matrices, e.g. the activity
  from a regulon restricted to the candidates without and with the pleiotropy ("shadow") correction of VIPER, to see how much of the
  correlation lives in the shared targets.

The activity matrix of the whole network is the reference (``activity``, regulators x samples, e.g. from :func:`pyrea.viper_activity`
with ``eset_filter=False``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as t_dist

__all__ = ["activity_specificity", "ActivitySpecificity", "activity_partners"]

_DIRECTIONS = ("abs", "positive", "negative")


def _standardize(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    sd = np.sqrt((a ** 2).sum(axis=-1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / sd


def _design(covariates: Optional[pd.DataFrame], samples: pd.Index) -> Optional[np.ndarray]:
    """Rank-transformed covariate columns (n x q), or None (a binary column is unchanged in effect)."""
    if covariates is None:
        return None
    cov = covariates.to_frame() if isinstance(covariates, pd.Series) else covariates
    cov = cov.reindex(samples).astype(float)
    return np.apply_along_axis(rankdata, 0, cov.to_numpy())


def _residualize(Y: np.ndarray, Z: Optional[np.ndarray]) -> np.ndarray:
    """Rows of ``Y`` (m x n) with the covariates ``Z`` (n x q) and an intercept regressed out, then standardized."""
    if Z is None or Z.shape[1] == 0:
        return _standardize(Y)
    D = np.column_stack([np.ones(Z.shape[0]), Z])
    coef, *_ = np.linalg.lstsq(D, Y.T, rcond=None)
    return _standardize(Y - (D @ coef).T)


def _rank_rows(A: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(rankdata, 1, A)


def _corr_p(rho: float, df: int) -> float:
    if not np.isfinite(rho) or df < 1:
        return np.nan
    if abs(rho) >= 1:
        return 0.0
    t = rho * np.sqrt(df / (1 - rho ** 2))
    return float(2 * t_dist.sf(abs(t), df))


def _check_sets(sets) -> list:
    if isinstance(sets, (str, bytes)) or not len(sets):
        raise ValueError("sets must be a non-empty sequence of regulator sequences, e.g. [('A', 'B'), ('A', 'B', 'C')]")
    out = []
    for s in sets:
        if isinstance(s, str):
            raise ValueError(f"Every set is a sequence of regulators, got the string {s!r}")
        s = tuple(map(str, s))
        if len(s) < 2:
            raise ValueError(f"A set needs at least two regulators, got {s}")
        if len(set(s)) != len(s):
            raise ValueError(f"Regulators of a set must be unique, got {s}")
        out.append(s)
    return out


class _Prepared:
    """Rank-transformed (and optionally covariate-adjusted) activities of one matrix over the usable samples."""

    def __init__(self, activity: pd.DataFrame, samples: pd.Index, Z: Optional[np.ndarray]):
        A = activity.loc[:, samples]
        self.names = A.index
        self.pos = pd.Series(np.arange(len(A)), index=A.index)
        R = _rank_rows(A.to_numpy(dtype=float))
        self.raw = _standardize(R)
        self.adj = _residualize(R, Z) if Z is not None else None
        self.n = len(samples)
        self.q = 0 if Z is None else Z.shape[1]


def _usable(activity: pd.DataFrame, others: Sequence[pd.DataFrame], covariates, min_samples: int) -> tuple:
    """Samples shared by all matrices with complete covariates, and the regulators without missing values."""
    samples = activity.columns
    for o in others:
        samples = samples.intersection(o.columns)
    n_start = len(samples)
    if covariates is not None:
        cov = covariates.to_frame() if isinstance(covariates, pd.Series) else covariates
        if cov.index.duplicated().any():
            raise ValueError("covariates has duplicated sample names")
        ok = cov.reindex(samples).notna().all(axis=1)
        samples = samples[ok.to_numpy()]
    if len(samples) < min_samples:
        raise ValueError(f"Only {len(samples)} usable samples (min_samples={min_samples}) after matching the matrices and covariates")
    return samples, n_start - len(samples)


@dataclass
class ActivitySpecificity:
    """Result of :func:`activity_specificity`.

    Attributes
    ----------
    pairs : pd.DataFrame
        Every pair inside the sets, once per activity variant (``variant``: ``"full"`` for ``activity``, or the keys of ``variants``):
        ``rho`` (Spearman) with ``p``, and, if covariates were given, ``partial_rho`` with ``partial_p``; ``n`` samples.
    table : pd.DataFrame
        One row per set: ``n_samples``, ``rho_mean`` (mean pairwise Spearman), ``abs_rho_min`` (the weakest pair), the partial
        versions ``partial_rho_mean`` / ``partial_abs_rho_min``, and from ``extension``: ``ext_rank_max`` / ``ext_percentile_max``
        (the worst rank / percentile of a member; the set is only as specific as its weakest link), ``mutual_best`` (every member is the
        best partner of the others) and the same for the adjusted ranks (``padj_ext_rank_max``, ``padj_ext_percentile_max``,
        ``padj_mutual_best``).
    extension : pd.DataFrame
        Per set, member and ``adjusted``: the rank of the member among **all** regulators as a partner of the other members (``base``),
        ``n_candidates``, ``percentile`` (rank / candidates, small = good), ``stat`` (R² of the member on the base; for a pair the
        signed correlation), ``top`` (the best candidates with their statistic).
    settings : dict
        ``direction``, ``n_samples``, ``n_dropped`` (samples without covariates), ``covariates`` (names), ``n_regulators``, ``variants``.
    """

    pairs: pd.DataFrame
    table: pd.DataFrame
    extension: pd.DataFrame
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ActivitySpecificity({len(self.table)} set(s), {self.settings.get('n_samples')} samples, "
                f"{self.settings.get('n_regulators')} regulators in the reference, covariates={self.settings.get('covariates')})")


def _fit_stat(Y: np.ndarray, base: np.ndarray, direction: str) -> np.ndarray:
    """``Y`` (m x n standardized rows), ``base`` (n x b standardized columns): the statistic per row used for the ranking."""
    if base.shape[1] == 1:
        r = Y @ base[:, 0]
        if direction == "abs":
            return r ** 2
        return r if direction == "positive" else -r
    coef, *_ = np.linalg.lstsq(base, Y.T, rcond=None)
    resid = Y.T - base @ coef
    return 1.0 - (resid ** 2).sum(axis=0) / (Y ** 2).sum(axis=1)


def activity_specificity(activity: pd.DataFrame, sets: Sequence[Sequence[str]], covariates: "pd.DataFrame | pd.Series | None" = None,
                         variants: Optional[Mapping[str, pd.DataFrame]] = None, direction: str = "abs", top: int = 5,
                         min_samples: int = 20) -> ActivitySpecificity:
    """How closely are the activities of a set of regulators linked, compared with the other regulators of the network?

    Parameters
    ----------
    activity : pd.DataFrame
        VIPER activity, regulators x samples, of the **whole network** (``viper_activity(..., eset_filter=False)``). All regulators
        without missing values are the reference.
    sets : sequence of sequences of str
        The sets to evaluate, e.g. ``[("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "BNC2")]``.
    covariates : DataFrame or Series, optional
        Samples (index, matching ``activity.columns``) x covariates, e.g. the tumour purity. Adds the partial correlations and the adjusted
        ranks; samples with a missing covariate are dropped from **all** statistics (``settings["n_dropped"]``).
    variants : mapping name -> DataFrame, optional
        Further activity matrices (regulators x samples) for the correlations of the pairs, e.g. ``{"candidate": ..., "pleiotropy": ...}``
        (activity from a regulon restricted to the candidates, without and with the pleiotropy correction). They need the members only.
    direction : {"abs", "positive", "negative"}
        Ranking of a *pair*: by the squared correlation (default; anti-correlated partners count), or by the positive / negative correlation.
        For sets of three or more regulators the R² of the regression is used.
    top : int
        Number of best candidates listed in ``extension["top"]``.
    min_samples : int
        Smallest number of usable samples.

    Returns
    -------
    ActivitySpecificity

    Notes
    -----
    This is descriptive: members of one module (EMT, stroma) correlate with each other and with many other regulators. The ranks show
    whether the pair is among the best partners of each other, the adjusted values whether the tie survives the covariates. The
    correlations are Spearman correlations; the adjusted ones regress the ranks of the activities on the ranks of the covariates (linear),
    which removes a monotone purity effect but not every confounder.

    Examples
    --------
    ::

        act = pr.viper_activity(expression, reg, method="mad", minsize=25, eset_filter=False)
        sp = pr.activity_specificity(act, [("ZCCHC24", "ZEB1")], covariates=purity)
        sp.table; sp.pairs; sp.extension[["set", "member", "adjusted", "rank", "n_candidates", "top"]]
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    sets = _check_sets(sets)
    variants = dict(variants or {})
    if "full" in variants:
        raise ValueError("'full' is the name of the activity matrix; use another name for a variant")
    members = sorted({t for s in sets for t in s})
    for name, mat in {"activity": activity, **variants}.items():
        missing = [m for m in members if m not in mat.index]
        if missing:
            raise ValueError(f"Regulators not in {name}: {missing}")
    samples, n_dropped = _usable(activity, list(variants.values()), covariates, min_samples)
    Z = _design(covariates, samples)
    cov_names = [] if covariates is None else ([covariates.name] if isinstance(covariates, pd.Series) else list(covariates.columns))
    A = activity.loc[:, samples]
    A = A[A.notna().all(axis=1) & (A.std(axis=1) > 0)]
    missing = [m for m in members if m not in A.index]
    if missing:
        raise ValueError(f"Regulators with missing or constant activity: {missing}")
    full = _Prepared(A, samples, Z)
    preps = {"full": full}
    for name, mat in variants.items():
        B = mat.loc[members, samples]
        if B.isna().any().any():
            raise ValueError(f"Missing values in the activity variant {name!r}")
        preps[name] = _Prepared(B, samples, Z)

    pair_rows = []
    seen, uniq = set(), []
    for s in sets:
        for p in combinations(s, 2):
            if frozenset(p) not in seen:
                seen.add(frozenset(p))
                uniq.append(p)
    for vname, prep in preps.items():
        for a, b in uniq:
            ia, ib = prep.pos[a], prep.pos[b]
            rho = float(prep.raw[ia] @ prep.raw[ib])
            rec = {"variant": vname, "tf1": a, "tf2": b, "n": prep.n, "rho": rho, "p": _corr_p(rho, prep.n - 2)}
            if prep.adj is not None:
                pr_ = float(prep.adj[ia] @ prep.adj[ib])
                rec.update(partial_rho=pr_, partial_p=_corr_p(pr_, prep.n - 2 - prep.q))
            pair_rows.append(rec)
    pairs_df = pd.DataFrame(pair_rows)

    def lookup(vname, a, b, col):
        r = pairs_df[(pairs_df["variant"] == vname) & (((pairs_df["tf1"] == a) & (pairs_df["tf2"] == b)) | ((pairs_df["tf1"] == b) & (pairs_df["tf2"] == a)))]
        return float(r[col].iloc[0])

    rows, ext_rows = [], []
    n_ref = len(A)
    adjusted_modes = [False] + ([True] if full.adj is not None else [])
    for tfs in sets:
        label = " | ".join(tfs)
        rec = {"set": label, "n_samples": full.n}
        pw = [lookup("full", a, b, "rho") for a, b in combinations(tfs, 2)]
        rec["rho_mean"], rec["abs_rho_min"] = float(np.mean(pw)), float(np.min(np.abs(pw)))
        if full.adj is not None:
            pp = [lookup("full", a, b, "partial_rho") for a, b in combinations(tfs, 2)]
            rec["partial_rho_mean"], rec["partial_abs_rho_min"] = float(np.mean(pp)), float(np.min(np.abs(pp)))
        for adjusted in adjusted_modes:
            Y = full.adj if adjusted else full.raw
            ranks, pcts = [], []
            for m, tf in enumerate(tfs):
                others = [t for t in tfs if t != tf]
                base = Y[[int(full.pos[o]) for o in others]].T
                stat = _fit_stat(Y, base, direction)
                cand = np.ones(n_ref, dtype=bool)
                cand[[int(full.pos[o]) for o in others]] = False
                ci = np.flatnonzero(cand)
                me = int(full.pos[tf])
                sm = stat[me]
                rank = 1 + int(np.sum(stat[ci] > sm + 1e-12))
                order = ci[np.argsort(-stat[ci])[:top]]
                single = base.shape[1] == 1
                if single and direction == "abs":
                    r1 = Y @ base[:, 0]
                    shown = {i: r1[i] for i in order}
                    stat_me = float(Y[me] @ base[:, 0])
                elif single:
                    shown = {i: stat[i] * (1 if direction == "positive" else -1) for i in order}
                    stat_me = float(sm * (1 if direction == "positive" else -1))
                else:
                    shown, stat_me = {i: stat[i] for i in order}, float(sm)
                ext_rows.append({"set": label, "member": tf, "base": " & ".join(others), "adjusted": adjusted, "rank": rank,
                                 "n_candidates": len(ci), "percentile": rank / len(ci), "stat": stat_me,
                                 "top": ";".join(f"{full.names[i]}({shown[i]:.2f})" for i in order)})
                ranks.append(rank)
                pcts.append(rank / len(ci))
            pre = "padj_" if adjusted else ""
            rec[pre + "ext_rank_max"] = float(max(ranks))
            rec[pre + "ext_percentile_max"] = float(max(pcts))
            rec[pre + "mutual_best"] = bool(all(r == 1 for r in ranks))
        rows.append(rec)
    table = pd.DataFrame(rows)
    ext = pd.DataFrame(ext_rows)
    settings = {"direction": direction, "n_samples": int(full.n), "n_dropped": int(n_dropped), "covariates": cov_names,
                "n_regulators": int(n_ref), "variants": ["full"] + list(variants)}
    return ActivitySpecificity(pairs=pairs_df, table=table, extension=ext, settings=settings)


def activity_partners(activity: pd.DataFrame, base: "str | Sequence[str]", covariates: "pd.DataFrame | pd.Series | None" = None,
                      direction: str = "abs", top: Optional[int] = 20, min_samples: int = 20) -> pd.DataFrame:
    """The regulators whose activity is most closely linked to the activity of ``base`` (one regulator or several): every regulator of
    ``activity`` ranked by R² (one base regulator: squared correlation, or signed with ``direction``) of the regression on the base,
    adjusted for the covariates if given. Columns ``regulator``, ``stat`` (correlation for one base regulator, else R²), ``rank``."""
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    base = [base] if isinstance(base, str) else [str(b) for b in base]
    missing = [b for b in base if b not in activity.index]
    if missing:
        raise ValueError(f"Regulators not in the activity matrix: {missing}")
    samples, _ = _usable(activity, [], covariates, min_samples)
    A = activity.loc[:, samples]
    A = A[A.notna().all(axis=1) & (A.std(axis=1) > 0)]
    prep = _Prepared(A, samples, _design(covariates, samples))
    Y = prep.adj if prep.adj is not None else prep.raw
    bi = [int(prep.pos[b]) for b in base]
    stat = _fit_stat(Y, Y[bi].T, direction)
    shown = Y @ Y[bi[0]] if len(bi) == 1 else stat
    keep = np.ones(len(A), dtype=bool)
    keep[bi] = False
    ci = np.flatnonzero(keep)
    order = ci[np.argsort(-stat[ci])]
    out = pd.DataFrame({"regulator": prep.names[order], "stat": shown[order]})
    out["rank"] = np.arange(1, len(out) + 1)
    return out if top is None else out.head(top)
